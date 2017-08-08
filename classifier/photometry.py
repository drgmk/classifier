'''Functions and scripts for machine learning training and 
classification of photometry and spectra.'''

import os
import shutil
from functools import lru_cache
import itertools
import pickle
import argparse

import numpy as np
import requests
from astropy.table import Table
import astropy.units as u
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

import sdf.filter
import sdf.spectrum
import sdf.db
import sdf.photometry

from . import utils
from . import config as cfg

# routines related to photometry

def download_photometry(sdbid):
    """Get and return an sdf.Photometry object for a given sdbid.
    
    Once retrieved, files are stored in photometry training directory.

    Parameters
    ----------
    sdbid : string
        sdb id of desired target.
        
    Returns
    -------
    sdf.Photometry object.
    """
    
    fname = sdbid+'-rawphot.txt'
    dir = cfg.training_dir+'photometry/photometry_files/'
    try:
        file = '/Users/grant/a-extra/sdb/masters/'+sdbid+'/public/'+fname
        p = sdf.photometry.Photometry.read_sdb_file(file)

    except FileNotFoundError:
        try:
            file = dir+fname
            p = sdf.photometry.Photometry.read_sdb_file(file)

        except FileNotFoundError:
            r = requests.get('http://drgmk.com/sdb/seds/masters/'+
                sdbid+'/public/'+fname,auth=(cfg.sdb_user,cfg.sdb_pass))

            with open(dir+fname,'w') as f:
                f.write(r.text)

            p = sdf.photometry.Photometry.read_sdb_file(dir+fname)

    # copy the file to the training dir
    if not os.path.exists(dir+fname):
        shutil.copyfile(file,dir+fname)

    return p


@lru_cache()
def interpolator(colour_name,out_column='Teff'):
    """Return interpolator to get a value from the spectral type table.
    
    Interpolation uses the colour table from Eric Mamajek's website.
    Various derived columns are added so that as many input colours as
    possible can be used.
    
    Parameters
    ----------
    colour_name : string
        Name of the colour used for the interpolator.
    out_column : string, optional
        Name of column to retrieve, effective temperature by default.
        
    Returns
    -------
    A scipy interp1d interpolator object.
    """
    
    # assume eem means H_Ks when he says H_K
    t = Table.read(os.path.dirname(os.path.abspath(__file__))+'/eem_colours.txt',
                   format='ascii',fill_values=[('...',0),('....',0),('.....',0)])
       
    t['Rc-Ic'] = t['V-Ic'] - t['V-Rc']
    t['Ks-W3'] = t['Ks-W1'] + t['W1-W3']
    t['Ks-W4'] = t['Ks-W1'] + t['W1-W4']
    t['W2-W3'] = t['W1-W3'] - t['W1-W2']
    t['W3-W4'] = t['W1-W4'] - t['W1-W3']
    
    # conversion from sdf filter names to eem ones
    c = {'UJ_BJ':'U-B', 'BJ_VJ':'B-V', 'BT_VT':'Bt-Vt', 'VJ_RC':'V-Rc',
         'VJ_IC':'V-Ic', 'RC_IC':'Rc-Ic',
         'VJ_2MKS':'V-Ks', '2MJ_2MH':'J-H', '2MH_2MKS':'H-K',
         '2MKS_WISE3P4':'Ks-W1', '2MKS_WISE12':'Ks-W3',
         '2MKS_WISE22':'Ks-W4', 'WISE3P4_WISE4P6':'W1-W2',
         'WISE3P4_WISE12':'W1-W3', 'WISE3P4_WISE22':'W1-W4',
         'WISE4P6_WISE12':'W2-W3', 'WISE12_WISE22':'W3-W4',
         'Teff':'Teff'}
    
    if colour_name not in c.keys():
        return None
    
    # out of range interpolation yields cool and hot temperatures,
    # otherwise zero colour (i.e. Vega)
    if out_column == 'Teff':
        fill = (30000.0,1500.0)
    else:
        fill = 0.0
        
    ok = np.logical_and( t[c[colour_name]].mask == False,
                         t[c[out_column]].mask == False )
    
    return interp1d(t[c[colour_name]][ok], t[c[out_column]][ok],
                    kind='linear',bounds_error=False,fill_value=fill)


class Colours(object):
    """Class to sort out colours for given photometry.
    
    The goal is to return a list of colours for a given target, which 
    can then be used for the machine learning. Where the desired colours
    don't exist they are either inferred from something close, or 
    interpolated based on an estimated effective temperature. Where very
    little information exists the colours default to zero, meaning that
    an object is by default assumed to be a star, and that a 
    classification is always possible.
    
    There are various quirks of this approach.

    - Stars bright enough to be detected at far-IR wavelengths are so
      bright that no useful WISE W1/2 photometry exists. Therefore very
      few stars with 'complete' photometry that can be derived without
      interpolation exist.

    Parameters
    ----------
    phot : sdf.photometry.Photometry
        Photometry object
    wanted : list, optional
        The colours we want returned.
    extras : list, optional
        Colours that may be useful in deriving what we want.
    """

    def __init__(self,phot=None,
                 wanted=['2MJ_2MH','2MH_2MKS','2MKS_WISE3P4',
                         'WISE3P4_WISE4P6','WISE4P6_WISE12','WISE12_WISE22',
                         'KP_WAV100','WISE22_WAV100'],
                 extras=['2MKS_WISE12','2MKS_WISE22',
                         'WISE3P4_WISE12','WISE3P4_WISE22']):

        self.wanted = wanted

        if phot is not None:

            # swap Read numbers in 2MASS for "plain" (2MR1KS -> 2MKS)
            for i,f in enumerate(phot.filters):
                if '2MR' in f:
                    phot.filters[i] = f[0:2]+f[4:]
        
            self.extras_in = extras
            ok = np.logical_and(phot.ignore == 0,phot.upperlim == 0)
            self.in_names = phot.filters[ok]
            self.in_values = phot.measurement[ok]
            self.in_units = phot.unit[ok]

            self.in_dict = {}
            for key,val in zip(self.in_names,self.in_values):
                self.in_dict[key] = val

            # we'll fill these
            self.colours = {}
            self.extras = {}
            self.filled = {}
            # do it
            self.fill_wav100()   
            self.fill_wise()
            self.fill_kp()
            self.fill_extras()
            self.fill_colours()
            self.fill_missing()
        
        
    def fill_colours(self):
        """Fill missing colours that we want.
        
        These are assumed to be sufficiently accurate, as opposed to
        guessing or interpolating, that the values in the self.colours
        dictionary are considered not to have been 'filled'.
        """
        
        # if colours given
        for i,name in enumerate(self.in_names):
            if name in self.wanted:
                self.colours[name] = self.in_values[i]
                self.filled[name] = False
                
        # if components of colours given
        for name in self.wanted:
            c = sdf.filter.Colour.get(name)
            if c.filters[0] in self.in_names and c.filters[1] in self.in_names:
                self.colours[name] = self.in_dict[c.filters[0]] - \
                                     self.in_dict[c.filters[1]]
                self.filled[name] = False
        
        # attempt to fill BJ_VJ if not present, this doesn't count as filled
        if 'BJ_VJ' in self.wanted and 'BJ_VJ' not in self.colours.keys():

            # Tycho -> Johnson, 2002AJ....124.1670M
            if 'BT' in self.in_names and 'VT' in self.in_names:
                BT = self.in_dict['BT']
                VT = self.in_dict['VT']
                self.colours['BJ_VJ'] = (BT - VT) - 0.006 - \
                                        0.1069*(BT - VT) + \
                                        0.1459*(BT-VT)**2
                self.filled['BJ_VJ'] = False

            # or from APASS 1996AJ....111.1748F
            elif 'GAPASS' in self.in_names and 'RAPASS' in self.in_names:
                g_r = self.in_dict['GAPASS'] - self.in_dict['RAPASS']
                VJ = self.in_dict['RAPASS'] + 0.44 * g_r - 0.02
                BJ = VJ + 1.04 * g_r + 0.19
                self.colours['BJ_VJ'] = BJ - VJ
                self.filled['BJ_VJ'] = False
            
    
    def fill_extras(self):
        """Fill missing extras from what we have, if possible."""
        
        # if components of colours given
        for name in self.extras_in:
            c = sdf.filter.Colour.get(name)
            if c.filters[0] in self.in_names and c.filters[1] in self.in_names:
                self.extras[name] = self.in_dict[c.filters[0]] - \
                                    self.in_dict[c.filters[1]]
        
        
    def fill_missing(self):
        """Interpolate missing colours.
        
        These are considered to have been 'filled', as they are guessed
        (for zero colour) or interpolated.
        """
        for name in self.wanted:
            
            # if this key exists we already have the colour
            if name in self.colours.keys():
                continue

            # zero colour at long wavelengths
            elif name == 'WISE22_WAV100' or name == 'KP_WAV100':
                self.colours[name] = 0.0
                self.filled[name] = True
                
            # else interpolate based on median Teff from what we have,
            # setting zero colour if interpolation is not possible
            else:
                f = interpolator('Teff',name)
                med_temp = self.med_temp()
                self.filled[name] = True
                if np.isfinite(med_temp):
                    self.colours[name] = f( med_temp ).tolist() # a float
                else:
                    self.colours[name] = 0.0

        
    def sorted_colours(self):
        """Return colours sorted by mean wavelength."""
        names = list(self.colours.keys())
        waves = []
        for c in names:
            col = sdf.filter.Colour.get(c)
            waves.append( col.mean_wavelength )
            
        srt = np.argsort(waves)
        
        cols = []
        for i in srt:
            cols.append(self.colours[names[i]])

        if not np.all(np.isfinite(cols)):
            print("Not all colours finite ({})".format(cols))
            return None

        if len(cols) != len(self.wanted):
            print("Didn't get all wanted colours")
            return None
            
        return cols
    
    
    def med_temp(self):
        """Return the median temperature for defined colours."""
        temp = []
        
        for key in self.extras.keys():
            f = interpolator(key)
            if f is not None:
                temp.append( f(self.extras[key]) )

        for key in self.colours.keys():
            f = interpolator(key)
            if f is not None and not self.filled[key]:
                temp.append( f(self.colours[key]) )

        if len(temp) > 0:
            med_t = np.nanmedian(temp)
        else:
            med_t = np.nan
            
        return med_t
        
        
    def fill_wise(self):
        """Fill WISE mags from Spitzer and AKARI.
        
        These are assumed to be sufficiently accurate that they are 
        considered not to have been 'filled'.
        """

        # IRAC fluxes in uJy
        if 'WISE3P4' not in self.in_names and 'IRAC3P6' in self.in_names:
            f = sdf.filter.Filter.get('IRAC3P6')
            mag36 = f.flux2mag(self.in_dict['IRAC3P6']/1e6)
            self.in_names = np.append(self.in_names,'WISE3P4')
            self.in_values = np.append(self.in_values,mag36)
            self.in_dict['WISE3P4'] = mag36
            self.filled['WISE3P4'] = False

        if 'WISE4P6' not in self.in_names and 'IRAC4P5' in self.in_names:
            f = sdf.filter.Filter.get('IRAC4P5')
            mag45 = f.flux2mag(self.in_dict['IRAC4P5']/1e6)
            self.in_names = np.append(self.in_names,'WISE4P6')
            self.in_values = np.append(self.in_values,mag45)
            self.in_dict['WISE4P6'] = mag45
            self.filled['WISE4P6'] = False

        # AKARI fluxes in Jy
        if 'WISE12' not in self.in_names and 'AKARI9' in self.in_names:
            f = sdf.filter.Filter.get('AKARI9')
            mag9 = f.flux2mag(self.in_dict['AKARI9'])
            self.in_names = np.append(self.in_names,'WISE12')
            self.in_values = np.append(self.in_values,mag9)
            self.in_dict['WISE12'] = mag9
            self.filled['WISE12'] = False
        
        # MIPS 24 fluxes in uJy
        if 'WISE22' not in self.in_names and 'MIPS24' in self.in_names:
            f = sdf.filter.Filter.get('MIPS24')
            mag24 = f.flux2mag(self.in_dict['MIPS24']/1e6)
            self.in_names = np.append(self.in_names,'WISE22')
            self.in_values = np.append(self.in_values,mag24)
            self.in_dict['WISE22'] = mag24
            self.filled['WISE22'] = False

        # AKARI fluxes in Jy
        if 'WISE22' not in self.in_names and 'AKARI18' in self.in_names:
            f = sdf.filter.Filter.get('AKARI18')
            mag18 = f.flux2mag(self.in_dict['AKARI18'])
            self.in_names = np.append(self.in_names,'WISE22')
            self.in_values = np.append(self.in_values,mag18)
            self.in_dict['WISE22'] = mag18
            self.filled['WISE22'] = False

        # IRAS fluxes in Jy
        if 'WISE12' not in self.in_names and 'IRAS12' in self.in_names:
            f = sdf.filter.Filter.get('IRAS12')
            mag12 = f.flux2mag(self.in_dict['IRAS12'])
            self.in_names = np.append(self.in_names,'WISE12')
            self.in_values = np.append(self.in_values,mag12)
            self.in_dict['WISE12'] = mag12
            self.filled['WISE12'] = False

        # IRAS fluxes in Jy
        if 'WISE22' not in self.in_names and 'IRAS25' in self.in_names:
            f = sdf.filter.Filter.get('IRAS25')
            mag25 = f.flux2mag(self.in_dict['IRAS25'])
            self.in_names = np.append(self.in_names,'WISE22')
            self.in_values = np.append(self.in_values,mag25)
            self.in_dict['WISE22'] = mag25
            self.filled['WISE22'] = False

        
    def fill_wav100(self,bands=['IRAS60','IRAS100','MIPS70','PACS70','PACS100']):
        """Derive a WAV100 magnitude, using the given bands.
        
        Parameters
        ----------
        bands : list, optional
            Bands that can be used to derive the WAV100 magnitude.
        """
        f = []
        for i,name in enumerate(self.in_names):
            ok = np.where(name == np.array(bands))[0]
            if len(ok) > 0:
                filt = sdf.filter.Filter.get(name)
                if self.in_values[i] <= 0:
                    continue
                flux_jy = (self.in_values[i]*self.in_units[i]).to('Jy').value
                wave = np.linspace(25,150,50)
                _,cc = filt.synthphot(
                    sdf.spectrum.ModelSpectrum(wavelength=wave,
                                               fnujy_sr=1/wave**2))
                f.append( filt.flux2mag(flux_jy) )
                
        if len(f) > 0:
            wav100 = np.median(f)
            self.in_names = np.append(self.in_names,'WAV100')
            self.in_values = np.append(self.in_values,wav100)
            self.in_dict['WAV100'] = wav100


    def fill_kp(self,bands=['VJ','VT','HP','2MJ','2MH','2MKS','WISE3P4']):
        """Derive a 'Kepler' magnitude using the given bands.

        This photometry is to have an optical magnitude to create the
        KP-WAV100 colour. KP is chosen because no observed photometry
        from this band exists (in sdb).

        Parameters
        ----------
        bands : list, optional
            Bands that can be used to derive the KP magnitude.
        """
        if 'KP' not in self.in_dict.keys():
            for f in bands:
                if f in self.in_dict.keys():
                    self.in_names = np.append(self.in_names,'KP')
                    self.in_values = np.append(self.in_values,self.in_dict[f])
                    self.in_dict['KP'] = self.in_dict[f]


    def skipped(self):
        """Return a boolean indicating if this object should be skipped.
        
        Conditions to be skipped are:
        - no effective temperature estimate
        - more than two 'filled' colours (>1 missing phot)
        """

        if not np.isfinite(self.med_temp()):
            return True
        elif np.sum(list(self.filled.values())) > 2:
            return True
        else:
            return False


def get_data(files):
    """Get and prepare input data given a list of label files.
    
    Files should have sdbid and label columns. The photometry files are
    obtained by download_data. Whether a file should be skipped is
    decided in colours.skipped.

    Parameters
    ----------
    files : list
        List of csv files, first column sdbid name, second label
        
    Returns
    -------
    A tuple of:
     - data [n_data,n_colurs]
     - one hot array of labels [n_data,n_classes]
     - sdbids [n_data]
     - skip [n_data] (whether this file should be skipped)
     - label names [n_classes]
    """
    
    sdbids = []
    data = []
    labels = []
    skip = []

    # go through all files to get labels
    label_names = np.array([])
    for file in files:
        t = Table.read(file,comment='#')
        label_names = np.append(label_names, 
                                np.unique(t['label'].tolist()) )

#    label_names = np.unique(label_names)
    # we know the names, so do them in this order
    label_names = ['class I','class II','transition','debris','star']

    n_labels = len(label_names)
    print('Classes: {}'.format(label_names))
    l_dict = dict(zip(label_names,range(n_labels)))
    
    for file in files:

        t = Table.read(file,comment='#')
        if 'sdbid' not in t.colnames:
            t['sdbid'] = sdf.db.get_sdbids(t['name'].tolist())

        for (sdbid,label) in zip(t['sdbid'],t['label']):
            p = download_photometry(sdbid)
            c = Colours(phot=p)

            skip.append( int(c.skipped()) )
            sdbids.append(sdbid)
            data.append(c.sorted_colours())

            onehot = np.zeros(n_labels)
            onehot[l_dict[label]] = 1
            labels.append(onehot)

    # remove duplicate sdbids
    n_in = len(sdbids)
    _,keep = np.unique(sdbids,return_index=True)
    data   = np.array(data)[keep]
    labels = np.array(labels,dtype=int)[keep]
    sdbids = np.array(sdbids)[keep]
    skip = np.array(skip,dtype=bool)[keep]
    print('Kept',len(sdbids),'of',n_in,'(duplicates discarded)')
            
    return (data,labels,sdbids,skip,label_names)


def split_data(sdbids,data,labels,test_train=0.5,shuffle=True):
    """Split data based on labels.
    
    Aside from also splitting the sdbids, this could be done
    with sklearn.model_selection.train_test_split.
    
    Parameters
    ----------
    sdbids : list
        List of sdbids.
    data : numpy.ndarray
        Array of measurements for each sdbid.
    labels : numpy.ndarray
        One hot array of labels.
    test_train : float, optional
        Fraction to retain in training set.
    shuffle : bool, optional
        Whether to shuffle data before splitting or not.
    """
    
    n_points = data.shape[1]
    n_labels = labels.shape[1]
    n_each = np.sum(labels,axis=0)
    n_train = np.array(test_train * n_each,dtype=int)
    n_test = n_each - n_train

    # shuffle contents
    if shuffle:
        rnd_state = np.random.get_state()
        np.random.shuffle(sdbids)
        np.random.set_state(rnd_state)
        np.random.shuffle(data)
        np.random.set_state(rnd_state)
        np.random.shuffle(labels)

    train = np.zeros(data.shape[0],dtype=bool)
    for i in range(n_labels):
        n = 0
        j = 0
        while n < n_train[i]:
            if labels[j,i] == 1:
                train[j] = 1
                n += 1
            j += 1
            
    test = np.invert(train)
    
    sdbids_train = sdbids[train]
    data_train = data[train,:]
    labels_train = labels[train,:]
    sdbids_test = sdbids[test]
    data_test = data[test,:]
    labels_test = labels[test,:]
     
    if not np.all(np.sum(labels_train,axis=0)==n_train):
        print("ERROR: train labels doesn't sum to n_train")
    if not np.all(np.sum(labels_test,axis=0)==n_test):
        print("ERROR: test labels doesn't sum to n_test")
        
    print('Initial:',data.shape[0],'->','Train:',n_train,np.sum(n_train),
          '| Test:',n_test,np.sum(n_test))
    
    return (sdbids_train,data_train,labels_train,
            sdbids_test,data_test,labels_test)


def model_success(pred,sdbids,label_names,label_nums):
    """Print the failures and success rate of a set of predictions."""
    fail = 0
    for i,p in enumerate(pred):
        if p != label_nums[i]:
            print(i,sdbids[i],label_names[p],label_names[label_nums[i]])
            fail += 1
    print("This set: {}/{}={:4.1f}% failed".format(fail,len(sdbids),
                                                   100*fail/len(sdbids)))


def predict_phot(phot_file):
    '''Predict class for a given photometry file.
    
    Parameters
    ----------
    phot_file : string
        Path to photometry file to be classified.
    '''

    pickle_dir = os.path.dirname(os.path.abspath(__file__))

    p = sdf.photometry.Photometry.read_sdb_file(phot_file)
    c = Colours(phot=p)

    if c.skipped():
        labels, clf = utils.get_clf(
                    pickle_dir+'/phot-nn-partial-classifier.pkl'
                                    )
    else:
        labels, clf = utils.get_clf(
                    pickle_dir+'/phot-nn-full-classifier.pkl'
                                    )

    data = np.array(c.sorted_colours()).reshape(1, -1)
    pred = clf.predict(data)[0]

    return labels[pred].lower()


def predict_phot_shell():
    '''Shell script to run photometry classification.'''

    # inputs
    parser = argparse.ArgumentParser(description='Photometry classifier')
    parser.add_argument('--file','-f',nargs='+',action='append',
                        help='Classify rawphot file or files')
    args = parser.parse_args()

    if args.file is not None:
        files = args.file[0]

    for f in files:

        print('Classifying file: {}'.format(os.path.basename(f)))
        print(predict_phot(f))
        
