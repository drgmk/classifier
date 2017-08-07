# coding: utf-8

'''Classify IRS spectra.

Create a widget that allows selection of some buttons to classify IRS
spectra.

A fair amount of by-hand fiddling below depending on what is being done.
'''

import glob
import os
import shutil
import pickle
import json
import numpy as np
from scipy import optimize
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons
from matplotlib import gridspec
from astropy.table import Table
import astropy.coordinates as coord
import astropy.units as u
from astropy.io import fits
from astroquery.simbad import Simbad

import sdf.spectrum
import sdf.utils
import classifier.spectra
import classifier.config as cfg

def fit_func(par,x):
    '''Function to fit a spectrum.'''
    return par[0] / x**2 + par[1] * x**par[2] + \
            par[3] * sdf.utils.bnu_wav_micron(x,par[4])


def err_func(par,x,y):
    '''Difference between model and data.'''
    return fit_func(par,x) - y


def classify_one(file,labels,labels_init):
    """Show a plot to classify a spectrum, returning the labels."""

    if 'fits' in file:
        s = sdf.spectrum.ObsSpectrum.read_cassis(file,module_split=False)
        s_split = sdf.spectrum.ObsSpectrum.read_cassis(file,module_split=True)
    elif 'tbl' in file:
        t = Table.read(file,format='ascii.ipac')
        s = [sdf.spectrum.ObsSpectrum(wavelength=t['wavelength'],
                                      fnujy=t['flux_density'])]
        s_split = s
    else:
        raise TypeError('wrong file type')

    fig = plt.figure(figsize=(9,6))
    gs = gridspec.GridSpec(2,1,height_ratios=[3,1])
    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1],sharex=ax0)
    ax = [ax0,ax1]
#    ax = [plt.subplot(g) for g in gs]
    fig.subplots_adjust(right=0.8)
    fig.subplots_adjust(top=0.9)
    ax[0].set_title(file)

    # fit the model to the whole spectrum
    best,_ = optimize.leastsq(err_func,[1,1,1,1e-17,150.],
                              args=(s[0].wavelength,s[0].fnujy))
    # Rayleigh-Jeans tail for guidance
    ax[0].plot(s[0].wavelength,
               np.mean(s[0].fnujy[:10]*s[0].wavelength[:10]**2)/s[0].wavelength**2,':')

    # each module in the spectrum
    for spec in s_split:

        # the spectrum
        ax[0].loglog(spec.wavelength,spec.fnujy)

        fit = fit_func(best,spec.wavelength)
        ax[0].plot(spec.wavelength,fit)

        # plot of residuals
        ax[1].semilogx(spec.wavelength,spec.fnujy-fit)
        ax[1].semilogx(spec.wavelength,np.zeros(len(spec.wavelength)),'--')

    # check buttons for labelling
    stop = False
    n_but = len(labels)
    sp = 0.035
    rax = plt.axes([0.81, 0.99-sp*n_but, 0.19, sp*n_but])
    label_button = CheckButtons(rax, labels, labels_init)
    nextax = plt.axes([0.81, 0.05+sp, 0.19, sp])
    next_button = CheckButtons(nextax, ['next'], [False])
    stopax = plt.axes([0.81, 0.05, 0.19, sp])
    stop_button = CheckButtons(stopax, ['stop'], [False])

    def click_func(label):
        for i,l in enumerate(labels):
            if label == l:
                labels_init[i] = not labels_init[i]
        
    def next_func(label):
        plt.close(fig)

    def stop_func(label):
        plt.close(fig)
        print('stopping')
        exit()

    label_button.on_clicked(click_func)
    next_button.on_clicked(next_func)
    stop_button.on_clicked(stop_func)

    # add some lines to help classification
    line_names = ['$H_2O$','$H_2O$','$CO_2$','Enstatite','Forsterite','$H_2$']
    line_waves = [ 6.0,     9.0,     15.2,     9.2,        11.3,        28.2]
    line_names += ['$H_2$','$H_2$','[Nev]','Forsterite','Silica & HI','C23','C28']
    line_waves += [ 9.66,   6.1,    24.32,  33.6,        12.5,         23.,  28. ]
    line_names += ['PAH','PAH','PAH','PAH','PAH','AmSil & HI','HI','PAH']
    line_waves += [ 6.2,  7.7,  8.6,  11.2, 12.7, 19.,         7.5, 17.1 ]
    for i,a in enumerate(ax):
        ylim = a.get_ylim()
        for n,w in zip(line_names,line_waves):
            a.plot([w,w],ylim,':',alpha=0.5)
            if i == 0:
                a.text(w,ylim[0],n,va='bottom',rotation='vertical')

    ax[0].set_xlim(5,38)
    plt.show()

    if stop:
        return None
    else:
        return labels_init


# start of the script fo real

# a dict to put classifications in
data = {}
class_file = 'irs_labels_cassis.txt'
with open('irs_labels_cassis.txt','r') as f:
    data = json.load(f)

# decide which files we want to look at
files = glob.glob(cfg.spectra + 'training/*fits',recursive=True)
files = [os.path.basename(f) for f in files]
#files = list( data.keys() )
print('{} files'.format(len(files)))

# the labels to use
label_names = np.array(['Class I','Class II','Transition',
                        'Kuiper','Star','Be Star',
                        'Am Sil Em','Cryst Sil Em',
                        'Am Sil Abs','Ice Abs','Gas Abs',
                        'PAH Em','Gas Em','[NeV]',
                        'O-type','B-type','A-type','F-type','G-type',
                        'K-type','M-type','Brown Dwarf',
                        'SL/LL offset'])

# check we have all labels
label_counts = {}
for k in label_names:
    label_counts[k] = 0

for k in data.keys():
    for l in data[k]:
        label_counts[l] += 1
        if l not in label_names:
            raise ValueError('label {} not in label list ({})'.format(l,k))

print('Read {} spectra'.format(len(data)))
for l in label_counts.keys():
    print(l,label_counts[l])


for file in files:

    file = cfg.spectra + 'irsstare/' + file
    file_name = os.path.basename(file)

    # store files in here
#    store = 'spectra/training/'
#    if not os.path.exists(store+file_name):
#        shutil.copy(file,store)

    # look for spectra that have specific predictions
    pred = classifier.spectra.predict_spectra_class(file)
    print('Pred : {}'.format(pred))
#    if pred == 'star':
#        continue

    # read in, skip according to rules in read_spec
    s_learn = classifier.spectra.read_spec(file)
    if s_learn is None:
        print('skipping {}'.format(file_name))
        continue

    # initial statuses of the labels, which we might change below
    labels_init = np.zeros(len(label_names),dtype=bool)

    if file_name in data.keys():
        
        # store labeled files in here
        store = cfg.spectra + 'training/'
        if not os.path.exists(store+file_name):
            shutil.copy(file,store)

        # skip ones we did already
        if len(data[file_name]) > 0:
            print('Already did {}, with {}'.format(file_name,
                                                   data[file_name]))
#            continue

        # or only do ones we want to re-do
        if 'Cryst Sil Em' not in data[file_name]:
            continue
#        skip = 0
#        for label in data[file_name]:
#            if 'type' in label:
#                skip = 1
#        if skip:
#            continue

        # fill initial labels from file
        for i,label in enumerate(label_names):
            if label in data[file_name]:
                labels_init[i] = True

    # attempt to grab some info from simbad
    res = None
    if 'fits' in file:
        head = fits.getheader(file)
        print('Object: {}, at {},{}'.format(head['OBJECT'],head['RA_RQST'],
                                            head['DEC_RQST']))
        custSimbad = Simbad()
        custSimbad.remove_votable_fields('coordinates')
        custSimbad.add_votable_fields('sp','otype')
        custSimbad.SIMBAD_URL = u'http://simbad.harvard.edu/simbad/sim-script'
        res = custSimbad.query_region(coord.SkyCoord(head['RA_RQST'], head['DEC_RQST'],
                                      unit=(u.deg, u.deg)),radius='0d0m10s')
        print(res)

    # and add to initial labels
    if res is not None and 0:
        sp_type = res['SP_TYPE'][0].decode()
        if len(sp_type) > 0:
            sp_type = sp_type[0]
            for i,l in enumerate(label_names):
                if 'type' in l:
                    if sp_type == l[0]:
                        labels_init[i] = True
            if sp_type == ['L','T','Y']:
                labels_init[np.where(label_names == 'Brown Dwarf')] = True

        if 'BD*' in res['OTYPE'][0].decode():
            labels_init[np.where(label_names == 'Brown Dwarf')] = True

        if 'Be*' in res['OTYPE'][0].decode():
            labels_init[np.where(label_names == 'Be Star')] = True
            labels_init[np.where(label_names == 'Gas Em')] = True

    print('Initial: {}'.format(np.array(labels_init,dtype=int)))

    # do the classification and put it in the dict
    out = classify_one(file,label_names,labels_init)
    if out is None or np.sum(out) == 0:
        print('Skipping {}'.format(file_name))
        continue

    labels_tmp = label_names[out]
    data[file_name] = labels_tmp.tolist()

    # save it quick
    with open(class_file,'w') as f:
        json.dump(data,f)

    print(file_name,labels_tmp)

print('final number of spectra: {}'.format(len(data.keys())))

# save it
with open(class_file,'w') as f:
    json.dump(data,f)
