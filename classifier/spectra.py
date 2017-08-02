import os
import json
import argparse
import numpy as np
import scipy.ndimage.filters

import sdf.spectrum

from . import config as cfg
from . import utils

def read_spec(f,norm=False,median=None):
    '''Read a CASSIS IRS spectrum file and return an array.
    
    Conditions that set whether to skip a spectrum are hard coded below.

    Parameters
    ----------
    f : string
        File name
    norm : bool, optional
        Normalise each module to the median
    '''
    sp = sdf.spectrum.ObsSpectrum.read_cassis(f,module_split=True)

    # require all four modules
    if len(sp) < 4:
        return

    arr = np.array([])
    sn = np.array([])
    wave = np.array([])
    for s in sp:
        ok = s.wavelength < 99.0
        sn = np.append(sn,s.fnujy[ok]/s.e_fnujy[ok])
        wave = np.append(wave,s.wavelength[ok])
        if norm:
            arr = np.append(arr,s.fnujy[ok]/np.mean(s.fnujy))
        else:
            arr = np.append(arr,s.fnujy[ok])

    # ignore spectra without this length
    if len(arr) != 382:
        return

    # smooth or otherwise modify the spectrum
    if median is not None:
        arr = scipy.ndimage.filters.median_filter(arr,size=median)

    # this doesn't make any significant difference
    arr /= np.mean(arr)

    return arr,sn,wave


def get_data(label_file='irs_labels_cassis.txt',label_skip=[],
             norm=False,median=None,sn_cut=0.0):
    '''Read in data given a file with spectra and labels.
    
    Parameters
    ----------
    label_file : string, optional
        File with labels.
    label_skip : list, optional
        List of labels to skip.
    norm : bool, optional
        Normalise individual modules.
    median : int or None, optional
        Apply median filter to spectra.
    sn_cut : float, optional
        Only keep spectra with average S/N higher than this.
    '''

    with open(label_file,'r') as f:
        data = json.load(f)

    # the label order we desire, all labels in label_file must be here
    label_order  = [
    'Class I','Class II','Transition','Kuiper','Star','Be Star',
    'O-type','B-type','A-type','F-type','G-type','K-type','M-type','Brown Dwarf',
    'Am Sil Em','Cryst Sil Em','PAH Em','Gas Em',
    'Am Sil Abs','Ice Abs','Gas Abs',
    '[NeV]',
    'SL/LL offset'
                    ]

    files = []
    spectra = []
    sn = []
    labels_onehot = []

    # loop through files and get spectra
    for i,file in enumerate(data.keys()):

        # read file and optionally reject
        tmp = read_spec(cfg.spectra+'training/'+file,
                        norm=norm,median=median)
        if tmp is None:
            print('Skipping {}'.format(file))
            continue
        else:
            tmp,tmp1,_ = tmp
            # low mean s/n criterion 
            if np.mean(tmp1) < sn_cut:
                continue

        # append to spectra, s/n, and file name arrays
        spectra.append(tmp)
        sn.append(tmp1)
        files.append(file)

    # now loop to get labels
    for file in files:
        onehot_tmp = np.zeros(len(label_order))
        for label in data[file]:
            for i,lab in enumerate(label_order):
                if lab == label and label not in label_skip:
                    onehot_tmp[i] = 1
        labels_onehot.append(onehot_tmp)

    labels_onehot = np.array(labels_onehot)
    spectra = np.array(spectra)
    sn = np.array(sn)

    # remove empty label columns (no labels, or skipped)
    ok = np.sum(labels_onehot,axis=0).astype(bool)
    labels_onehot = labels_onehot[:,ok]
    label_names = np.array(label_order)[ok]

    return spectra,sn,files,label_names,labels_onehot


def split_data(data,labels,train_fraction=0.7):
    '''Split into test/train by randomising.'''

    train_fraction = 0.7
    nspec = data.shape[0]
    n_train = int(nspec*train_fraction)
    n_test = nspec - n_train
    train_i = np.zeros(nspec,dtype=bool)
    train_i[np.random.choice(nspec,n_train,replace=False)] = True
    test_i = np.invert(train_i)

    data_train = data[train_i]
    labels_train = labels[train_i]
    data_test = data[test_i]
    labels_test = labels[test_i]

    return data_train,labels_train,data_test,labels_test


def augment_class(spectra,labels_onehot,cls,n_new=10):
    '''Return spectra and labels with additional based on those given.
    
    Parameters
    ----------
    spectra : numpy.ndarray
        Two dimensional array of spectra, nspec x spec_length
    labels_onehot : numpy.ndarray
        One-hot array of labels.
    cls : int
        Integer corresponding to class to augment.
    n_new : int, optional
        Number of new spectra to create.
    '''

    new_labels = np.zeros((n_new,labels_onehot.shape[1]))
    new_labels[:,cls] = 1

    ok = labels_onehot[:,cls] == 1
    arr = spectra[ok]

    done = 0
    new = np.zeros((n_new,arr.shape[1]))
    while done < n_new:
        i, j = np.random.randint(arr.shape[0],size=2)
        if i == j:
            continue
        new[done,:] = (arr[i,:] + arr[j,:]) / 2.0
        done += 1

#    return new,new_labels
    return np.vstack((spectra,new)),np.vstack((labels_onehot,new_labels))


def predict_spectra_class(spec_file,return_labels=False,
                           return_bool=False):
    '''Predict class for a given spectrum file.
    
    Parameters
    ----------
    spec_file : string
        Path to spectrum file to be classified
    return_labels : bool, optional
        Return array of labels.
    return_bool : bool, optional
        Return array of booleans corresponding to label.
    '''

    pickle_dir = os.path.dirname(os.path.abspath(__file__))
    labels, clf = utils.get_clf(pickle_dir+'/irs-nn-classifier.pkl')

    out = read_spec(spec_file,median=cfg.median_classifying)
    if out is None:
        return None
    else:
        spec,_,_ = out

    data = spec.reshape(1, -1)
    pred = clf.predict(data)[0]

    if return_labels:
        return labels

    elif return_bool:
        onehot = np.zeros(len(labels),dtype=bool)
        onehot[pred] = True
        return onehot

    else:
        return labels[pred].lower()


def predict_spectra_labels(spec_file,return_labels=False,
                           return_bool=False):
    '''Predict labels for a given spectrum file.
    
    Parameters
    ----------
    spec_file : string
        Path to spectrum file to be classified
    return_labels : bool, optional
        Return array of labels.
    return_bool : bool, optional
        Return array of booleans corresponding to labels.
    '''

    pickle_dir = os.path.dirname(os.path.abspath(__file__))
    labels, clf = utils.get_clf(pickle_dir+'/irs-nn-labeller.pkl')

    out = read_spec(spec_file,median=cfg.median_labelling)
    if out is None:
        return None
    else:
        spec,_,_ = out

    data = spec.reshape(1, -1)
    pred = clf.predict(data)[0]

    if return_labels:
        return labels

    elif return_bool:
        return pred.astype(bool)
        
    else:
        return labels[pred.astype(bool)]


def predict_irs_shell():
    '''Shell script to run photometry classification.'''

    # inputs
    parser = argparse.ArgumentParser(description='IRS spectrum classifier')
    parser.add_argument('--file','-f',nargs='+',action='append',
                        help='Classify CASSIS file or files')
    args = parser.parse_args()

    if args.file is not None:
        files = args.file[0]

    for f in files:

        print('Classifying file: {}'.format(os.path.basename(f)))
        cls = predict_spectra_class(f)
        lab = predict_spectra_labels(f)
        all = np.append(cls,lab)
        if len(all) == 0:
            print('none')
        else:
            print(all)

