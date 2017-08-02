from functools import lru_cache
import itertools
import pickle
import numpy as np
import matplotlib.pyplot as plt


#@lru_cache()
def get_clf(file):
    '''Memoised getting of pickled classifier.'''
    
    with open(file,'rb') as f:
        labels = pickle.load(f)
        clf = pickle.load(f)

    return labels,clf


def plot_multilabel_confusion_matrix(true,pred,label_names,lines=[]):
    '''Make a multi-label "confusion" matrix.

    Columns are predicted labels, row are true for each incorrect
    predicted label, every true label that exists for a spectrum
    is given +1.
    '''
    nlabels = len(label_names)
    cm = np.zeros((nlabels,nlabels))
    diag = np.zeros(nlabels,dtype=int)
    for i,p_list in enumerate(pred):
        t_list = true[i]

        for k,p in enumerate(p_list):
            if p == 1:
                if t_list[k] == 1: # note correct labels
                    diag[k] += 1
                    continue
                for l,t in enumerate(t_list):
                    if t == 1:
                        cm[l,k] += 1

    fig,ax = plt.subplots(figsize=(9.5,7))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)

    for i in range(nlabels):
        plt.text(i, i, diag[i],
                 horizontalalignment="center",
                 color="black")

    for n in lines:
        ax.plot(ax.get_xlim(),[n,n],color='grey')
        ax.plot([n,n],ax.get_ylim(),color='grey')

    fig.colorbar(im)

    tick_marks = np.arange(nlabels)
    ax.set_xticks(tick_marks)
    ax.set_xticklabels( label_names, rotation=90)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels( label_names)
    ax.set_ylabel('True labels')
    ax.set_xlabel('Predicted labels')


def plot_confusion_matrix(cm, classes,
                          lines=[],
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """Print and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.
    
    From http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
    """
    fig,ax = plt.subplots(figsize=(9.5,5))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.set_title(title)
    fig.colorbar(im)
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels( classes, rotation=45)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels( classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if cm[i, j] != 0:
            plt.text(j, i, cm[i, j],
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    for n in lines:
        ax.plot(ax.get_xlim(),[n,n],color='grey')
        ax.plot([n,n],ax.get_ylim(),color='grey')

    fig.tight_layout()
    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')

