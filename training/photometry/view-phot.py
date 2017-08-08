# coding: utf-8

'''Quick viewer to look at photometry'''

import glob
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons

import sdf.result
import sdf.plotting
import sdf.utils

from classifier.photometry import *
import classifier.config as cfg


def view_one(file,label):
    """Show a plot."""

    print(os.path.basename(file),label)

    r = sdf.result.Result.get(file,('phoenix_m',))

    fig,_ = plt.subplots(figsize=(8,5))
    sdf.plotting.hardcopy_sed(r,fig=fig)

    # check buttons for labelling
    stop = False
    sp = 0.035
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

    next_button.on_clicked(next_func)
    stop_button.on_clicked(stop_func)

    plt.show()


# start of the script fo real

# get all the data, labels are one hot vectors
files = glob.glob('labels/*.csv')
files = ['labels/absil_2013.csv']

for f in files:

    print("Files:",f.split('/')[1])
    data,labels,sdbids,skip,label_names = get_data([f])

    for sdbid,label in zip(sdbids,labels):

        dir = '/Users/grant/a-extra/sdb/masters/'
        lab = label_names[np.argmax(label)]
        view_one(dir+sdbid+'/public/'+sdbid+'-rawphot.txt',lab)

