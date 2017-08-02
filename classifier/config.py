import os
# where the module files are
here = os.path.dirname(os.path.realpath(__file__))

# fill in to get rawphot files from sdb, only needed for getting new
# files for training
sdb_user = ''
sdb_pass = ''

# where in-module training stuff is
training_dir = here + '/../training/'

# where external (i.e. not with this module) spectra are
spectra = here + '/../../spectra/'

# median filters to apply when reading spectra for classifying/labelling
median_classifying = 30
median_labelling = 3
