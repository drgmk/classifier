from setuptools import setup

setup(
    name='classifier',
    version='0.1',
    description='machine learning classification of stars',
    url='http://github.com/drgmk/classifier',
    author='Grant M. Kennedy',
    author_email='gkennedy@ast.cam.ac.uk',
    license='MIT',
    packages=['classifier'],
    classifiers=['Programming Language :: Python :: 3'],
    install_requires = [
        'astropy == v1.3.2','matplotlib','numpy','requests','scipy'
                        ],
    entry_points = {
        'console_scripts': [
            'classify-phot=classifier.photometry:predict_phot_shell',
            'classify-irs=classifier.spectra:predict_irs_shell'
                            ]
        },
    zip_safe=False
    )
