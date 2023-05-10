# PLEnuM
This repository contains code that aims to 
show the potential of a PLanEtary NeUtrino Monitoring network.
First studies are published as part of the ICRC2021 proceedings which 
you can find at https://pos.sissa.it/395/1185/.
Potential analyses as presented in the proceeding are illustrated in Jupyter notebooks.
See also the proceedings for important references.

## Basics
The basic assumption of this study is currently that all new telescopes 
included in what we call the PLEnuM network have properties similar to IceCube.
More specific representation of the telescopes under construction will be part of future releases.
We implemented basic analyses like the search and characterization 
of the diffuse astrophysical neutrino flux and of point-like neutrino sources.
While we currently use only IceCube's data release (http://doi.org/DOI:10.21234/sxvs-mt83)
to estimate the performance of PLEnuM, other detector exposures and resolution function
can be easily added into the framework.

The notebooks and core software require standard tools like numpy, scipy, astropy, ..., the machine-learning library scikit-learn, as well as specific tools like MCEq and mephistogram (https://github.com/lisajschumacher/mephistogram).
Further helpful packages include cartopy and tqdm.
Please refer to the Installation section below on how to setup your environment.

## Citation
Please cite our work! Ideally, cite the software and the (latest) accompanying publication. 
```
@misc{github-plenum,
	title = {{PLEnuM software tools}},
	copyright = {BSD-2-Clause license},
	url = {https://github.com/PLEnuM-group/Plenum},
	abstract = {Planetary Neutrino Monitoring},
	publisher = {PLEnuM Group},
	author = {Schumacher, Lisa Johanna and Huber, Matthias and Bustamante, Mauricio},
	collaborator = {Agostini, Matteo and Oikonomou, Foteini and Resconi, Elisa},
}
```


## Installation:
Currently working solution with `conda environments`:
* Create conda env with `conda create -n new_env`
* Install `pip` within the `new_env` with `conda install -n new_env pip`
* Optional: Install `scikit-learn` with `conda install -n new_env -c conda-forge scikit-learn`. It's needed for a new interpolation+smoothing of the energy resolution function. The option to calculate the resolution with the old KDE approach is still available.
* Optional: install `cartopy` with `conda install -n new_env -c conda-forge cartopy`. It's needed for the Earth map plots.
* Install `Plenum` and `mephistogram` packages editable within `new_env` after download in the respective folders with `pip install -e .`
* Possibly, you also need to install `jupyter` explicitly

Packaging will be provided soon!


## Usage:
* The effective area and resolution functions all need to be pre-computed. The `demo` notebooks contain all necessary steps to set up an analysis and are a good way to get started.

If you want to run the calculations yourself or change anything, you can follow the instructions below.
* Run `aeff_calculations.py`: it will rotate and add up desired effective areas based on detector locations and IceCube's effective area. In addition, it contains some helper functions for the notebooks.
* Run `atmospheric_background.py`: it will calculate the reference flux of atmospheric neutrinos needed for the diffuse flux study. The calculation is based on MCEq.
* Run `resolution.py`: it will calculate the energy and angular resolution functions.
* Run `prepare_histograms.py`: it will make sure all your histograms will have the correct binning and then convert them to mephistograms.
* Optional: Run `event_numbers.py` as preparation for skymap illustrations and the simple point-source extrapolation study.
* Now you should be good to go to test the analysis notebooks!
