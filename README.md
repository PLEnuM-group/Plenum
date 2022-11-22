# PLEnuM
This repository contains code that aims to 
show the potential of a global neutrino telescope network.
First studies are published as part of the ICRC2021 proceedings which 
you can find at https://pos.sissa.it/395/1185/.
Potential analyses as presented in the proceeding are illustrated in Jupyter notebooks.
See also the proceedings for important references.

## Basics
The basic assumption of this study is currently that all new telescopes 
included in what we call the PLEnuM network have properties similar to IceCube.
We implemented basic analyses like the search and characterization 
of the diffuse astrophysical neutrino flux and of point-like neutrino sources.
While we currently use only IceCube's data release (http://doi.org/DOI:10.21234/sxvs-mt83)
to estimate the performance of PLEnuM, other detector exposures and resolution function
can be easily added into the framework.

The notebooks and the tools require numpy, scipy, astropy, matplotlib + seaborn, pickle, and MCEq.
Further helpful (but not necessary) packages include cartopy, colorsys and tqdm.

## Usage:
* The effective area and resolution functions are all pre-computed. But if you want
to run the calculations yourself or change anything, you can follow the instructions below.
* Run `aeff_calculations.py`: it will rotate and add up desired effective areas based on
detector locations and IceCube's effective area. In addition, it contains some helper functions
for the notebooks.
* Run `atmospheric_background.py`: it will calculate the reference flux
of atmospheric neutrinos needed for the diffuse flux study. The calculation is based on MCEq.
* Run `energy_resolution.ipynb`: it will calculate the energy resolution function (WIP, needs to be put into core functions)
* Optional: Run `event_numbers.py` as preparation for skymap illustrations and the simple point-source extrapolation study.
* Now you should be good to go to test the analysis notebooks!
