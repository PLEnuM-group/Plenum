# PLEnuM
This repository contains code that aims to 
show the potential of a global neutrino telescope network.
First studies are published as part of the ICRC2021 proceedings which 
you can find at https://pos.sissa.it/395/1185/.
Potential analyses as presented in the proceeding are illustrated in the jupyter notebooks.
See also the proceedings for important references.

The basic assumption of this study is, that all new telescopes 
included in what we call the PLENUM network behave similar to IceCube.
With these notebooks the neutrino point source discovery potential 
and capability of observing features in the diffuse neutrino flux
can be tested; 
of arbitrary detector networks and even new detector 
locations could be motivated.

The notebooks and the tools require numpy, scipy, astropy, matplotlib + seaborn, pickle, and MCEq.
Further helpful (but not necessary) packages include cartopy, colorsys and tqdm.

## Usage:
* Run `aeff_calculations.py`: it will rotate and add up desired effective areas based on
detector locations and IceCube's effective area. In addition, it contains some helper functions
for the notebooks.
* Run `atmospheric_background.py`: it will calculate the reference flux
of atmospheric neutrinos needed for the diffuse flux study.
* Run `event_numbers.py` as preparation for skymap illustrations and point-source study.
* Now you should be good to go to test the notebooks!
