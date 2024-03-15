# pygsm

This is the gaussian version for the Python Sky Model package.

Install the package my clone this package and run:

`
python setup.py install
`

See example directory for a quick walk through for the package.

Use `re_init` option if you want to recreate spectra / maps for CMB, dust, and synchrotron.

The package only support polarization (Q and U) for synchrotron and dust, no temperature T (yet?).

Everything should be in `CMB uK` unit (I know unit is a pain, but if you keep your parameter inputs this 
way, the package will honor the consistency).

Email Andrew Liu (andrew.liu@princeton.edu) if you have any questions