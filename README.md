Ensemble Kalman Filter Example for TELEMAC-MASCARET
===================================================

Ensemble Kalman Filtering example with the Telemac 2D hydrodynamic code

This is a twin experiment based on the 'Estimation' test case where the goal
    is to estimate the bottom friction coefficient

<p align="center">
  <img src="./doc/enk.gif" width="800"/>
</p>
<p align="center">
   <b>Example of convergence for Ne=10 and Ks_initial_value = 20</b>
</p>

The 3 scripts contain:
* telenkf_t2d_estimation_python2_pathos.py
  Python 2 version of the script Telemac 2D in sequential but the runs are mutlithreaded with pathos.
* telenkf_t2d_estimation_python3_pathos.py
  Same as before but Python 3 compatible.
* telenkf_t2d_estimation_python3_mpi.py
  Python 3 version of the script with the Telemac 2D study that can run in
  parallel (mpirun -n X python3 telenkf_t2d_estimation_python3.py to run the script with
  Telemac 2D running on X cores) but the EnKF in sequential


This Python script uses:
* EnKF algorithm from the 'filterpy' module (http://github.com/rlabbe/filterpy)

* TelApy a set of Python API for the Telemac-Mascaret system

* Mpi4py to use MPI in Python

Requirements
============

- NumPy
- Matplotlib
- Pathos
- Mpi4py

License
=======

This package is provided under the MIT license

References
==========

- [F. Zaoui, C. Goeury, Y. Audouin. "Ensemble Integrations of Telemac-Mascaret for the optimal model calibration"](https://hal.archives-ouvertes.fr/hal-01908756)

- John L Crassidis and John L. Junkins. "Optimal Estimation of
Dynamic Systems", CRC Press, second edition. 2012. pp, 257-9

- Y. Audouin, C. Goeury, F. Zaoui, R. Ata, S. El Idrissi Essebtey,
A. Torossian, and D. Rouge. "Interoperability applications of
TELEMAC-MASCARET System", 24th Telemac-Mascaret User Conference,
Graz, Austria, 2017, pp. 57-64

Contributions
=============

Contributions are always welcome ;-)

When contributing to please consider discussing the changes you wish to make via issue or e-mail to the maintainer.
