"""
Ensemble Kalman Filtering example with the Mascaret 1D hydrodynamic code

This is a twin experiment based on the 'Test1' test case where the goal
    is to estimate the bottom friction coefficient

This Python script uses:
* EnKF algorithm from the 'filterpy' module (http://github.com/rlabbe/filterpy)

    Reference: John L Crassidis and John L. Junkins. "Optimal Estimation of
      Dynamic Systems", CRC Press, second edition. 2012. pp, 257-9

    'filterpy' is distributed under The MIT License (MIT) and
        Copyright (c) 2015 Roger R. Labbe Jr

  The MIT License (MIT)

  Permission is hereby granted, free of charge, to any person obtaining a copy
  of this software and associated documentation files (the "Software"), to deal
  in the Software without restriction, including without limitation the rights
  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
  copies of the Software, and to permit persons to whom the Software is
  furnished to do so, subject to the following conditions:

  The above copyright notice and this permission notice shall be included in
  all copies or substantial portions of the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
  THE SOFTWARE.

* TelApy a set of Python API for the Telemac-Mascaret system
    Reference: Y. Audouin, C. Goeury, F. Zaoui, R. Ata, S. El Idrissi Essebtey,
        A. Torossian, and D. Rouge. "Interoperability applications of
        TELEMAC-MASCARET System", 24th Telemac-Mascaret User Conference,
        Graz, Austria, 2017, pp. 57-64

Author(s): Fabrice Zaoui, Yoann Audouin, Cedric Goeury

To cite this work please use:
    Fabrice Zaoui, Cédric Goeury, Yoann Audouin
    Ensemble Integrations of Telemac-Mascaret for the Optimal Model Calibration
    XXVth Telemac & Mascaret User Club, Oct 2018, Norwich, United Kingdom
    https://hal.archives-ouvertes.fr/hal-01908756

Copyright (c) EDF 2018-2021
"""
from telapy.api.masc import Mascaret
import os
import numpy as np
from numpy.random import multivariate_normal
from matplotlib import pyplot as plt
from mpi4py import MPI
from scipy.linalg import inv


class ModelMascaret1D(object):

    def __init__(self, studyFiles, fobs):
        """
        Instantiation
        """
        self.masc = Mascaret()
        # Model creation
        self.masc.create_mascaret(iprint=0)
        # read data files
        self.masc.import_model(studyFiles[0], studyFiles[1])
        # Observation frequency
        self.fobs = fobs
        # Number of 1D mesh nodes
        self.npoin = self.masc.get_var_size('Model.X')[0]
        # Initialize Mascaret before the computations with
        # pseudo-values (steady kernel)
        self.masc.init_hydro([0.]*self.npoin, [0.]*self.npoin)

    def Run(self, K):
        """
        HX operator
        """
        # Set the new values
        for j in range(self.npoin):
            self.masc.set_double('Model.FricCoefMainCh', K, j+1, 0, 0)
        # Compute telemac2d
        for j in range(self.fobs):
            self.masc.compute(0., 1., 1.)
        # Get results
        Zres, Qres = self.masc.get_hydro()
        # Return the new water levels
        return Zres


gbl_comm = MPI.COMM_WORLD
gbl_rank = gbl_comm.Get_rank()
gbl_ncsize = gbl_comm.Get_size()
ncsize_run = 1  # only one processor is possible for Mascaret
# Checking consitensy of parallel information
if gbl_ncsize % ncsize_run != 0:
    print("Number of cores for a telemac run must divide the total number\
           of cores")
    print("Total number of cores:", gbl_ncsize)
    print("Telemac run number of cores:", ncsize_run)
    raise ValueError
# Creating local communicator
color = gbl_rank//ncsize_run
key = gbl_rank - (color*ncsize_run)
comm = gbl_comm.Split(color, key)
rank = comm.Get_rank()
ncsize = comm.Get_size()
niter = gbl_ncsize//ncsize_run
proc0_grp = gbl_comm.group.Incl(list(range(0, gbl_ncsize, ncsize_run)))
proc0_comm = gbl_comm.Create(proc0_grp)

# EnKF initialization
nparam = 1  # Number of parameters to estimate
# Frequency of observations (in number of time steps)
fobs = 1

if gbl_rank == 0:
    # Background solution
    KS = input("Choose an initial background value in the range"
               " [10., 50.] for the Strickler's coefficient : ")
    try:
        KS = float(KS)
    except ValueError:
        print("Invalid value for KS")
    if KS < 10. or KS > 50.:
        print('KS is not in the range [10., 50.]')
        exit()
    Param0 = np.array([KS])
    KsOPT = 30.6  # Optimal value for the plotting part
    # Number of members of the ensemble
    Ne = input("Choose a number of members for the ensemble: ")
    try:
        Ne = int(Ne)
    except ValueError:
        print("Invalid value for Ne")
    if Ne < 2:
        print('Ne must be greater than 1')
        exit()
    # Number of assimilation cycles
    Na = input("Choose a number of assimilation cycles: ")
    try:
        Na = int(Na)
    except ValueError:
        print("Invalid value for Na")
    if Na < 1:
        print('Na must be greater than 1')
        exit()
    tmp = np.zeros(2, dtype=int)
    tmp[0] = Ne
    tmp[1] = Na
else:
    tmp = np.zeros(2, dtype=int)

# Transferring Ne
gbl_comm.Bcast(tmp, root=0)
Ne = tmp[0]
Na = tmp[1]

# Changing of directory : 'examples' directory of the Telemac sources
CURDIR = os.getcwd()
HOMETEL = os.environ.get('HOMETEL')
os.chdir(HOMETEL + '/examples/mascaret/1_Steady_Kernel')

#  Mascaret files
files_name = ['sarap.xcas', 'geometrie', 'hydrogramme.loi',
              'limnigramme.loi', 'mascaret0.lis', 'mascaret0.opt']
files_type = ['xcas', 'geo', 'loi', 'loi', 'lis', 'res']
studyFiles = [files_name, files_type]

# Class Instantiation
study = ModelMascaret1D(studyFiles, fobs)

# Covariance for the initial draw
P = np.diag([1])
# Defining New ensemble on proc 0 and broadcasting it to the others
if gbl_rank == 0:
    # Draw
    Ensemble = np.random.multivariate_normal(mean=Param0, cov=P, size=Ne)
else:
    Ensemble = np.zeros((Ne, nparam))
gbl_comm.Bcast(Ensemble, root=0)

# Twin experiments: compute one step with the optimal friction coefficient
Zobs = study.Run(np.ones((1, 1)) * 30.6)

# Error Statistics
Err_EnKF = []
# Covariances
nobs = study.npoin
R = np.diag([1.e-3] * nobs)  # obs
Q = np.diag([1.e-8])  # process
# Results for parameters
Param_Ensemble = np.zeros((Ne, nparam))
Param_Ensemble[:, :] = Ensemble
# H.x operator
nx = nobs
Y = np.zeros((Ne, nx))  # HX
# Loop
k = 0
if gbl_rank == 0:
    # Save the global results in a list
    result_EnKF = []
while True:  # time loop
    # Print the representative mean value of the Ensemble (global result)
    if gbl_rank == 0:
        print(np.mean(Ensemble))
        # Save this value for the plotting of the convergence
        result_EnKF.append(np.mean(Ensemble))
    my_ne = Ne//niter
    # Compute and save each member with Telemac run in parallel
    Y[:, :] = 0.0
    start = my_ne*color
    end = my_ne*(color+1)
    # Forcing last process to do the rest of the loop
    if(color == (gbl_ncsize-1)//ncsize_run):
        end = Ne
    # For each member
    for i in range(start, end):
        Param = Ensemble[i, ]
        # HX operator
        Y[i, ] = study.Run(Param[0])
    # All the proc 0 of each telemac run needs to merge their results
    if rank == 0:
        tmp = np.zeros_like(Y)
        proc0_comm.Reduce(Y, tmp, op=MPI.SUM, root=0)

    if gbl_rank == 0:
        Y = tmp
        # Add process noise
        Ensemble[:, :] += multivariate_normal([0]*nparam, Q, Ne)
        # Mean of the results
        Paramoy = np.mean(Ensemble[:, :], axis=0)
        # Mean of the forced observations
        Ymoy = np.mean(Y[:, :], axis=0)
        # Filterpy Pyy
        Pyy = 0
        for y in Y[:, :]:
            e_yy = y - Ymoy
            Pyy += np.outer(e_yy, e_yy)
        Pyy = Pyy / (Ne-1) + R
        # Filterpy Pxy
        Pxy = 0
        for i in range(Ne):
            Pxy += np.outer(Ensemble[i, ] - Paramoy, Y[i, ] - Ymoy)
        Pxy /= (Ne-1)
        # Kalman gain
        K = np.dot(Pxy, inv(Pyy))
        # Observation errors
        e_obs = multivariate_normal([0]*nobs, R, Ne)
        # Update the ensemble
        for i in range(Ne):
            Ensemble[i, ] += np.dot(K, Zobs + e_obs[i, ] - Y[i, ])
    else:
        Ensemble = np.zeros((Ne, nparam))
    # Broadcasting New ensemble to all processors
    gbl_comm.Bcast(Ensemble, root=0)
    # Stop criterion
    if k >= Na:
        break
    k = k + 1

if gbl_rank == 0:
    # plot the convergence for the parameter (Strickler friction coef.)
    plt.plot(np.asarray(result_EnKF), label='EnKF convergence')
    plt.axhline(y=KsOPT, color='r', linestyle='-',
                label='Optimal solution')
    plt.plot(KS, color='steelblue', marker='o', markersize=10)
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
               ncol=2, mode="expand", borderaxespad=0.)
    plt.grid()
    plt.xlabel('Assimilation cycle')
    plt.ylabel('Strickler coefficient m1/3/s')
    plt.show()


gbl_comm.Barrier()

# Ending
del study.masc
os.chdir(CURDIR)
