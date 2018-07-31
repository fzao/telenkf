"""
Ensemble Kalman Filtering example with the Telemac 2D hydrodynamic code

This is a twin experiment based on the 'Estimation' test case where the goal
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

Auteur(s): Fabrice Zaoui

Copyright (c) EDF 2018
"""

from __future__ import print_function
import numpy as np
import os
from matplotlib import pyplot as plt
from pathos.multiprocessing import ProcessingPool as Pool
from pathos.multiprocessing import cpu_count
from TelApy.api.t2d import Telemac2d
from mpi4py import MPI
from numpy.random import multivariate_normal
from scipy.linalg import inv


class ModelTelemac2D:

    def __init__(self, studyFiles, fobs):
        """
        Instantiation
        """
        self.t2d = Telemac2d(studyFiles['t2d.cas'],
                             user_fortran=studyFiles['t2d.f'],
                             comm=MPI.COMM_WORLD)
        # Read the steering file
        self.t2d.set_case()
        # Observation frequency
        self.fobs = fobs
        # State initialization
        self.t2d.init_state_default()
        # Save the initial state
        self.new_state = self.t2d.get_state()
        # Number of 2D mesh nodes
        self.npoin = self.t2d.get('MODEL.NPOIN')
        # Final simulation time
        self.final_time = self.t2d.get("MODEL.NTIMESTEPS")
        # AT and LT info variables on time
        self.AT = self.t2d.get("MODEL.AT")
        self.LT = self.t2d.get("MODEL.LT")
        # Test Telemac for one time step
        self.t2d.run_one_time_step()

    def Run(self, K, State):
        """
        HX operator
        """
        # Set time info (for re-start)
        self.t2d.set("MODEL.LT", self.LT)
        self.t2d.set("MODEL.AT", self.AT)
        # Set the new values for the friction parameter and for all the nodes
        for j in range(self.npoin):
            self.t2d.set('MODEL.CHESTR', K, j)
        # Set the correct initial state corresponding to a particular member
        self.t2d.set_state(State[0], State[1], State[2])
        # Compute telemac2d
        for j in range(self.fobs):
            ierr = self.t2d.run_one_time_step()
            if ierr:
                print('Error with Telemac 2D!')
                break
        # Save new info on times
        self.LT = self.t2d.get("MODEL.LT")
        self.AT = self.t2d.get("MODEL.AT")
        # Return the new hydraulic state
        return self.t2d.get_state()


def simulT2D(Param, Etat):
    """
    A calling sequence for ModelTelemac2D.simulT2D() as Fortran objects are
    not pickable with Python built-in parallelism
    """
    K = Param[0]
    state = etude.Run(K, Etat)
    return state


if __name__ == '__main__':
    # Reading pseudo-observations data (twin experiment)
    # Water depth | x-velocity | y-velocity
    Obs = np.loadtxt("ObsHUV.txt")
    Obs = Obs.transpose()
    # Observation point (it is in the middle of the domain)
    point_obs = 152
    # Frequency of observations
    fobs = 100
    # Number of observations
    ndata = Obs.shape[0]
    # Changing of directory : 'examples' directory of the Telemac sources
    CURDIR = os.getcwd()
    HOMETEL = os.environ.get('HOMETEL')
    os.chdir(HOMETEL + '/examples/telemac2d/estimation')
    #  Telemac 2d files
    studyFiles = {'t2d.f': 'user_fortran',
                  't2d.cas': 't2d_estimation_basic.cas',
                  'f2d.slf': 'f2d_estimation.slf',
                  't2d.geo': 'geo_estimation.slf'}
    # Class Instantiation
    etude = ModelTelemac2D(studyFiles, fobs)
    # EnKF initialization
    nparam = 1  # Number of parameters to estimate
    # Background solution
    KS = raw_input("Choose an initial background value in the range"
                   " [10., 100.] for the Strickler's coefficient : ")
    try:
        KS = float(KS)
    except ValueError:
        print("Invalid value for KS")
    if KS < 10. or KS > 100.:
        print('KS is not in the range [10., 100.]')
        exit()
    Param0 = np.array([KS])
    KsOPT = 35.  # Optimal value for the plotting part
    # Number of members of the ensemble
    Ne = raw_input("Choose a number of members for the ensemble: ")
    try:
        Ne = int(Ne)
    except ValueError:
        print("Invalid value for Ne")
    if Ne < 2:
        print('Ne must be greater than 1')
        exit()
    # States of each member
    State_Ensemble = np.zeros((Ne, 3, etude.npoin))
    State_Ensemble[:, 0, :] = etude.new_state[0]
    State_Ensemble[:, 1, :] = etude.new_state[1]
    State_Ensemble[:, 2, :] = etude.new_state[2]
    # Covariance for the initial draw
    P = np.diag([1])
    # Draw
    Ensemble = np.random.multivariate_normal(mean=Param0, cov=P, size=Ne)
    # Covariance error on observations
    nobs = 3  # number of obs
    e1 = 1.e-2  # mean error for the water depth
    e2 = 1.e-6  # mean error for the x-velocity
    e3 = 1.e-4  # mean error for the y-velocity
    R = np.matrix([[e1*e1, 0, 0], [0, e2*e2, e2*e3], [0, e3*e2, e3*e3]])
    # Covariance error of the model
    Q = np.diag([1.e-8])
    # Dimension of H.x operator
    nhx = 3
    Y = np.zeros((Ne, nhx))  # HX
    # Save results in a list
    result_EnKF = []
    # Computational parameters
    LT = 0
    AT = 0.0
    k = 1
    # Parallelism
    nproc = max(1, cpu_count() - 1)
    pool = Pool(nodes=nproc)
    # Data assimilation cycle loop
    while True:
        print(np.mean(Ensemble))
        result_EnKF.append(np.mean(Ensemble))
        new_state = pool.map(simulT2D, Ensemble, State_Ensemble)
        for i in range(Ne):
            State_Ensemble[i, 0, :] = new_state[i][0]
            State_Ensemble[i, 1, :] = new_state[i][1]
            State_Ensemble[i, 2, :] = new_state[i][2]
            # Save results
            Y[i, 0] = State_Ensemble[i, 0, point_obs]
            Y[i, 1] = State_Ensemble[i, 1, point_obs]
            Y[i, 2] = State_Ensemble[i, 2, point_obs]
        # Noise of the model
        Ensemble[:, :] += multivariate_normal([0]*nparam, Q, Ne)
        # Mean of the model results
        Paramoy = np.mean(Ensemble[:, :], axis=0)
        # Extraction of observation info
        ntps = k * fobs
        Z = Obs[ntps - 1, ]
        # Mean of the forced observations
        Ymoy = np.mean(Y[:, :], axis=0)
        #  filterpy Pyy
        Pyy = 0
        for y in Y[:, :]:
            e_yy = y - Ymoy
            Pyy += np.outer(e_yy, e_yy)
        Pyy = Pyy / (Ne - 1) + R
        # filterpy Pxy
        Pxy = 0
        for i in range(Ne):
            Pxy += np.outer(Ensemble[i, ] - Paramoy, Y[i, ] - Ymoy)
        Pxy /= (Ne - 1)
        # Kalman gain
        K = np.dot(Pxy, inv(Pyy))
        # Error on observations
        e_obs = multivariate_normal([0]*nobs, R, Ne)
        # Update the ensemble
        for i in range(Ne):
            Ensemble[i, ] += np.dot(K, Z + e_obs[i, ] - Y[i, ])
        # End of computation?
        if ntps < ndata:
            k = k + 1
        else:
            break
    # plot the convergence for the paramter (Strickler friction coefficient)
    plt.plot(np.asarray(result_EnKF), label='EnKF convergence')
    plt.axhline(y=KsOPT, color='r', linestyle='-', label='Optimal solution')
    plt.plot(KS, color='steelblue', marker='o', markersize=10)
    plt.legend()
    plt.grid()
    plt.xlabel('Assimilation cycle')
    plt.ylabel('Strickler coefficient m1/3/s')
    plt.show()

    # Ending
    pool.terminate()
    pool.join()
    del(etude.t2d)
    os.chdir(CURDIR)
