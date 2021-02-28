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

Author(s): Fabrice Zaoui, Yoann Audouin, Cedric Goeury

To cite this work please use:
    Fabrice Zaoui, Cédric Goeury, Yoann Audouin
    Ensemble Integrations of Telemac-Mascaret for the Optimal Model Calibration
    XXVth Telemac & Mascaret User Club, Oct 2018, Norwich, United Kingdom
    https://hal.archives-ouvertes.fr/hal-01908756

Copyright (c) EDF 2018-2021
"""
from telapy.api.t2d import Telemac2d
import os
import numpy as np
from numpy.random import multivariate_normal
from matplotlib import pyplot as plt
from mpi4py import MPI
from scipy.linalg import inv


class ModelTelemac2D(object):

    def __init__(self, studyFiles, fobs, comm=MPI.COMM_SELF):
        """
        Instantiation
        """
        self.t2d = Telemac2d(studyFiles['t2d.cas'],
                             user_fortran=studyFiles['t2d.f'],
                             comm=comm,
                             stdout=0)
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

    def Run(self, K, State):
        """
        HX operator
        """
        # Set the new values for the friction parameter and for all nodes
        K_array = np.ones(self.npoin) * K
        self.t2d.set_array('MODEL.CHESTR', K_array)
        # Set the correct initial state corresponding to a particular member
        self.t2d.set_state(State[0], State[1], State[2])
        # Compute telemac2d
        for j in range(self.fobs):
            ierr = self.t2d.run_one_time_step()
            if ierr:
                print('Error with Telemac 2D!')
                break
        # Return the new hydraulic state
        return self.t2d.get_state()


def main():
    """
    Main function
    """
    gbl_comm = MPI.COMM_WORLD
    gbl_rank = gbl_comm.Get_rank()
    gbl_ncsize = gbl_comm.Get_size()
    ncsize_run = 2  # two procs for the physics of Telemac are requested...
    #  ...(one can change this)
    # Checking consistensy of parallel information
    if gbl_ncsize % ncsize_run != 0:
        print("Number of cores for a telemac run must divide\
               the total number of cores")
        print("Total number of cores:", gbl_ncsize)
        print("Telemac run number of cores:", ncsize_run)
        raise ValueError
    # Creating local communicator
    color = gbl_rank//ncsize_run
    key = gbl_rank - (color*ncsize_run)
    comm = gbl_comm.Split(color, key)
    rank = comm.Get_rank()
    niter = gbl_ncsize//ncsize_run
    proc0_grp = gbl_comm.group.Incl(list(range(0, gbl_ncsize, ncsize_run)))
    proc0_comm = gbl_comm.Create(proc0_grp)

    # EnKF initialization
    nparam = 1  # Number of parameters to estimate
    # Observation point (it is in the middle of the domain)
    point_obs = 152
    # Frequency of observations (in number of time steps)
    fobs = 100

    if gbl_rank == 0:
        # Reading pseudo-observations data (twin experiment)
        # Water depth | x-velocity | y-velocity
        Obs = np.loadtxt("ObsHUV.txt")
        Obs = Obs.transpose()
        # Number of observations
        ndata = Obs.shape[0]

        # Background solution
        KS = input("Choose an initial background value in the range"
                   " [10., 90.] for the Strickler's coefficient : ")
        try:
            KS = float(KS)
        except ValueError:
            print("Invalid value for KS")
        if KS < 10. or KS > 90.:
            print('KS is not in the range [10., 90.]')
            exit()
        Param0 = np.array([KS])
        KsOPT = 35.  # Optimal value for the plotting part
        # Number of members of the ensemble
        Ne = input("Choose a number of members for the ensemble: ")
        try:
            Ne = int(Ne)
        except ValueError:
            print("Invalid value for Ne")
        if Ne < 2:
            print('Ne must be greater than 1')
            exit()
        tmp = np.zeros(nparam+1, dtype=int)
        tmp[0] = ndata
        tmp[1] = Ne
    else:
        tmp = np.zeros(nparam+1, dtype=int)

    # Transferring ndata and Ne
    gbl_comm.Bcast(tmp, root=0)
    ndata = tmp[0]
    Ne = tmp[1]

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
    study = ModelTelemac2D(studyFiles, fobs, comm=comm)

    # States of each member
    State_Ensemble = np.zeros((Ne, 3, study.npoin))
    State_Ensemble[:, 0, :] = study.new_state[0]
    State_Ensemble[:, 1, :] = study.new_state[1]
    State_Ensemble[:, 2, :] = study.new_state[2]
    # Covariance for the initial draw
    P = np.diag([1])
    # Defining New ensemble on proc 0 and broadcasting it to the others
    if gbl_rank == 0:
        # Draw
        Ensemble = np.random.multivariate_normal(mean=Param0, cov=P, size=Ne)
    else:
        Ensemble = np.zeros((Ne, nparam))
    gbl_comm.Bcast(Ensemble, root=0)

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
    Y = np.zeros((Ne, nhx), 'd')  # HX

    if gbl_rank == 0:
        # Save results in a list
        result_EnKF = []
    # Computational parameters
    k = 1
    # Data assimilation cycle loop
    while True:
        # Print the representative mean value of the Ensemble
        if gbl_rank == 0:
            print(np.mean(Ensemble))
            # Save this value for plotting the convergence
            result_EnKF.append(np.mean(Ensemble))

        my_ne = Ne//niter
        # Compute and save each member with Telemac run in parallel
        Y[:, :] = 0.0
        start = my_ne*color
        end = my_ne*(color+1)
        # Forcing last process to do the rest of the loop
        if(color == (gbl_ncsize-1)//ncsize_run):
            end = Ne
        for i in range(start, end):
            new_state = study.Run(Ensemble[i][0], State_Ensemble[i])
            State_Ensemble[i, 0, :] = new_state[0]
            State_Ensemble[i, 1, :] = new_state[1]
            State_Ensemble[i, 2, :] = new_state[2]
            # Save results using global numebring
            Y[i, 0] = study.t2d.mpi_get('MODEL.WATERDEPTH', i=point_obs)
            Y[i, 1] = study.t2d.mpi_get('MODEL.VELOCITYU', i=point_obs)
            Y[i, 2] = study.t2d.mpi_get('MODEL.VELOCITYV', i=point_obs)

        # All the proc 0 of each telemac run needs to merge their results
        if rank == 0:
            tmp = np.zeros_like(Y)
            proc0_comm.Reduce(Y, tmp, op=MPI.SUM, root=0)

        # Terminate with ntps when no more observations
        ntps = k * fobs
        if gbl_rank == 0:
            Y = tmp
            # Noise of the model
            Ensemble[:, :] += multivariate_normal([0]*nparam, Q, Ne)
            # Mean of the model results
            Paramoy = np.mean(Ensemble[:, :], axis=0)
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
        else:
            Ensemble = np.zeros((Ne, nparam))
        # Broadcasting New ensemble to all processors
        gbl_comm.Bcast(Ensemble, root=0)

        # End of computation?
        if ntps < ndata:
            k = k + 1
        else:
            break

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
    del study.t2d
    os.chdir(CURDIR)


if __name__ == '__main__':
    main()
