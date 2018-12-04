# -*- coding: utf-8 -*-
"""
Created on Mon May  7 22:19:42 2018

@author: jliubt
"""
import numpy as np
import os, pickle

def read(xdata_file):
    xdatcar = open(xdata_file, 'r')
    lines = xdatcar.readlines()
    xdatcar.close()
    len_xdatcar = len(lines)                                                    # get the total lenght of the XDATCAR file to decide the time step of MD
    print (len_xdatcar)
    
    xdatcar = open(xdata_file, 'r')
    system = xdatcar.readline()
    print ("Reading the XDATCAR file of system %s into the program." % system)
    
    # get the scale and the lattice vectors in the head of the XDATCAR file
    scale = float(xdatcar.readline().rstrip('\n'))
    lattice = np.zeros((3,3))
    #basis vectors in cartesian coords
    for i in range(3):
        lattice[i] = np.array([float(s)*scale for s in xdatcar.readline().rstrip('\n').split()])

    # get the element of the system and the correpsonding atom numbers
    element_list = xdatcar.readline().rstrip('\n').split()
    element_dict = {}
    element_numbers = [int(s) for s in xdatcar.readline().rstrip('\n').split()]
    N_atoms = sum(element_numbers)
    
    for t in range(len(element_list)):
        element_dict[element_list[t]] = element_numbers[t]
    
    # MD time steps calculated
    N_frame = (len_xdatcar-7) // (N_atoms+1)
    print ("MD simulation runs for total %d time steps" % N_frame)
    
    # Store all the positions in the variable pos
    if os.path.isfile('xdatcar.pckl'):
        f = open('xdatcar.pckl', 'rb')
        pos = pickle.load(f)
        f.close()
    else:
        pos = np.zeros((N_frame, N_atoms, 3))
        for t in range(N_frame):
            xdatcar.readline()
            for j in range(N_atoms):
                pos[t,j] = np.array([ float(s) for s in xdatcar.readline().rstrip('\n').split() ])
            if t % 1000 == 0:
                print (t)
        f = open("xdatcar.pckl", "wb")
        pickle.dump(pos, f)
        f.close()
    xdatcar.close()                                                             # close the file
    print ("Successfully read the XDATACAR file!!!")
    return system, lattice, element_list, element_dict, pos

if __name__ == "__main__":
    
    import sys
    read(sys.argv[1])