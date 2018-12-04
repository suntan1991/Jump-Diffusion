# -*- coding: utf-8 -*-
"""
Created on Thu Jun  7 13:17:58 2018

@author: jliubt
This is the last version for calculation, checked on Sep-13, 2018
"""

# This script is to calculate the one step jump rate of Na ions in Na11Ge2PS12
# There are total 5 Na sites, at differetn z coordinate, Na has differnt distribution
# at Z = 0, 0.25, 0.5, 0.75, 1.0: Na2, Na3, Na4
# at Z = 0.125, 0.375, 0.625, 0.875: Na1, Na5

import numpy as np
import copy, time
import xdatcarRead

###############################################################################
#    Trace the grid index and site lable of specific ion. Here is Na.         #
#    traj is the MD trajectories, in dimension of [N_time, N_atom, 3]         #
#    site_dict is the site label for Na at each grid point.                   #
#    The grid index is numbered as shown in ReadSiteIndex.                    #
###############################################################################
def JumpTrace(traj, site_dict):
    
    N_frame, N_atom, dim = traj.shape
    # the grid number and grid size for every direction, in fractional coordinate
    N_grid = np.array([5, 5, 9])
    grid = 1 / (N_grid-1)
    # trace Na site and grid index for every time step
    site_Label = np.zeros((N_atom, N_frame), dtype = 'int32')
    grid_Index = np.zeros((N_atom, N_frame), dtype = 'int32')
    
    for k in range(N_frame):
        pos = traj[k]
        index = np.int_(np.rint(pos / grid))
        flat_index = np.int_(np.array(index[:,2] * N_grid[0] * N_grid[1] + index[:,0] * N_grid[1] + index[:,1]))
        site_Label[:,k] = site_dict[flat_index]
        grid_Index[:,k] = copy.copy(flat_index)
    
    
    return N_grid, grid_Index, site_Label


###############################################################################
#    Count the jumps of Na ions between two differnt grids. The site lable    #
#    can be identical or different, e.g., from Na1->Na1 or Na2->Na5           #
#    Only consider one step jumps here in this function.                      #
#    Na_site is the total sites of Na, is 5 for this material                 #
###############################################################################
def SingleJump(Na_grid_trace, Na_site_trace, N_site, N_grid):
    
    N_Na, N_frame = Na_grid_trace.shape
    N_site += 1    # to start from 1 to 5
    # initialize the jump count matrix for Na
    # Na_jump_count matrix is consited of 3 components, jumps along c axis, jumps in ab-plane, and jumps sum of both
    # Na_jump_count[0] for c axis, Na_jump_count[1] for ab plane, Na_jump_count[2] for total
    Na_jump_count = np.zeros((3, N_Na, N_site, N_site))
    Na_grid_prev = np.copy(Na_grid_trace[:,0])
    Na_site_prev = np.copy(Na_site_trace[:,0])
    Na_site_summary = {}
    
    file = open('Na_Site_Jump.out', 'w')
    file.write("Na_atom    total_jump    jump_record\n")
    
    for n in range(N_Na):
        Na_site = []
        Na_grid_prev = Na_grid_trace[n,0]
        Na_site_prev = Na_site_trace[n,0]
        Na_site.append(Na_site_prev)
        for k in range(1, N_frame):
            grid_index = Na_grid_trace[n,k]
            # measure if the grid index changes. if YES, means the ion may jump to another site
            if grid_index != Na_grid_prev:
                c_index = np.floor(np.array([grid_index, Na_grid_prev]) / (N_grid[0] * N_grid[1]))
                Na_grid_prev = grid_index
                Na_site_new = Na_site_trace[n,k]
                Na_site.append(Na_site_new)
                # Count the jumps in different c axis. If c index are same, ignore the jump
                if c_index[0] != c_index[1]:
                    Na_jump_count[0, n, Na_site_prev, Na_site_new] += 1
                else:
                    Na_jump_count[1, n, Na_site_prev, Na_site_new] += 1
                    
                # Count all the jumps, including both in ab-plane and along c axis
                Na_jump_count[2, n, Na_site_prev, Na_site_new] += 1
                Na_site_prev = Na_site_new
                
        Na_site_summary['Na'+str(n+1)] = Na_site
        file.write("{:6} {:>4d}{:>4}".format("Na"+str(n+1),len(Na_site), ":"))
        for i in range(len(Na_site)-1):
            file.write(str(Na_site[i])+"-->")
        file.write(str(Na_site[-1]))
        file.write("\n")
    file.close()
    
    # Sum the diagonal entries in the Na_jump_count matrix
    Na_jump_count = np.sum(Na_jump_count, axis=1)
#    Na_jump_total = np.zeros((3, N_site, N_site))
#    for i in range(3):
#        Na_jump_total[i] = np.tril(Na_jump_count[i])
#        Na_jump_total[i] += np.triu(Na_jump_count[i], k=1).T
    
    return Na_site_summary, Na_jump_count


###############################################################################
#    Count the jumps similar to the above function. The differnce is that in  #
#    this fucntion we consider a series jumps, e.g., Na1->Na1->Na3 or         #
#    Na3->Na1->Na4. Only consider two steps jump                              #
###############################################################################   
def ContinueJump(Na_site_summary, N_site):

    N_Na = len(Na_site_summary)
    N_site += 1    # to start from 1 to 5
    # initialize the jump count matrix for Na
    Na_jump_count = np.zeros((N_site, N_site, N_site))
    
    for n in range(N_Na):
        Na_jump = Na_site_summary['Na'+str(n+1)]
        for k in range(2, len(Na_jump)):
            site = Na_jump[k-2:k+1]
            Na_jump_count[site[0], site[1], site[2]] += 1
    
    return Na_jump_count


###############################################################################
#    Read in the grid index and Na site labels at each grid for differnt Z    #
#    coordinates. z at [0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0]. #
#    The grid index are first indexed for b(y), then a(x), last c(z). So the  #
#    index is numbered as                                                     #
#    z_index * len(a_grid)*len(b_grid) + a_index*len(b_grid) + b_index        #
#    In this project, the len(a_grid)=len(b_grid)=4+1
###############################################################################
def ReadSiteIndex(filename):
    
    with open(filename, 'r') as file:
        file.readline()
        lines = file.readlines()
        Na_site_dict = np.asarray([int(line) for line in lines])
    
    return Na_site_dict



def writeJumpResults(Na_Single_Jump, Na_Contin_Jump, N_site):
    
    with open("Jump_result.out", 'w') as file:
        file.write("Single_Jump    Count\n")
        for i in range(1, N_site+1):
            for j in range(1, N_site+1):
                file.write("{}-->{}: {:.0f}\n".format(i,j,Na_Single_Jump[2,i,j]))
        file.write("\n\n\n")


        file.write("Single_Jump_along_c_axis    Count\n")
        for i in range(1, N_site+1):
            for j in range(1, N_site+1):
                file.write("{}-->{}: {:.0f}\n".format(i,j,Na_Single_Jump[0,i,j]))
        file.write("\n\n\n")


        file.write("Single_Jump_in_ab_plane    Count\n")
        for i in range(1, N_site+1):
            for j in range(1, N_site+1):
                file.write("{}-->{}: {:.0f}\n".format(i,j,Na_Single_Jump[1,i,j]))
        file.write("\n\n\n")

        
        file.write("Double_Jump    Count\n")
        for i in range(1, N_site+1):
            for j in range(1, N_site+1):
                for k in range(1, N_site+1):
                    file.write("{}-->{}-->{}: {:.0f}\n".format(i,j,k,Na_Contin_Jump[i,j,k]))
        file.close()
    
###############################################################################
# Start the calculations
start_time = time.clock()

# Read the grid index and the Na site lable from the file
Na_site_dict = ReadSiteIndex('Na_grid_site')

# Read the XDATCAR file and extract the Na trajectories for jump analysis
system, lattice, element_list, element_dict, trajectories = xdatcarRead.read("XDATCAR_1000K")
Na_traj = trajectories[:, :88]

# Trace the grid index and site lable for each Na ion
N_grid, Na_grid_trace, Na_site_trace = JumpTrace(Na_traj, Na_site_dict)

# calculate the single jump for Na ions
Na_site_summary, Na_Single_Jump = SingleJump(Na_grid_trace, Na_site_trace, 5, N_grid)

# calculate the two continuous jumps for Na ions
Na_Contin_Jump = ContinueJump(Na_site_summary, 5)

# write the jump results
writeJumpResults(Na_Single_Jump, Na_Contin_Jump, 5)

time_cost = time.clock() - start_time
print("------ Time used: %s ------" % time.strftime("%H:%M:%S", time.gmtime(time_cost)))
