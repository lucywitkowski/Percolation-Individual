#The following code is adapted from Anders Malthe-Sørenssen,
#'Percolation theory using Python', Section 11.2

import numpy as np
import numba
import matplotlib.pyplot as plt

@numba.njit(cache=True)
def percwalk(cluster, max_steps):
    walker_map = np.zeros((2, max_steps))
    displacement = np.zeros_like(walker_map)

    directions = np.zeros((2, 4), dtype=np.int64)
    neighbor_arr = np.zeros(4, dtype=np.int64)

    # nearest-neighbour directions
    directions[0, 0] = 1;  directions[1, 0] = 0
    directions[0, 1] = -1; directions[1, 1] = 0
    directions[0, 2] = 0;  directions[1, 2] = 1
    directions[0, 3] = 0;  directions[1, 3] = -1

    Lx, Ly = cluster.shape

    ix = np.random.randint(Lx)
    iy = np.random.randint(Ly)

    walker_map[0, 0] = ix
    walker_map[1, 0] = iy

    step = 0

    if not cluster[ix, iy]:
        return walker_map, displacement, step

    while step < max_steps - 1:

        neighbor = 0
        for idir in range(4):
            dr = directions[:, idir]
            iix = ix + dr[0]
            iiy = iy + dr[1]

            if 0 <= iix < Lx and 0 <= iiy < Ly and cluster[iix, iiy]:
                neighbor_arr[neighbor] = idir
                neighbor += 1

        if neighbor == 0:
            return walker_map, displacement, step

        randdir = np.random.randint(neighbor)
        dirr = neighbor_arr[randdir]

        ix += directions[0, dirr]
        iy += directions[1, dirr]

        step += 1
        walker_map[0, step] = ix
        walker_map[1, step] = iy
        displacement[:, step] = displacement[:, step-1] + directions[:, dirr]

    return walker_map, displacement, step

#### Average MSD ####
@numba.njit(cache=True)
def find_displacements(p, L, num_systems, num_walkers, max_steps):

    displacements = np.zeros(max_steps)

    for system in range(num_systems):
        z = np.random.rand(L, L) < p

        for j in range(num_walkers):
            num_steps = 0
            while num_steps <= 1:
                walker_map, displacement, num_steps = percwalk(z, max_steps)

            displacements += np.sum(displacement**2, axis=0)

    displacements = displacements / (num_walkers * num_systems)
    return displacements

#### PLOT IT ####
# parameters 
pvals = [0.5, 0.5927, 0.7, 1.0]
L = 100
max_steps = 10000
num_walkers = 500
num_systems = 100

plt.figure()

for i in range(len(pvals)):
    p = pvals[i]
    displacements = find_displacements(p, L, num_systems, num_walkers, max_steps)
    
    dr1 = displacements[1:]
    t = np.arange(len(dr1))
    
    plt.loglog(t,dr1, label = fr'$p ={p}$')

plt.xlabel("t", fontsize = 20)
plt.xticks(fontsize = 15)
plt.ylabel(r"$\langle r^2(t) \rangle$", fontsize = 20)
plt.yticks(fontsize = 15)
plt.legend(loc = 'lower right', fontsize = 15)
plt.tight_layout()

plt.show()



