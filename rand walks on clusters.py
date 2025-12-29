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

    # X-dir: east and west, Y-dir: north and south.
    directions[0, 0] =  1
    directions[1, 0] =  0

    directions[0, 1] = -1
    directions[1, 1] =  0

    directions[0, 2] =  0
    directions[1, 2] =  1

    directions[0, 3] =  0
    directions[1, 3] = -1

    # Initial random position
    Lx, Ly = cluster.shape
    ix = np.random.randint(Lx)
    iy = np.random.randint(Ly)

    walker_map[0, 0] = ix
    walker_map[1, 0] = iy

    step = 0

    # Not in cluster to start with 
    if not cluster[ix, iy]:
        return walker_map, displacement, step

    while step < max_steps - 1:

        # Make list of valid moves
        neighbor = 0
        for idir in range(directions.shape[1]):
            dr = directions[:, idir]
            iix = ix + dr[0]
            iiy = iy + dr[1]

            if (0 <= iix < Lx and
                0 <= iiy < Ly and
                cluster[iix, iiy]):

                neighbor_arr[neighbor] = idir
                neighbor += 1

        # No places to move
        if neighbor == 0:
            return walker_map, displacement, step

        # Select random valid direction 
        randdir = np.random.randint(neighbor)
        dir = neighbor_arr[randdir]

        ix += directions[0, dir]
        iy += directions[1, dir]

        step += 1
        walker_map[0, step] = ix
        walker_map[1, step] = iy
        displacement[:, step] = displacement[:, step - 1] + directions[:, dir]

    return walker_map, displacement, step



#### PLOT 10 INDEP RANDOM WALKS ####

L = 50
pvals = [0.5, 0.7, 1.0] # varying p values 

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for ax, p in zip(axes, pvals):

    z = np.random.rand(L, L) < p

    ax.imshow(
    z.astype(int),
    origin="lower",
    cmap="gray",
    vmin=0,
    vmax=1
)

    for i in range(10):
        walker_map, displacement, steps = percwalk(z, 200)

        if steps > 1:
            ax.plot(
                walker_map[1, :steps],
                walker_map[0, :steps],
                linewidth = 2.7
            )

    ax.set_title(f"$p = {p}$", fontsize=20)
    ax.set_xlabel("y", fontsize=15)
    ax.set_ylabel("x", fontsize=15)
    ax.tick_params(labelsize=15)
    ax.set_aspect("equal")

plt.tight_layout()

plt.show()
