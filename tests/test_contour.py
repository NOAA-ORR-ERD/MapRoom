import os

import numpy as np
import scipy.stats

from mock import maproom_dir, MockProject

from maproom import loaders
from maproom.library.Boundary import Boundaries, PointsError


def surface_conc_kde(sc):
    """
    Computes the surface concentration using scipy's

    Kernel Density Estimator code

   

    a "surface_concentration" array will be added to the spill container

    :param sc: spill container that you want the concentrations computed on
    """
    spill_num = sc['spill_num']
    sc['surface_concentration'] = np.zeros(spill_num.shape[0],)
    for s in np.unique(spill_num):
        sid = np.where(spill_num==s)
        positions = sc['positions'][sid]
        mass = sc['mass'][sid]
        age = sc['age'][sid]
        c = np.zeros(positions.shape[0],)
        lon = positions[:, 0]
        lat = positions[:, 1]

        bin_length = 6*3600 #kde will be calculated on particles 0-6hrs, 6-12hrs,...
        t = age.min()
        max_age = age.max()
        
        while t<=max_age:
            id = np.where((age<t+bin_length))[0] #we use all particles < t + bin_length for kernel
            lon_for_kernel = lon[id]
            lat_for_kernel = lat[id]
            age_for_kernel = age[id]
            mass_for_kernel = mass[id]
            id_bin = np.where(age_for_kernel>=t)[0] #we only calculate pdf for particles in bin

            if len(np.unique(lat_for_kernel))>2 or len(np.unique(lon_for_kernel))>2: # can't compute a kde for less than 3 unique points!
                lon0, lat0 = min(lon_for_kernel), min(lat_for_kernel)
                # FIXME: should use projection code to get this right.
                x = (lon_for_kernel - lon0) * 111325 * np.cos(lat0 * np.pi / 180)
                y = (lat_for_kernel - lat0) * 111325
                xy = np.vstack([x, y])
                if len(np.unique(mass_for_kernel)) > 1:
                    kernel = gaussian_kde(xy,weights=mass_for_kernel/mass_for_kernel.sum()) 
                else:
                    kernel = gaussian_kde(xy) 
                if mass_for_kernel.sum() > 0:
                    c[id[id_bin]] = kernel(xy[:,id_bin]) * mass_for_kernel.sum() 
                else:
                    c[id[id_bin]] = kernel(xy[:,id_bin]) * len(mass_for_kernel)
            t = t + bin_length
            
        sc['surface_concentration'][sid] = c


class TestContour(object):
    def setup(self):
        self.project = MockProject()
        self.project.load_file(maproom_dir + "/TestData/NC_particles/gnome_output_spill_start_after_model.nc", "application/x-nc_particles")
        self.folder = self.project.layer_manager.get_nth_oldest_layer_of_type("particles", 1)

    def test_simple(self):
        timesteps = self.folder.get_particle_layers()
        print(len(timesteps))
        assert 25 == len(timesteps)
        layer = timesteps[-2]
        print(layer)
        assert 100 == len(layer.points)
        print(layer.status_code_names)
        print(layer.scalar_var_names)
        print(layer.points.x)
        print(layer.points.y)
        print(layer.scalar_vars['surface_concentration'])
        bounds = layer.compute_bounding_rect()
        print(bounds)
        h, xedges, yedges = np.histogram2d(layer.points.x, layer.points.y, bins=21, weights=layer.scalar_vars['surface_concentration'])
        print(h, xedges, yedges)

        # pos = np.vstack([layer.points.x, layer.points.y])
        # print(pos)
        # #values = np.vstack([])
        # kernel = scipy.stats.gaussian_kde(pos, weights=layer.scalar_vars['latitude'])


        # # x, y = np.mgrid[bounds[0][0]:bounds[1][0]:20j, bounds[0][1]:bounds[1][1]:20j]
        # grid = np.vstack([x, y])
        # print(grid)
        # values = kernel(grid)
        # print(values)

        x = layer.points.x - layer.points.x.min()
        y = layer.points.y - layer.points.y.min()
        xy = np.vstack([x, y])

        weights = layer.scalar_vars['surface_concentration']
        weights -= weights.min()
        weights /= weights.max() - weights.min()
        total_weight = weights.sum()
        print(xy)
        print(weights)
        kernel = scipy.stats.gaussian_kde(xy, weights=weights/total_weight) 

        # Attempt from https://stackoverflow.com/questions/4128699/using-scipy-stats-gaussian-kde-with-2-dimensional-data
        # x_flat = np.r_[xy[:,0].min():xy[:,0].max():16j]
        # y_flat = np.r_[xy[:,1].min():xy[:,1].max():16j]
        x_flat = np.linspace(x.min(), x.max(), 17)
        y_flat = np.linspace(y.min(), y.max(), 17)
        x,y = np.meshgrid(x_flat,y_flat)
        grid_coords = np.append(x.reshape(-1,1),y.reshape(-1,1),axis=1)
        print(grid_coords.T)

        values = kernel(grid_coords.T) / total_weight
        print(values.reshape(17,17))



        # xyBnds  = np.linspace(-1.0, 1.0, N+1)  #boundaries of histogram bins
        # xy  = (xyBnds[1:] + xyBnds[:-1])/2      #centers of histogram bins
        # xx, yy = np.meshgrid(xy,xy)

        # #DEFINE SAMPLES, TWO OPTIONS
        # #samples = rv.rvs(size=(nSamp,2))
        # samples = np.array([[0.5,0.5],[0.2,0.5],[0.2,0.2]])

        # #points to sample the KDE at, in a form gaussian_kde likes:
        # grid_coords = np.append(xx.reshape(-1,1),yy.reshape(-1,1),axis=1)

        # #NON-FFT IMPLEMTATION FROM SCIPY
        # KDEfn = scipy.gaussian_kde(samples.T)
        # KDE2 = KDEfn(grid_coords.T).reshape((N,N))





if __name__ == "__main__":
    t = TestContour()
    t.setup()
    t.test_simple()
