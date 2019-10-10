import os

import numpy as np
import scipy.stats

from mock import maproom_dir, MockProject

from maproom import loaders
from maproom.library.Boundary import Boundaries, PointsError
from maproom.library.contour_utils import contour_layer, contour_layer_to_line_layer_data

INTERACTIVE = False


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
        bounds = layer.compute_bounding_rect()
        print(bounds)

        xmin = layer.points.x.min()
        ymin = layer.points.y.min()
        x = layer.points.x - xmin
        y = layer.points.y - ymin
        xy = np.vstack([x, y])

        weights = layer.scalar_vars['surface_concentration']
        total_weight = weights.sum()
        kernel = scipy.stats.gaussian_kde(xy, weights=weights)

        binsize = 101
        x_flat = np.linspace(x.min(), x.max(), binsize)
        y_flat = np.linspace(y.min(), y.max(), binsize)
        xx,yy = np.meshgrid(x_flat,y_flat)
        grid_coords = np.append(xx.reshape(-1,1),yy.reshape(-1,1),axis=1)

        values = kernel(grid_coords.T) * total_weight
        values = values.reshape(binsize,binsize)

        max_density = values.max()
        levels = [0.1, 0.4, 0.8, 1]
        levels.sort()
        particle_contours = [lev * max_density for lev in levels]

        if INTERACTIVE:
            try:
                import matplotlib.pyplot as plt
            except ImportError:
                pass
            else:
                plt.contour(xx + xmin, yy + ymin, values, particle_contours)
                plt.scatter(x + xmin, y + ymin, 10, weights)
                plt.show()

        print(xy)
        print(weights)
        print(grid_coords.T)
        print(values)
        print(particle_contours)

        try:
            import py_contour
        except ImportError:
            pass
        else:
            segs = py_contour.contour(values, x_flat, y_flat, particle_contours)
            print(x_flat)
            print(y_flat)
            print(segs)
            print(segs.keys())
            print(particle_contours)
            for level in particle_contours:
                if level in segs:
                    print(level)
                    for i, seg in enumerate(segs[level]):
                        print("  ", level, i, seg[0][0]+xmin, seg[0][1]+ymin, seg[1][0]+xmin, seg[1][1]+ymin)

            # import matplotlib.pyplot as plt
            # from matplotlib import collections as mc
            # fig, ax = plt.subplots()
            # for level in particle_contours:
            #     if level in segs:
            #         lc = mc.LineCollection(segs[level], linewidths=2)
            #         ax.add_collection(lc)
            # # plt.scatter(x + xmin, y + ymin, 10, weights)
            # plt.show()

    def test_library(self):
        timesteps = self.folder.get_particle_layers()
        print(len(timesteps))
        assert 25 == len(timesteps)
        layer = timesteps[-2]
        print(layer)
        assert 100 == len(layer.points)
        print(layer.status_code_names)
        print(layer.scalar_var_names)
        segs, bbox = contour_layer(layer, 'surface_concentration')
        for level in segs.keys():
            print(level)
            for i, seg in enumerate(segs[level]):
                print("  ", level, i, seg[0][0]+bbox[0][0], seg[0][1]+bbox[0][1], seg[1][0]+bbox[0][0], seg[1][1]+bbox[0][1])

    def test_library_to_polylines(self):
        timesteps = self.folder.get_particle_layers()
        print(len(timesteps))
        assert 25 == len(timesteps)
        layer = timesteps[-2]
        print(layer)
        assert 100 == len(layer.points)
        print(layer.status_code_names)
        print(layer.scalar_var_names)
        polyline_levels = contour_layer_to_line_layer_data(layer, 'surface_concentration')
        # for level in polyline_levels.keys():
        #     print(level)
        #     for i, p in enumerate(polyline_levels[level]):
        #         print("  ", level, i, p)



if __name__ == "__main__":
    INTERACTIVE = True

    t = TestContour()
    t.setup()
    # t.test_simple()
    # t.test_library()
    t.test_library_to_polylines()
