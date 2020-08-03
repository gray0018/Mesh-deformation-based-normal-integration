# Implemented by Zhuoyu Yang, Matsushita Lab., Osaka University, Mar. 30th 2020.
# Modified at July 20th 2020. Support adding depth prior.
import cv2
import argparse

import numpy as np
import scipy.sparse as sparse

from time import time
from scipy.sparse.linalg import spsolve
from sklearn.preprocessing import normalize

# command line parser
parser = argparse.ArgumentParser(description='Normal Integration by mesh deformation')
parser.add_argument('normal', help='the path of normal map')
parser.add_argument('-d', '--depth', default=None, help='the path of depth prior')
parser.add_argument('--d_lambda', type=int, default=100, help='how much will the depth prior influence the result')
parser.add_argument('-o', '--output', default='output', help='name of the output object and depth map')
parser.add_argument('--vertex_depth', dest='depth_type', action='store_const',
                    const='vertex', default='pixel', help='output vertex depth map, by default pixel depth map')

def write_depth_map(filename, depth, mask, v_mask, depth_type='pixel'):
    if depth_type == 'pixel':
        from scipy.signal import convolve2d
        cov_mask = np.array([0.25,0.25,0.25,0.25]).reshape(2,2)
        depth = convolve2d(depth, cov_mask, mode="valid") # keep the resolution of depth map same with the input normal map
        depth[~mask] = np.nan
    else:
        depth[~v_mask] = np.nan
    np.save(filename, depth)

def write_obj(filename, d, d_ind):
    f = open(filename, "w")

    ind = np.zeros_like(d,dtype=np.int32)
    mask = (d_ind != 0)
    ind[mask] = range(1, np.sum(mask.astype(np.int32))+1)

    h, w = d.shape

    for i in range(h):
        for j in range(w):
            if ind[i, j]:
                f.write("v {0} {1} {2}\n".format(j-0.5, h-(i-0.5), d[i, j]))
    for i in range(h):
        for j in range(w):
            if ind[i, j] and j+1<w and i+1<h:
                if ind[i, j+1] and ind[i+1, j+1]:
                    f.write("f {0} {1} {2}\n".format(ind[i, j], ind[i+1, j+1], ind[i, j+1]))
                if ind[i+1, j] and ind[i+1, j+1]:
                    f.write("f {0} {1} {2}\n".format(ind[i, j], ind[i+1, j], ind[i+1, j+1]))
    f.close()

class Normal_Integration(object):

    def __init__(self, nomral_path, depth_path=None, d_lambda=100): # you can tune d_lambda to decide how much the depth prior will influence the result.
        self.n, self.mask = self.read_normal_map(nomral_path) # read normal map and background mask
        self.N = None # mean-substraction matrix N defined in equation (5), "Surface-from-Gradients: An Approach Based on Discrete Geometry Processing"
        self.d_lambda = d_lambda # lambda for depth prior

        self.ilim, self.jlim = self.mask.shape # size of normal map and mask
        self.mesh_count = (self.mask.astype(np.int32)).sum() # how many pixels in the normal map

        self.lu = np.pad(self.mask, [(0, 1), (0, 1)], mode='constant') # mask for left upper vertices
        self.ru = np.pad(self.mask, [(0, 1), (1, 0)], mode='constant') # mask for right upper vertices
        self.ld = np.pad(self.mask, [(1, 0), (0, 1)], mode='constant') # mask for left down vertices
        self.rd = np.pad(self.mask, [(1, 0), (1, 0)], mode='constant') # mask for right down vertices

        self.v_mask = self.lu|self.ru|self.ld|self.rd # mask for all vertices
        self.v_count = (self.v_mask.astype(np.int32)).sum() # total number of all vertices

        self.v_index = np.zeros_like(self.v_mask, dtype='uint') # indices for all vertices
        self.v_index[self.v_mask] = np.arange(self.v_count)+1

        self.v_depth = np.zeros_like(self.v_index, dtype=np.float32) # depth for the vertices

        self.A, self.b = self.construct_Ax_b_without_depth_prior()

        # below we are going to solve NAx=Nb, let's construct them first
        self.N = self.construct_N()
        self.NA = self.N@self.A
        self.Nb = self.N@self.b

        if depth_path is not None: # add depth prior
            A, b = self.add_depth_prior(depth_path)
            self.NA = sparse.vstack([self.NA, A])
            self.Nb = np.vstack([self.Nb, b])

        # since NA is a tall matrix, we use normal equation NA.T@NA@x=NA.T@Nb to solve this linear system
        self.NATNA = self.NA.T@self.NA
        self.NATNb = self.NA.T@self.Nb

    def add_depth_prior(self, depth_path):

        d = np.load(depth_path) # read depth prior
        d_mask = ~np.isnan(d) # mask for depth prior
        d_count = (d_mask.astype(np.int32)).sum() # total number of depth prior

        # scipy.sparse.coo_matrix((data, (row, col))
        # build row for sparse coo matrix
        row = np.arange(d_count).repeat(4)

        # build col for sparse coo matrix
        lu = np.pad(d_mask, [(0, 1), (0, 1)], mode='constant') # mask for left upper vertices
        ru = np.pad(d_mask, [(0, 1), (1, 0)], mode='constant') # mask for right upper vertices
        ld = np.pad(d_mask, [(1, 0), (0, 1)], mode='constant') # mask for left down vertices
        rd = np.pad(d_mask, [(1, 0), (1, 0)], mode='constant') # mask for right down vertices

        col = np.vstack((self.v_index[lu],self.v_index[ru],self.v_index[ld],self.v_index[rd])).T.ravel()

        # build data for sparse coo matrix
        data = np.ones(4*d_count)
        data = 1/4*data*self.d_lambda
        
        # build A for Ax=b
        A = sparse.coo_matrix((data, (row, col)), shape=(d_count, self.v_count)).tocsc()
        # build b for Ax=b
        b = d[d_mask].reshape(-1,1)*self.d_lambda
        
        return A, b

    def construct_Ax_b_without_depth_prior(self):

        # scipy.sparse.coo_matrix((data, (row, col))
        # build row for sparse coo matrix
        row = np.arange(4*self.mesh_count)
        # build col for sparse coo matrix
        col = np.vstack((self.v_index[self.lu],self.v_index[self.ru],self.v_index[self.ld],self.v_index[self.rd])).T.ravel()
        # build data for sparse coo matrix
        data = np.ones(4*self.mesh_count)

        x = self.n[...,0][self.mask]
        y = self.n[...,1][self.mask]
        z = -self.n[...,2][self.mask]
        # build A for Ax=b
        A = sparse.coo_matrix((data, (row, col-1)), shape=(4*self.mesh_count, self.v_count)).tocsc()
        # build b for Ax=b
        b = np.vstack(((x/-2+y/2)/z,(x/2+y/2)/z,(x/-2+y/-2)/z,(x/2+y/-2)/z)).T.reshape(-1, 1)

        return A, b

    def read_normal_map(self, path):
        '''
        description:
            read a normal map(jpg, png, bmp, etc.), and convert it to an normalized (x,y,z) form

        input:
            path: path of the normal map

        output:
            n: normalized normal map
            mask: normal map mask
        '''

        if ".npy" in path:
            n = np.load(path)
            mask_bg = (n[...,2] == 0) # get background mask

        else:
            n = cv2.imread(path)

            n[...,0], n[...,2] = n[...,2], n[...,0].copy() # Change BGR to RGB
            mask_bg = (n[...,2] == 0) # get background mask
            n = n.astype(np.float32) # uint8 -> float32

            # x,y:[0,255]->[-1,1] z:[128,255]->[0,1]
            n[...,0] = n[...,0]*2/255-1
            n[...,1] = n[...,1]*2/255-1
            n[...,2] = (n[...,2]-128)/127

            n = normalize(n.reshape(-1,3)).reshape(n.shape)

        # fill background with [0,0,0]
        n[mask_bg] = [0,0,0]
        return n, ~mask_bg

    def construct_N(self):
        N_ = np.eye(4)-0.25
        data = np.array([N_ for i in range(self.mesh_count)])
        indptr = range(self.mesh_count+1)
        indices = range(self.mesh_count)
        N = sparse.bsr_matrix((data,indices,indptr))
        return N.tocsc()

    def mesh_deformation(self):
        x = spsolve(self.NATNA, self.NATNb) # solve NATNAx = NATNb by SciPy
        self.v_depth[self.v_mask] = x

if __name__ == '__main__':

    args = parser.parse_args()

    start = time()

    print("Start reading input data...")
    task = Normal_Integration(args.normal, args.depth, args.d_lambda)
    print("Start normal integration...")
    task.mesh_deformation()

    end = time()
    print("Time elapsed: {:0.2f}".format(end - start))
    np.save(args.output+'_woloop_time', end - start)

    print("Start writing obj file...")
    write_obj("{0}.obj".format(args.output), task.v_depth, task.v_index) # write obj file
    print("Start writing depth map...")
    write_depth_map("{0}_depth.npy".format(args.output), task.v_depth, task.mask, task.v_mask, depth_type=args.depth_type) # write depth file
    print("Finish!")
