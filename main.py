# "Surface-from-Gradients: An Approach Based on Discrete Geometry Processing", W. Xie et al., CVPR 2014.
# Implemented by Zhuoyu Yang, Matsushita Lab., Osaka University, Mar. 30th 2020.
import cv2
import os
import sys
import matlab.engine

import numpy as np
import scipy.sparse as sparse

from scipy.io import savemat
from scipy.io import loadmat
from sklearn.preprocessing import normalize

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

class DGP(object):

    def __init__(self, path):
        self.n, self.mask_bg = self.read_normal_map(path) # read normal map and background mask
        self.N = None

        self.ilim, self.jlim = self.mask_bg.shape[0], self.mask_bg.shape[1]
        self.mesh_count = np.sum((~self.mask_bg).astype(np.int32))

        self.vertices, self.vertices_count= self.construct_vertices()
        self.vertices_depth = np.zeros_like(self.vertices, dtype=np.float32)

        self.construct_A()
        self.construct_N()
        self.eng = matlab.engine.start_matlab()

    def read_normal_map(self, path):
        '''
        description:
            read a normal map(jpg, png, bmp, etc.), and convert it to an normalized (x,y,z) form

        input:
            path: path of the normal map

        output:
            n: normalized normal map
            mask_bg: background mask
        '''
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

        return n, mask_bg

    def construct_vertices(self):
        count = 0
        vertices = np.zeros((self.ilim+1,self.jlim+1), dtype="uint")
        for i in range(self.ilim):
            for j in range(self.jlim):
                if ~self.mask_bg[i, j]:
                    if vertices[i, j] == 0:
                        count += 1
                        vertices[i, j] = count
                    if vertices[i, j+1] == 0:
                        count += 1
                        vertices[i, j+1] = count
                    if vertices[i+1, j] == 0:
                        count += 1
                        vertices[i+1, j] = count
                    if vertices[i+1, j+1] == 0:
                        count += 1
                        vertices[i+1, j+1] = count
        return vertices, count

    def construct_A(self):
        row = range(4*self.mesh_count)
        col = []
        data = [1 for i in row]
        for i in range(self.ilim):
            for j in range(self.jlim):
                if ~self.mask_bg[i, j]:
                    col.append(self.vertices[i,j]-1)
                    col.append(self.vertices[i,j+1]-1)
                    col.append(self.vertices[i+1,j]-1)
                    col.append(self.vertices[i+1,j+1]-1)

        A = sparse.coo_matrix((data, (row, col)), shape=(4*self.mesh_count, self.vertices_count))

        savemat('A.mat', {'A': A}) # save A into A.mat, solve_sparse.m use A.mat later

    def construct_b(self):
        b = []
        for i in range(self.ilim):
            for j in range(self.jlim):
                if ~self.mask_bg[i, j]:
                    if self.n[i,j,2]<=0.0871557:
                        b.append(self.vertices_depth[i,j])
                        b.append(self.vertices_depth[i,j+1])
                        b.append(self.vertices_depth[i+1,j])
                        b.append(self.vertices_depth[i+1,j+1])
                    else:
                        c = 0.25*(self.vertices_depth[i,j]+self.vertices_depth[i,j+1]+self.vertices_depth[i+1,j]+self.vertices_depth[i+1,j+1])
                        b.append(c-(self.n[i,j,0]/-2+self.n[i,j,1]/2)/self.n[i,j,2])
                        b.append(c-(self.n[i,j,0]/2+self.n[i,j,1]/2)/self.n[i,j,2])
                        b.append(c-(self.n[i,j,0]/-2+self.n[i,j,1]/-2)/self.n[i,j,2])
                        b.append(c-(self.n[i,j,0]/2+self.n[i,j,1]/-2)/self.n[i,j,2])
        b = np.array(b).reshape(-1, 1)
        savemat('b.mat', {'b': b}) # save b into b.mat, solve_sparse.m use b.mat later

    def construct_N(self):
        N_ = np.eye(4)-0.25
        data = np.array([N_ for i in range(self.mesh_count)])
        indptr = range(self.mesh_count+1)
        indices = range(self.mesh_count)
        N = sparse.bsr_matrix((data,indices,indptr))
        savemat("N.mat", {'N': N}) # save N into N.mat, solve_sparse.m use N.mat later

    def DGP_closed(self):
        self.construct_b()
        x = np.asarray(self.eng.solve_sparse()).reshape(-1) # solve NAx = Nb by Matlab
        for i in range(self.ilim+1):
            for j in range(self.jlim+1):
                if self.vertices[i, j] != 0:
                    self.vertices_depth[i, j] = x[self.vertices[i, j]-1]

        os.remove("A.mat") # remove *.mat file generated when solving NAx = Nb by Matlab
        os.remove("N.mat")
        os.remove("b.mat")

if __name__ == '__main__':

    path = sys.argv[1]
    
    task = DGP(path)
    task.DGP_closed() # DGP step

    write_obj("output.obj", task.vertices_depth, task.vertices) # write obj file
