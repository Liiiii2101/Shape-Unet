import numpy as np
import scipy.ndimage
import scipy.signal
import matplotlib.pyplot as plt
from skimage import color, io
import nibabel as nib
import skimage.util
import skimage.io as io
from skimage import data_dir
import os
import random
import shutil
from skimage import data
from scipy import misc
#from skimage.segmentation.find_boundaries
from numpy import linalg as LA



from scipy import ndimage
def convert_sdf(img_label):
    ''' convert the mask to mask'''
    img_label_bi = (img_label>0.5) -1
    img_label_bii = (img_label_bi == -1)
    final = ndimage.distance_transform_edt(img_label_bii)
    return final 



def contour(trainy):
    ''' to find the contour of masks'''
    
    trainy_contour = find_boundaries(trainy, mode='inner')
    trainy_contour_sdf = convert_sdf(trainy_contour)
    coord = np.where(trainy == 1)
    trainy_contour_sdf[coord[0],coord[1],coord[2],coord[3]] = -trainy_contour_sdf[coord[0],coord[1],coord[2],coord[3]]
    trainy_contour_sdf_mean = np.mean(trainy_contour_sdf,axis=0)
    return trainy_contour_sdf, trainy_contour_sdf_mean


def eigen_matrix(trainy):
    '''This calculation is for non-contour masks, which means there's no negative part
    in the signed distance map'''
    trainy_sdf = convert_sdf(trainy)
    trainy_sdf_mean = np.mean(trainy_sdf,axis=0)
    trainy_sdf_diff = trainy_sdf - trainy_sdf_mean
    size = trainy_sdf_diff.shape
    trainy_flat = np.reshape(trainy_sdf_diff,(size[0],4096))
    covar_matrix = np.dot(trainy_flat.T,trainy_flat)/(size[0]-1)
    
    [w,v] = LA.eig(covar_matrix)
    w = w.real
    v = v.real

    b = np.dot(v.T, trainy_flat.T)
    b = b.T

    return b,v,trainy_sdf_mean
    


def eigen_matrix_contour(trainy):
    
    '''This calculation is for contour masks, which means it's negative inside contour of
     signed distance map'''

    trainy_contour_sdf, trainy_contour_sdf_mean = contour(trainy)

    trainy_contour_sdf_diff = trainy_contour_sdf - trainy_contour_sdf_mean

    size = trainy_contour_sdf_diff.sahpe

    trainy_flat = np.reshape(trainy_contour_sdf_diff,(size[0],4096))

    covar_matrix = np.dot(trainy_flat.T,trainy_flat)/(size[0]-1)

    [w,v] = LA.eig(covar_matrix)
    w = w.real
    v = v.real

    b = np.dot(v.T, trainy_flat.T)
    b = b.T

    return b,v
    





