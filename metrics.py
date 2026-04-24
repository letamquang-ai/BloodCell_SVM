import mgcreate as mgc
import scipy.ndimage as nd
import scipy.signal as ss
import numpy as np
import skimage as sk
from sklearn import cluster
from skimage import filters
from skimage import feature
from skimage.color import label2rgb
from skimage.feature import corner_harris, corner_peaks
import os
import cv2
import math
import colorsys

#convert image to square centered at center of image
#assuming input has the shape as black
def convert(fname, bname):
    print(fname)
    adata = bin_edges(fname=fname, bname=bname)
    #plopping into square centered at center of image with max of original image
    d = max(adata.shape)*2
    pic = mgc.Plop(adata, (d, d), 0)
    #swapping black and white
    new = pic
    # find the center of mass
    v = nd.center_of_mass(new)
    v = (int(v[0]), int(v[1]))
    #finding cetner of new image
    n = (d // 2, d // 2)
    #shift in horizontal
    hori = v[1]-n[1]
    #shift in vertical
    vert = v[0]-n[0]
    #shift the image
    ultima = nd.shift(new, (-vert, -hori), cval=0)
    return ultima

#finding minimum encompassing circle - needs to be binary (1 and 0)
def enc_circ(pic):
    ultima = pic + 0
    v,h = ultima.shape
    z = np.ones((v,h))+0
    z[v//2,h//2] = 0
    dist = nd.distance_transform_edt(z)
    vals = dist*ultima
    r = vals.max()
    r = math.ceil(r)
    ultima = ultima[(v//2 - r) : r + v//2, h//2 -r : r + h//2]
    return ultima

#from page 266 of Kinser (2018)
#gives some shape metrics
#this version only provides eccentricity
def Metrics(orig):
    v, h = orig.nonzero()
    mat = np.zeros((2, len(v)))
    mat[0] = v
    mat[1] = h
    evls, evcs = np.linalg.eig(np.cov(mat))
    eccen = evls[0]/evls[1]
    if eccen < 1: eccen = 1/eccen
    return eccen, evls[0], evls[1]

def Shapes(pic):
    #setup
    shapes = np.zeros((1,14))
    #obtain circularity
    circ = ( sum(sum((nd.binary_dilation(pic , iterations=1) - pic ))) **2) /(4*np.pi*sum(sum(pic)))
    shapes[0][0] = circ
    #provides eccentricity, eigen1 and eigen2
    eccen, e1, e2 = Metrics(pic)
    shapes[0][1] = eccen
    shapes[0][2] = e1
    shapes[0][3] = e2
    #number of corners
    corners = corner_harris(pic, k=0.1, sigma=2)
    shapes[0][4] = corner_peaks(corners, min_distance=1).shape[0]
    #white and black pixel counts for min bounding box
    slice_x, slice_y = nd.find_objects(pic==1)[0]
    roi = pic[slice_x, slice_y]
    shapes[0][5] = np.unique(roi, return_counts=True)[1][1]
    shapes[0][6] = np.unique(roi, return_counts=True)[1][0]
    #calculting moments from data
    m = sk.measure.moments(pic)
    #centroid
    centroid = (m[0, 1] / m[0, 0], m[1, 0] / m[0, 0])
    #central moments
    mu = sk.measure.moments_central(pic, centroid)
    #normalizing moments
    nu = sk.measure.moments_normalized(mu)
    #calculting hu moments
    shapes[0][7:14] = sk.measure.moments_hu(nu)
    return shapes

#performs needed setup for other functions
#also provides binary edges info
#must use first
def bin_edges(fname, bname):
    # read in image as grayscale
    adata = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)

    # find edges
    gdata = np.abs(np.gradient(adata))
    val = 1.0
    ddata = ((gdata[0] > val) + (gdata[1] > val) + 0.0) + 0.0

    # fill holes
    shape = nd.binary_fill_holes(ddata + 0.0) + 0.0

    # get largest shape
    b, n = nd.label(shape + 0.0)

    simp = np.hstack(b)
    locs = np.nonzero(simp)
    counts = np.bincount(simp[locs])
    vals = counts.argsort()[-3:][::-1]

    clist = list(map(lambda x: b == x, (0, vals[0])))
    shape = (clist[1] + 0)

    # save image
    cv2.imwrite(bname + fname[4:], (shape * 255).astype(np.uint8))

    return shape

#fname='684530777.jpg'
#bname='data_binary/'

#getting all .jpeg's
def GetPicNames( indir ):
    a = os.listdir( indir )
    pgmnames= []
    for t in a:
        if '.jpeg' in t:
            pgmnames.append( indir + '/' + t )
    return pgmnames

#obtaining images
def GetAllImages( dirs, bname ):
    mgs = []
    for d in dirs:
        mgnames = GetPicNames( d )
        for j in mgnames:
            b = convert(j, bname)
            c = enc_circ(b)
            mgs.append( c )
    return mgs

#obtaining images for shape metrics except EIs
def GetAllImages_Shapes(dirs):
    mgs = []
    for d in dirs:
        mgnames = GetPicNames(d)
        for j in mgnames:
            b = cv2.imread(j, cv2.IMREAD_GRAYSCALE)

            b = (b > 125.0) + 0.0
            c = Shapes(b)
            mgs.append(c)

    return mgs

#histogram conversion for 256 intensities.
def GSHistogram(dirs, bname):
    imgs = []
    imgs.append(GetAllImages(dirs, bname)) #images obtained from directory
    hist = np.zeros( (len(imgs[0]),2) )
    #get histogram values
    for i in range(0,(len(imgs[0]))):
        temp = imgs[0][i].ravel()
        temp = (temp < 256/2) + 0
        hist[i] = np.array(np.histogram(temp, bins=range(0, 3))[0])
    #return vals
    return hist

#gets binary image histogram
def BinaryHist(dirs, bname):
    imgs = []
    imgs.append(GetAllImages(dirs, bname))
    hist = np.zeros( (len(imgs[0]),2) )
    #get histogram values
    for i in range(0,(len(imgs[0]))):
        hist[i][0] = imgs[0][i].sum()
        hist[i][1] = (imgs[0][i].shape[0]*imgs[0][i].shape[1]) - hist[i][0]
    #return vals
    return hist

#save histogram as .txt where first column is white counts
#and second column is black counts
def BinaryHistTXT(tname, dirs, bname):
    #obtain histogram
    hist = BinaryHist(dirs, bname)
    #get image names
    names = GetPicNames( dirs[0] )
    #save as txt
    np.savetxt(tname, hist, delimiter=',', header="white,black", comments='')

#save histogram as .txt where first column is white counts
#and second column is black counts
def BinaryShapesTXT(tname, dirs):
    #obtain shape metrics
    shapes = GetAllImages_Shapes(dirs)
    #get image names
    name6 = tname + "_SHAPES.txt"
    #save as txt
    np.savetxt(name6, np.vstack(shapes), delimiter=',', header="Shape_circ, Shape_eccent, Shape_e1, Shape_e2, Shape_corn, White_box, Black_box, Hu1, Hu2, Hu3, Hu4, Hu5, Hu6, Hu7", comments='')