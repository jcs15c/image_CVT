from scipy import misc
import numpy as np
from numpy.linalg import norm
import math as m
from random import randint

#Loads image and returns array of RGB values
def read_image(imname):
    data = misc.imread(imname, False, "RGB") 
    return data

#Convert CVT data into viewable image
def cvt_render(imdata, generators, weights, numw):
    imshape = imdata.shape        #Stores [# of rows, # of columns, 3]
    genshape = generators.shape   #       [# of generators, 3]

    imdata_new = np.zeros(imshape)
    
    #loop over all pixels of original image
    for i in range(0, imshape[0]):
        for j in range(0, imshape[1]):
            
            min_dist = float("inf")
            k_opt = 0
            
            # loop over generators
            for k in range(genshape[0]):
                dist = (weights[k]**numw) * \
                            norm(imdata[i,j] - generators[k])**2
                
                #Compute least energy from pixel to generator
                if dist < min_dist:
                    min_dist = dist
                    k_opt = k
                    
            #Give new image generator color
            imdata_new[i,j] = generators[k_opt]

    return imdata_new

#Performs a single CVT iteration
def cvt_step(imdata, generators_old, weights_old, numw, it):
    empty_gen = False                  #Flag to show empty generator
    
    imshape = imdata.shape             #Stores [# of rows, # of columns, 3]
    genshape = generators_old.shape    #       [# of generators, 3]
    
    bins = np.zeros([genshape[0],3])   #For computing centroids
    bin_count = np.zeros(genshape[0])  #Also serves as weights
    
    generators_new = np.zeros([genshape[0],3])
    
    #loop over all pixels in original image
    for i in range(0, imshape[0]):
        for j in range(0, imshape[1]):
            
            min_dist = float("inf")
            k_opt = 0
            
            # loop over generators
            for k in range(len(generators_old)):
                
                dist = (weights_old[k]**numw) * \
                            norm(imdata[i,j] - generators_old[k])**2
                
                #Compute least energy from pixel to generator
                if dist < min_dist:
                    min_dist = dist
                    k_opt = k
              
            #Begin computing centroids
            bins[k_opt] += imdata[i,j]
            bin_count[k_opt] += 1

    #Prevents divide by zero error
    for i in range(genshape[0]):
        if bin_count[i] == 0:
            empty_gen = True
            bin_count[i] = 1
            bins[i] = np.random.rand(3)
            
    #Finish computing centroids
    for i in range(genshape[0]):
        generators_new[i] = bins[i] / bin_count[i] 

    #Perform unweighted CVT if a color is absent
    if empty_gen:
        generators_new, bin_count = \
                cvt_step(imdata, generators_new, bin_count, 0, it)
            
    return generators_new, bin_count

#Main CVT manager function
def cvt(imdata, generators, tol, max_iter, numw):
    it = 0
    genshape = generators.shape            #Stores [# of generators, 3]
    weights = np.zeros(genshape[0]) + 1    #Creates initial weights
   
    E = []
    E.append(compute_energy(imdata, generators, weights, numw))
    dE = float("inf")
    
    #Repeats CVT iterations
    while (it < max_iter and dE > tol):
        generators, weights = cvt_step(imdata, generators, weights, numw, it)      
        it += 1  
        
        E.append(compute_energy(imdata, generators, weights, numw))
        dE = abs(E[it] - E[it-1])/E[it]    #Compute change in energy

    return generators, weights, E
    
#Computes total energy of a set of generators
def compute_energy(imdata, generators, weights, numw):
    imshape  = imdata.shape         #Stores [# of rows, # of columns, 3]
    genshape = generators.shape     #       [# of generators, 3]
    energy =  0

    #loop over all pixels in original image
    for i in range(0, imshape[0]):
        for j in range(0, imshape[1]):
            
            min_dist = float("inf")
            
            # loop over generators
            for k in range(genshape[0]):
                dist = (weights[k]**numw) * \
                            norm(imdata[i,j] - generators[k])**2
                
                #Compute least energy from pixel to generator
                if dist < min_dist:
                    min_dist = dist
                    
            energy += min_dist
                    
    return energy

#Image segmentation manager function
def image_segmentation(imdata, generators, tol, max_iter, numw):
    shape = imdata.shape            #Stores [# of rows, # of columns, 3]
    sketch = np.ones(shape[0:2])    #Sketch is two dimensional
    
    #Perform CVT
    generators, weights, evector = cvt(imdata, generators, tol, max_iter, numw)

    #Simplify CVT image
    zones = voronoi_zones(imdata, generators, weights, numw)
    
    #loop over all interior pixels, checks for neighboring clusters
    r, c = shape[0]-1, shape[1]-1   #Max row/collumn index
    for i in range(1, r):
        for j in range(1, c):

            z = zones[i,j]
            if not (z == zones[i-1,j]   and z == zones[i+1,j]   and 
                    z == zones[i,j-1]   and z == zones[i,j+1]   and 
                    z == zones[i-1,j-1] and z == zones[i-1,j+1] and
                    z == zones[i+1,j-1] and z == zones[i-1,j+1]):
                sketch[i,j] = 0
    
    #Check 4 corners
    z = zones[0,0]            
    if not (z == zones[1,0] and z == zones[0,1] and z == zones[1,1]):
        sketch[0,0] = 0
        
    z = zones[r,0]
    if not (z == zones[r,1] and z == zones[r-1,1] and z == zones[r-1,0]):
        sketch[r,0] = 0
        
    z = zones[0,c]
    if not (z == zones[0,c-1] and z == zones[1,c-1] and z == zones[1,c]):
        sketch[0,c] = 0

    z = zones[r,c]
    if not (z == zones[r,c-1] and z == zones[r-1,c-1] and z == zones[r-1,c]):
        sketch[r,c] = 0
        
    #Check vertical edges
    for i in range(1,r):
        z = zones[i,0]
        if not (z == zones[i-1,0] and z == zones[i-1,1] and
                z == zones[i,1]   and z == zones[i+1,1] and
                z == zones[i+1,0]):
            sketch[i,0] = 0   
    
        z = zones[i,c]
        if not (z == zones[i-1,c]   and z == zones[i-1,c-1] and
                z == zones[i,c-1]   and z == zones[i+1,c-1] and
                z == zones[i+1,c]):
            sketch[i,c] = 0   
    
    #Check horizontal edges
    for i in range(1,c):
        z = zones[0,i]
        if not (z == zones[0,i-1] and z == zones[1,i-1] and
                z == zones[1,i]   and z == zones[1,i+1] and
                z == zones[0,i+1]):
            sketch[0,i] = 0  

        z = zones[r,i]
        if not (z == zones[r,i-1] and z == zones[r-1,i-1] and
                z == zones[r-1,i] and z == zones[r-1,i+1] and
                z == zones[r,i+1]):
            sketch[r,i] = 0 
        
    return sketch, generators, weights, evector
 
#Simplifies CVT data into 2D array
def voronoi_zones(imdata, generators, weights, numw):
    shape = imdata.shape                #Stores [# of rows, # of columns, 3]
    zone_data = np.zeros(shape[0:2])

    #loop over all pixels
    for i in range(0, shape[0]):
        for j in range(0, shape[1]):

            min_dist = float("inf")
            k_opt = 0
            
            # loop over generators
            for k in range(len(generators)):
                dist = (weights[k]**numw) * \
                    norm(imdata[i,j] - generators[k])**2

                #Find Generator with least energy                    
                if dist < min_dist:
                    min_dist = dist
                    k_opt = k

            #Create array with generator index, not generator itself
            zone_data[i,j] = k_opt			

    return zone_data
 
#Averages image based on radial, weighted average
def smoothing_avg(imdata, s, b):
    shape = imdata.shape            #Stores [# of rows, # of columns, 3]
    imdata_new = np.zeros(shape)
    #rad = -s**2 * m.log(b)         #For use with Gaussian average
    rad = s / b                     #For use with Computational Average
    #print("Averaging with radius of", rad)  #Unneccesary printout
    
    #Loop over each pixel in original image
    for i in range(0, shape[0]):
        for j in range(0, shape[1]):
            #Calculate average based on radius and smoothing parameter
            imdata_new[i,j] = pxl_avg(imdata, i, j, rad, c_avg, s)

    return imdata_new
   
#Two possible averaging functions
def gauss(x,y,s):
    return m.exp(-(x**2 + y**2)/(s**2))
def c_avg(x,y,s):
    return s / (2*m.pi*(x**2 + y**2 + s**2)**(1.5))

#Takes in index of a single pixel and averages it according to function
def pxl_avg(imdata, xp, yp, r, avgf, s):
    shape = imdata.shape
    numer = np.zeros(3)
    denom = 0

    #Loops in largest square around circle radius
    for rx in range(m.floor(-r), m.ceil(r+1)):
        for ry in range(m.floor(-r), m.ceil(r+1)):
            xpxl = xp + rx   
            ypxl = yp + ry
            
            #Only consider pixel if it is inside image and circle
            if xpxl >= 0 and xpxl < shape[0] and \
               ypxl >= 0 and ypxl < shape[1] and \
               r**2 >= rx**2 + ry**2 :
                
                   weight = avgf(rx, ry, s)
                   numer += imdata[xpxl,ypxl] * weight
                   denom += weight
                   
    pxl_avg = numer / denom
    return pxl_avg

#Takes an array of image data and computes combined CVT
def multi_channel(imdata_arr, generators, max_iter):   
    imlen = []              #Code to find smallest length and width of
    imwid = []              # inputted arrays, and creates image with 
    for l in range(len(imdata_arr)):#  those dimensions
        imlen.append((imdata_arr[l].shape)[0])
        imwid.append((imdata_arr[l].shape)[1])
    imshape = (min(imlen),min(imwid),3)
                                         
    imdata_new = np.zeros((min(imlen),min(imwid),3))    #Stores final image

    it = 0
 
    #Repeats CVT iterations
    while (it < max_iter):
        generators = mc_cvt_step(imdata_arr, imshape, generators, max_iter)      
        it += 1  
        #print("Iteration " + str(it))  #Unnecessary printout

    #Renders completed CVT image
    for i in range(0, imshape[0]):
        for j in range(0, imshape[1]):
            
            min_dist = float("inf")
            k_opt = 0
            
            # loop over generators
            for k in range(len(generators)):
                
                dist = 0
                for l in range(len(imdata_arr)):
                    dist += norm(imdata_arr[l][i,j] - generators[k])**2   
                
                #Compute least energy from pixel to generator
                if dist < min_dist:
                    min_dist = dist
                    k_opt = k
              
            #Place generator colors into new image
            imdata_new[i,j] = generators[k_opt]
    return imdata_new

#Performs CVT iteration on array of images
def mc_cvt_step(imdata_arr, imshape, generators_old, max_iter):    
    genshape = generators_old.shape    #  [# of generators, 3]
    
    bins = np.zeros([genshape[0],3])   #For computing centroids
    bin_count = np.zeros(genshape[0])  #  as an average of points
        
    generators_new = np.zeros([genshape[0],3])
    
    #loop over all pixels in original image
    for i in range(0, imshape[0]):
        for j in range(0, imshape[1]):
            
            min_dist = float("inf")
            k_opt = 0
            
            # loop over generators
            for k in range(len(generators_old)):
                
                dist = 0
                for l in range(len(imdata_arr)):
                    dist += norm(imdata_arr[l][i,j] - generators_old[k])**2   
            
                #Compute least energy from pixel to generator
                if dist < min_dist:
                    min_dist = dist
                    k_opt = k
              
            #Begin computing centroids
            for l in range(len(imdata_arr)):
                bins[k_opt] += imdata_arr[l][i,j]
                bin_count[k_opt] += 1  

    #Prevents divide by zero error
    for j in range(genshape[0]):        
        if bin_count[j] == 0:
            bin_count[j] = 1
            bins[j] = np.mean(generators_old, axis = 0)

    #Finish computing centroids
    for i in range(genshape[0]):
        generators_new[i] = (bins[i] / bin_count[i])
        
    return generators_new
   
#Returns an optimal set of initial generator points
def plusplus(imdata, numgen):
    ppgens = np.zeros([numgen, 3]) - 1 
    counter = 0
    imshape = imdata.shape
    
    ppgens[0] = imdata[randpix(imshape)]
    counter += 1

    while (ppgens[numgen-1] == [-1,-1,-1]).all():
        dist = []
        randpt = imdata[randpix(imshape)]         #Picks random point in image
        samppt = np.random.rand() * m.sqrt(3)     #Picks sample point
        
        for c in ppgens:                #Computes distance from random point
            if (c != [-1,-1,-1]).all(): #  to each other generator
               dist.append(norm( c - randpt ) )
                                       #rejection sampling of shortest distance
        if ( ( min(dist)**2 / 3 ) > samppt ):
            ppgens[counter] = randpt
            counter += 1          
            
    return ppgens
    
def randpix(imshape):
    return randint(0,imshape[0]-1), randint(0,imshape[1]-1)
#################################################
####       Obsolete Averaging Algorithm      #### 
#################################################
 
def average_image(imdata):
    shape = imdata.shape #shape[0] = # rows, shape[1] = # columns
    imdata_new = np.ones(shape)
    r, c = shape[0]-1, shape[1]-1 #max row/collumn index

    for i in range(1, r):
        for j in range(1, c):
            nz = np.array([imdata[i-1,j], imdata[i+1,j],  imdata[i,j-1], 
                          imdata[i,j+1], imdata[i-1,j-1],imdata[i-1,j+1],
                          imdata[i+1,j-1], imdata[i-1,j+1], imdata[i,j]])
            imdata_new[i,j] = np.mean(nz, axis = 0)
    
    #Check 4 corners
    imdata_new[0,0] = np.mean(np.array([imdata[0,0],imdata[1,0],imdata[0,1],imdata[1,1]]), axis=0)
    imdata_new[r,0] = np.mean(np.array([imdata[r,0],imdata[r,1],imdata[r-1,1],imdata[r-1,0]]), axis=0)
    imdata_new[0,c] = np.mean(np.array([imdata[0,c],imdata[0,c-1],imdata[1,c-1],imdata[1,c]]), axis=0)
    imdata_new[r,c] = np.mean(np.array([imdata[r,c],imdata[r,c-1],imdata[r-1,c-1],imdata[r-1,c]]), axis=0)
    
    #Check vertical edges
    for i in range(1,r):
        nz = np.array([imdata[i-1,0], imdata[i-1,1], imdata[i,1],
                       imdata[i+1,1], imdata[i+1,0], imdata[i,0]])  
        imdata_new[i,0] = np.mean(nz, axis = 0)

        nz = np.array([imdata[i-1,c], imdata[i-1,c-1], imdata[i,c-1],
                       imdata[i+1,c-1], imdata[i+1,c], imdata[i,c]])
        imdata_new[i,c] = np.mean(nz, axis = 0)   
    
    #Check horizontal edges
    for i in range(1,c):
        nz = np.array([imdata[0,i-1], imdata[1,i-1], imdata[1,i],
                       imdata[1,i+1], imdata[0,i+1], imdata[0,i]])
        imdata_new[0,i] = np.mean(nz, axis = 0)  

        nz = np.array([imdata[r,i-1], imdata[r-1,i-1], imdata[r-1,i],
                       imdata[r-1,i+1], imdata[r,i+1], imdata[r,i]])
        imdata_new[r,i] = np.mean(nz, axis = 0) 
 
    return imdata_new
