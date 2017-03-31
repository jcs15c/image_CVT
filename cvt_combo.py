from scipy import misc
import numpy as np
from numpy.linalg import norm
import math as m
from random import randint

#Loads image and returns array of RGB values and geometric space
def read_image_combo(imname):
    c_data = misc.imread(imname, False, "RGB")
    imshape = c_data.shape
    g_data = np.zeros([imshape[0], imshape[1] , 2] )
    counter = 0
    for i in range(imshape[0]):
        for j in range(imshape[1]):
            g_data[i,j] = [i*256/imshape[0], j*256/imshape[0]]
            counter += 1

    return c_data, g_data
    
#Convert CVT data into viewable image
def cvt_render_combo(c_data, g_data, c_gens, g_gens, colw, geow):
    imshape = c_data.shape        #Stores [# of rows, # of columns, 3]
    genshape = c_gens.shape   #       [# of generators, 3]

    data_new = np.zeros([imshape[0], imshape[1], 3])
    
    #loop over all pixels of original image
    for i in range(0, imshape[0]):
        for j in range(0, imshape[1]):
            min_dist = float("inf")
            k_opt = 0
            
            # loop over generators
            for k in range(genshape[0]):
                dist = norm(c_data[i,j] - c_gens[k])**colw + \
                       norm(g_data[i,j] - g_gens[k])**geow
                
                #Compute least energy from pixel to generator
                if dist < min_dist:
                    min_dist = dist
                    k_opt = k
                    
            #Give new image generator color
            data_new[i,j] = c_gens[k_opt]

    return data_new

    
#Performs a single CVT iteration
def cvt_step_combo(c_data, g_data, c_gens, g_gens, it, colw, geow):
    energy = 0    
    imshape = c_data.shape             #Stores [# of rows, # of columns, 3]
    
    c_genshape = c_gens.shape    #       [# of generators, 3]
    g_genshape = g_gens.shape

    c_bins = np.zeros(c_genshape)   #For computing centroids
    g_bins = np.zeros(g_genshape)   #For computing centroids

    c_bin_count = np.zeros(c_genshape[0])  #Also serves as weights
    g_bin_count = np.zeros(g_genshape[0])  #Also serves as weights
    
    c_gens_new = np.zeros(c_genshape)
    g_gens_new = np.zeros(g_genshape)
    
    #loop over all pixels in original image
    for i in range(0, imshape[0]):
        for j in range(0, imshape[1]):
            
            min_dist = float("inf")
            k_opt = 0
            
            # loop over generators
            for k in range(len(c_gens)):
                dist = norm(c_data[i,j] - c_gens[k])**colw + \
                       norm(g_data[i,j] - g_gens[k])**geow
 #               print(norm(c_data[i,j] - c_gens[k])**2,norm(g_data[i,j] - g_gens[k])**geow)
                #Compute least energy from pixel to generator
                if dist < min_dist:
                    min_dist = dist
                    k_opt = k
              
            energy += min_dist
            #Begin computing centroids
            c_bins[k_opt] += c_data[i,j]
            c_bin_count[k_opt] += 1

            g_bins[k_opt] += g_data[i,j]
            g_bin_count[k_opt] += 1

    #Prevents divide by zero error
    for i in range(c_genshape[0]):
        if c_bin_count[i] == 0:
            c_bin_count[i] = 1
            c_bins[i] = np.random.rand(c_genshape[1])

        if g_bin_count[i] == 0:
            #empty_gen = True
            g_bin_count[i] = 1
            g_bins[i] = np.random.rand(g_genshape[1])
            
    #Finish computing centroids
    for i in range(c_genshape[0]):
        c_gens_new[i] = c_bins[i] / c_bin_count[i] 
        g_gens_new[i] = g_bins[i] / g_bin_count[i] 
            
    return c_gens_new, g_gens_new, energy

#Main CVT manager function
def cvt_combo(c_data, g_data, c_gens, g_gens, tol, max_iter, colw, geow):
    it = 0    
    
    shape = c_data.shape           #Stores [# of rows, # of columns, 3]
    sketch = np.ones(shape[0:2])    #Sketch is two dimensional

    E = []
    E.append(float("inf"))
    dE = float("inf")
    
    #Repeats CVT iterations
    while (it < max_iter and dE > tol):
        c_gens, g_gens, energy = cvt_step_combo(c_data, g_data, c_gens, g_gens, it, colw, geow)      
        it += 1  
        
        E.append(energy)
        dE = abs(E[it] - E[it-1])/E[it]    #Compute change in energy

    #Simplify CVT image
    zones = voronoi_zones(c_data, g_data, c_gens, g_gens, colw, geow)
    
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

    return c_gens, g_gens, sketch#, E
 
#Simplifies CVT data into 2D array
def voronoi_zones(c_data, g_data, c_gens, g_gens, colw, geow):
    shape = c_data.shape                #Stores [# of rows, # of columns, 3]
    zone_data = np.zeros(shape[0:2])

    #loop over all pixels
    for i in range(0, shape[0]):
        for j in range(0, shape[1]):

            min_dist = float("inf")
            k_opt = 0
            
            # loop over generators
            for k in range(len(c_gens)):
                dist = norm(c_data[i,j] - c_gens[k])**colw + \
                       norm(g_data[i,j] - g_gens[k])**geow

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
def plusplus(imdata, numgen, dim = 3, alpha = 2):
    ppgens = np.zeros([numgen, dim]) - 1 
    blank = np.zeros(dim) - 1
    counter = 0
    imshape = imdata.shape
    
    ppgens[0] = imdata[randpix(imshape)]
    counter += 1

    while (ppgens[numgen-1] == blank).all():
        dist = []
        randpt = imdata[randpix(imshape)]         #Picks random point in image
        samppt = np.random.rand() * m.sqrt(dim)     #Picks sample point
        
        for c in ppgens:                #Computes distance from random point
            if (c != blank).all(): #  to each other generator
               dist.append(norm( c - randpt ) )
                                       #rejection sampling of shortest distance
        if ( ( min(dist)**alpha / m.sqrt(dim) ) > samppt ):
            ppgens[counter] = randpt
            counter += 1          
            
    return ppgens
    
def randpix(imshape):
    return randint(0,imshape[0]-1), randint(0,imshape[1]-1)