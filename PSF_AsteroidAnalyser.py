"""
This program reads in .FITS files digitized from the ROE Plate Archive
It then builds a PSF model from the background stars
which can be fit to the asteroid line profile with a least squares fit.

This will determine whether statistically significant activity is present.

MPhys
2022/23
Denis Gergov S1839787
"""
from astropy.io import fits
from astropy.table import Table
from astropy.nddata import NDData
from astropy import wcs
from astropy.visualization import simple_norm
from astropy.time import Time

from photutils.psf import EPSFBuilder
from photutils.detection import find_peaks
from photutils.psf import extract_stars
from photutils.background.background_2d import Background2D
from photutils.background import BkgZoomInterpolator

import matplotlib.pyplot as plt

import scipy.interpolate
from scipy.ndimage.interpolation import rotate

import numpy as np
#print( cv2.__version__ )
#photutils.test()
visual_check = False

def ra_dec_to_degrees(ra, dec):
    #must convert ra and dec into decimal coordinates
    raN = (ra[0] + ra[1]/60 + ra[2]/3600)*15 
    decN = abs(dec[0]) + abs(dec[1])/60 + abs(dec[2])/3600 
        
    if dec[0] < 0 or dec[1] < 0 or dec[2] < 0:
        decN = (-1)*decN
    
    return raN, decN

def monthToNum(Month):
    return {'Jan': 1,
            'Feb': 2,
            'Mar': 3,
            'Apr': 4,
            'May': 5,
            'Jun': 6,
            'Jul': 7,
            'Aug': 8,
            'Sep': 9, 
            'Oct': 10,
            'Nov': 11,
            'Dec': 12}[Month]

def readHorizons(file):
    file = file.readlines()
    #print(file)
    index = file.index("$$SOE\n")
    print(index)
    
    #all end at -160: aprt from vesta which is at -169
    file = file[(index + 1):-169]
    dateTime = []
    raList = []
    decList = []
    APmagList = []
    
    for line in file:
        string = str(line)
        
        monthStr = string[6:9]
        monthNum = monthToNum(monthStr)
        
        dateTime.append(string[1:6] + str(monthNum) + string[9:12] + "T" + string[13:18] + ":0.0")
        ra = np.array([string[23:25], string[26:28], string[29:34]])
        dec = np.array([string[35:38], string[39:41], string[42:46]])
        sign = string[35]
        
        APmag = string[49:55]
        
        if APmag == '  n.a.':
            APmag = 0
        else:
            APmag = float(APmag)
        
        APmagList.append(APmag)
        
        ra = np.array([float(x) for x in ra])
        dec = np.array([float(x) for x in dec])
        
        if sign == '-' and dec[0] == 0:
            dec[1] = (-1)*dec[1]
            
        elif sign == '-' and dec[0] == 0 and dec[1] == 0:
            dec[2] = (-1)*dec[2]
        
        raList.append(ra)
        decList.append(dec)
    
    dateTime = np.asarray(dateTime)
    raList = np.asarray(raList)
    decList = np.asarray(decList)
    APmagList = np.asarray(APmagList)
    
    t = Time(dateTime)
    mjd = t.mjd
    
    return mjd, raList, decList, APmagList

def findAngle(ephemeris, mjdFile, plateInput):
    """
    Parameters
    ----------
    ephemeris : The Horizons Ephemeris file to read in
    mjdFile : File with every plate number and the time it was taken in MJD

    Returns
    -------
    The angle that the asteroid in question travels with respect to the equatorial plane
    (in degrees)

    """
    
    platesFile = np.loadtxt(mjdFile).T
    plateNums = platesFile[0]
    plateNums = np.array([int(x) for x in plateNums])
    plateTimes = platesFile[1]
    
    mjd, ra_h, dec_h, APmag_h = readHorizons(ephemeris)
    positions = np.zeros((len(mjd), 2))
    for i in range(mjd.size):
        positions[i] = ra_dec_to_degrees(ra_h[i], dec_h[i])

    positions = positions.T
    #interpolating functions from the read in ephemeris file:
    interpRA = scipy.interpolate.interp1d(mjd, positions[0], fill_value="extrapolate")#2.25
    interpDEC = scipy.interpolate.interp1d(mjd, positions[1],  fill_value="extrapolate")
    
    
    initialCoords = np.array([interpRA(plateTimes), interpDEC(plateTimes)])
    initialCoords = initialCoords.T
    
    #time interval - the time after observation that will be used to figure out direction of travel
    dt = 0.0208333333
    
    ind = np.where(plateNums == plateInput)[0][0]
    
    t_next = plateTimes[ind] + dt
    
    finalCoords = np.array([interpRA(t_next), interpDEC(t_next)])
    
    print(initialCoords[ind])
    print(finalCoords)
    
    tanTheta = (finalCoords[1] - initialCoords[ind][1])/(finalCoords[0] - initialCoords[ind][0])
    
    theta = np.arctan(tanTheta)
    thetaDeg = theta*(180/np.pi)
    print(thetaDeg)
    
    return thetaDeg, initialCoords[ind]

def skipper(datafile):
    
    file = open(datafile, "r")
    file = file.readlines()
    file = file[3:-1]
    return file
    
def findAstCoords(asteroidName, asteroidPlate):
    
    filepath = r'C:\Users\mackd\Documents\5th Year\MPhys\PlatePredictorOutput\plateOutput_{0}.txt'.format(asteroidName)
    file = np.loadtxt(skipper(filepath))
    
    asteroidRA = 0
    asteroidDEC = 0
    for i in file:
        print(i[0])
        if int(i[0]) == asteroidPlate:
            asteroidRA += i[3]
            asteroidDEC += i[4]
    
    print("HMMM", asteroidRA, asteroidDEC)
    
    return np.array([asteroidRA, asteroidDEC])

def rot(image, imageRot, xy, angle): 
    org_center = (np.array(image.shape[:2][::-1])-1)/2.
    rot_center = (np.array(imageRot.shape[:2][::-1])-1)/2.
    org = xy-org_center
    a = np.deg2rad(angle)
    new = np.array([org[0]*np.cos(a) + org[1]*np.sin(a),
            -org[0]*np.sin(a) + org[1]*np.cos(a) ])
    return new+rot_center

folder = "7968Elst-Pizarro"
#folder = "4_Vesta_CONTROL_NON-ACTIVE"
#folder = "6478Gault"
#folder = "Comet_Encke_CONTROL_ACTIVE"
#folder = "C1995_O1_Hale-Bopp"
#folder = "2675_Tolkien"
#folder = "279870_2001_NL19"
#folder = "P2017-S9-PANSTARRS"
#folder = "P2016-J1-A-PANSTARRS"
#folder = "P2013R3Catalina-PANSTARRS" # go back to this one when sss up again
#folder = "118401LINEAR"
#folder = "259P_Garradd"
#folder = "233P_LaSagra"
#folder = "324PLaSagra"
#folder = "432P N4"
#folder = "2005 QN 173"
#folder = "3001632006VW139"


#user input of the plate number, plate predictor file, and e           
plateNum = int(input("What is the plate to load in? "))
horizons = open(r"C:\Users\mackd\Documents\5th Year\MPhys\horizons_{0}.txt".format(folder), 'r')
thetaDeg, asteroidCoords = findAngle(horizons, "MJD.txt", plateNum)
print("ANOTHERTEST FOR COORDS ", asteroidCoords)

#user input of the asteroid name in the desired naming format for graphs and such
name = str(input("What is the name of the Asteroid you would like printed onto graphs? "))

#importing os package to save figures to the desired directory
import os
current = os.getcwd()
print(current)
new = "final_output\{0}\{1}".format(name, plateNum)
path = os.path.join(current, new)
os.makedirs(path)

#opening up the required plate image file based on the user specified asteroid and plate number
simplecos_image = r'C:\Users\mackd\Documents\5th Year\MPhys\PlatePredictorOutput\Analysis\scans_{0}\{1}_02_intensity.fits'.format(folder, plateNum)
print('Analysing %s'%(simplecos_image))

#remember you changed the format of the wcs header reading in
#to account for astrometry website vs linux

#finding the wcs file that matches the plate in question
wcs_filename = simplecos_image.replace('fits','wcs')
#wcs_filename = r"C:\Users\mackd\Documents\5th Year\MPhys\PlatePredictorOutput\Analysis\scans_{0}\{1}_02_intensity.wcs.fits".format(folder, plateNum)

# read back the WCS
hdulist = fits.open(wcs_filename)

# Parse the WCS keywords in the primary HDU
w = wcs.WCS(hdulist[0].header)

#properly opening the plate image and extracting the data to be used
hdu = fits.open(simplecos_image)
data = hdu[0].data

#setting the data type
data = data.astype('float64')
norm1 = simple_norm(data,'sqrt', percent=99.0)
plt.imshow(data, origin='lower', norm=norm1, cmap='viridis')
plt.show()

bck_box = 32 # default SimpleCOS image size (1768, 2652) HCF gives reasonable results? Check.
bkg_interp = BkgZoomInterpolator(order = 1) 
# ... override default bicubic spline interpolation since that yields crazy values (e.g. -ve rms) near VBS 
bck = Background2D(data, box_size = bck_box, interpolator = bkg_interp, filter_size = 5)
print("Background computed with typical RMS %.2f"%(bck.background_median))

# work with a background subtracted image
data_red = data - bck.background
norm2 = simple_norm(data_red, 'sqrt', percent=99.0)
plt.imshow(data_red, origin='lower', norm=norm2, cmap='viridis')
plt.figsize=(16, 9)
plt.title("Background Subtracted Scan of Plate {0}".format(plateNum))
plt.axis('off')
plt.savefig(path + "\BackSubtractedPlateScan.png", bbox_inches='tight')
plt.show()

# threshold for segmentation image is background plus kappa x sigma
threshold = 3.0 * bck.background_rms#2.3 * bck.background_rms 17012 needed 3 sigma for some reason!
# ... choose SuperCOSMOS equivalent for isophotal detection threshold 

# matched image detection filter: Gaussian FWHM = 3 pix
from astropy.convolution import Gaussian2DKernel
from astropy.stats import gaussian_fwhm_to_sigma
from photutils import detect_sources
sigma = 2.0 * gaussian_fwhm_to_sigma
kernel = Gaussian2DKernel(sigma, x_size=3, y_size=3)
kernel.normalize() 
segm = detect_sources(data_red, threshold, npixels=5, kernel=kernel, connectivity = 8)
print("Segmentation / source detection finished ... ")

# deblend
from photutils import deblend_sources
segm_deblend = deblend_sources(data_red, segm, npixels=5, kernel=kernel)#, mode = 'linear')
print("Source deblending finished ...")

from photutils.segmentation import SourceCatalog
# photometer the sources, assuming we're background-limited:
cat = SourceCatalog(data_red, segm_deblend, error = bck.background_rms, kernel = kernel)
print("Source photometry finished:")

#defines the columns in the table
columns = ['label', 'xcentroid', 'ycentroid', #'covar_sigx2', 'covar_sigy2',
           'area', 'semimajor_sigma', 'semiminor_sigma', 'orientation', 'eccentricity',
           'segment_flux', 'segment_fluxerr']

#creates the table
peaks_tbl = cat.to_table(columns)
peaks_tbl['xcentroid'].info.format = '.3f'  # optional format
peaks_tbl['ycentroid'].info.format = '.3f'

# a few extra table columns for convenience
flux_min = np.min(peaks_tbl['segment_flux'])

#creates magnitude column
peaks_tbl['mag'] = -2.5*np.log10(peaks_tbl['segment_flux'] / flux_min)
# Irwin (1984) rule-of-thumb: centroid error equals the flux relative error multiplied by the scale size of image
peaks_tbl['xsig_estimate'] = 2.5 * peaks_tbl['segment_fluxerr'] / peaks_tbl['segment_flux']

#list of pixel coordinates
pixcrd = np.column_stack((peaks_tbl['xcentroid'], peaks_tbl['ycentroid']))

# Convert pixel coordinates to world coordinates
# The second argument is "origin" -- in this case we're declaring we
# have 1-based (SExtractor-like) coordinates.
# https://docs.astropy.org/en/stable/wcs/loading_from_fits.html
# http://star-www.dur.ac.uk/~pdraper/extractor/Guide2source_extractor.pdf
world = w.wcs_pix2world(pixcrd, 1)
peaks_tbl.add_column(world.T[0], name='ra', index=0)
peaks_tbl.add_column(world.T[1], name='dec', index=1)

#Write the csv file that shows all of the detected sources
peaks_tbl.write(simplecos_image.replace(".fits", "_segextract.csv"), format="csv", overwrite = True)

#trying to single out the asteroid in question, by checking the coordinates of the extracted source
#with the coordinates of the asteroid from the ephemeris
astLabel = []
thrsh = 0.005
for i in peaks_tbl: #was 0.002 before i changed it
    if ( (asteroidCoords[0] + thrsh) > i['ra'] > (asteroidCoords[0] - thrsh) ) and ( (asteroidCoords[1] + thrsh) > i['dec'] > (asteroidCoords[1] - thrsh) ): 
        astLabel.append(i['label'])

print(astLabel)

#astLabel should be a list of 1 number
#if the catalogue has detected a single asteroid as two sources
#then astLabel is set up to be a list to account for this
#we will take the first source (0th element) and use this for now
#code below returns the index of the specified label of the asteroid we are looking at
ast_index = np.where(peaks_tbl['label'] == astLabel[0])
#ast_index = np.where(peaks_tbl['label'] == 656)
ast_index = ast_index[0][0]

#setting up the table of the asteroid's location to be used to extract it below
ast_tbl = Table()
print(peaks_tbl['label'][ast_index])
ast_tbl['label'] = [peaks_tbl['label'][ast_index]]
ast_tbl['x'] = [peaks_tbl['xcentroid'][ast_index]]
ast_tbl['y'] = [peaks_tbl['ycentroid'][ast_index]]

#ast_tbl = stars_tbl[ast_index]
print(ast_tbl)

#finding the value of the flux to use to remove brightest stars
magList = []
for i in astLabel:
    magIndex = np.where(peaks_tbl['label'] == i)[0][0]
    magList.append(peaks_tbl['mag'][magIndex])
finalMag = sum(magList)

#for finding the mag manually
#finalMag = peaks_tbl['mag'][ast_index]

rotatedData = rotate(data, thetaDeg, reshape=True, order=5)
from astropy.table import Table

hdu = fits.PrimaryHDU(rotatedData)
#hdu.writeto('new_image.fits')

coverage_mask = (rotatedData == 0)
bck_box1 = 32 # default SimpleCOS image size (1768, 2652) HCF gives reasonable results? Check.
bkg_interp1 = BkgZoomInterpolator(order = 1) 
# ... override default bicubic spline interpolation since that yields crazy values (e.g. -ve rms) near VBS 
bckRot = Background2D(rotatedData, box_size = bck_box1, coverage_mask=coverage_mask, fill_value=0.0, interpolator = bkg_interp1, filter_size = 5)
print("Rotated background computed with typical background %.2f"%(bckRot.background_median))
rotatedData -= bckRot.background

norm3 = simple_norm(rotatedData, 'sqrt', percent=99.0)
plt.imshow(rotatedData, norm=norm3,origin='lower', cmap='viridis')
plt.show()

#carry out x and y coordinate transformation
newCoords = rot(data_red, rotatedData,np.array([ast_tbl['x'][0], ast_tbl['y'][0]]), thetaDeg)
ast_tbl_rotate = Table()
ast_tbl_rotate['label'] = [peaks_tbl['label'][ast_index]]
ast_tbl_rotate['x'] = newCoords[0]
ast_tbl_rotate['y'] = newCoords[1]

print(ast_tbl_rotate)
#WILL NEED TO LOCATE THE ASTEROID IN UNROTATED FRAME
#THEN DO COORDINATE TRANSFORMATION TO ROTATED FRAME
#THEN ROTATE THE IMAGE
#THEN EXTRACT USING ROTATED IMAGE DATA AND THE TRANSFORMED COORDINATES
#THEN WE ARE DONE AND MUST DO THE SAME FOR THE STARS.

#adding the background back in for now, to allow us to fit a model with a background parameter
rotatedData += bckRot.background
#covert data array to NDData array for the extract_stars function used here and below
nddata = NDData(data=rotatedData)

#visualising the asteroid on it's own in order to see it being flipped
#with user imput to decide if image is good
asteroid = 0
finalImSize = 0
happy = False
while not happy:
    
    imSize = int(input("What is the size of your asteroid image? "))
    temp_asteroid = extract_stars(nddata, ast_tbl_rotate, size=imSize)
    norm = simple_norm(temp_asteroid, 'log', percent=99.0)

    #plotting the asteroid that we extracted
    
    fig, (ax1) = plt.subplots(1, 1, figsize=(10, 10))
    ax1.imshow(temp_asteroid, norm=norm, origin='lower',cmap = 'magma' , label = "Asteroid of Interest")
    #ax1.scatter(peaksX, ( peaksY/ck1 ), linewidth=8, color='#ffa200', label = "Peaks (Harmonics)")
    ax1.xaxis.set_label_text('\u03C9', fontsize=14)
    ax1.yaxis.set_label_text('', fontsize=14)
    ax1.set_title("")
    #ax1.set_ylim(0, 1.12)
    #ax1.legend()
    plt.show()
    
    print("Are you happy with the image?", "\n")
    answer = str(input("Answer YES or NO: "))
    
    if answer == "YES":
        asteroid = temp_asteroid
        norm = simple_norm(asteroid, 'log', percent=99.0)
        ax1.imshow(asteroid, norm=norm, origin='lower',cmap = 'magma' , label = "Asteroid of Interest")
        finalImSize += imSize
        happy = True

#rotating the image by the desired angle

#fig, ax2 =plt.subplots(1, 1)
#ax2.imshow(rot, norm=norm, origin='lower', cmap='magma')

#extract pixel data
asteroid = asteroid.data

satisfied1 = False
while not satisfied1:
    
    print("Any horizontal slicing of the image? ", "\n")
    answer1 = str(input("Answer YES or NO: "))
    
    if answer1 == "YES":
        print("Enter the approximate indices on the left and right you want to slice: ")
        index1, index2 = map(int, input().split())
        asteroid = (asteroid.T[index1:(index2 + 1) ]).T
        satisfied1 = True
    
    if answer1 == "NO":
        satisfied1 = True

satisfied2 = False
while not satisfied2:
    fig, ax4 =plt.subplots(1, 1)
    norm = simple_norm(asteroid, 'log', percent=99.0)
    ax4.imshow(asteroid, norm=norm, origin='lower', cmap='magma')
    
    print("Any vertical slicing of the image? ", "\n")
    answer2 = str(input("Answer YES or NO: "))
    
    if answer2 == "YES":
        print("Enter the approximate indices on the top and bottom and right you want to slice: ")
        index1, index2 = map(int, input().split())
        asteroid = asteroid[index1:(index2 + 1)]
        satisfied2 = True
    
    if answer2 == "NO":
        satisfied2 = True


#plotting final image
fig, ax5 = plt.subplots(1, 1)
norm = simple_norm(asteroid, 'log', percent=99.0)
ax5.imshow(asteroid, norm=norm, origin='lower', cmap='magma', )
fig.suptitle("Asteroid {0}\n Extracted From Plate {1}".format(name, plateNum))
plt.axis('off')
fig.savefig(path + "\Asteroid_{0}.png".format(name), bbox_inches='tight')

#plotting the psf (also saving to folder)
fig, (ax) = plt.subplots(1, 1, figsize=(16, 9))
ast_x = np.linspace(0, asteroid.shape[0], asteroid.shape[0])

#summing across the x axis, and dividing by the number of columns summed
ast_psf = asteroid.sum(axis=1)
astSumNorm = np.size(asteroid, axis=1)
ast_psf = ast_psf/astSumNorm

ax.plot(ast_x, ast_psf, linewidth=8, color='#ffa200')
ax.grid(True, linestyle='dashed')
ax.xaxis.set_label_text('Pixel Width Count', fontsize=14)
ax.yaxis.set_label_text('Mean Flux Value', fontsize=14)
ax.set_title("PSF of {0} Summed Across X-Axis".format(name), fontsize=25)
fig.savefig(path + "\{0}_summed_line_profile.png".format(name))
plt.show()

#indices to slice out the noise fluctuations in the background
#to be used to give to curve_fit
i1 = int(input("Enter the index to extract noise fluctuations (left) "))
i2 = int(input("Enter the index to extract noise fluctuations (right) "))

noise_fluc = np.concatenate((ast_psf[0:i1], ast_psf[i2:-1]))
noise_median = np.median(noise_fluc)
deviations = np.abs(noise_fluc - noise_median)
dev_median = np.median(deviations)
fit_sigma = dev_median*1.48

print("The sigma to give to the curve fit algorithm is: ", fit_sigma)

#subtracting the median background from the rotated image
rotatedData -= bckRot.background
nddata = NDData(data=rotatedData)

#creating x, y, and flux, columns from the peaks
x =  np.array([float(i) for i in peaks_tbl['xcentroid']])
y =  np.array([float(j) for j in peaks_tbl['ycentroid']])
mag = peaks_tbl['mag']

#print("THIS IS THE REAL TEST", "\n", temp_tbl)
adj = float(input("Any adjustments to the brightest magnitude? "))
print("The asteroid's magnitude is ", finalMag)
question = float(input("What is the dimmest magnitude to use? "))


#now we will deal with choosing stars that are "good" i.e. not near the edge
#size of the square being cut out
size = int(input("What is the box size of the PSF cutouts? "))
hsize = (size - 1) / 2

#creating a mask that cuts out a 25 x 25 pixel square around each peak (star) 
#ALSO CONTROLS THE BRIGHTEST AND DIMMEST STAR MAGNITUDES TO CHOOSE
mask = ( (x > hsize) & (x < (data.shape[1] - 1 - hsize)) & (y > hsize) & (y < (data.shape[0] - 1 - hsize)) & (mag > (finalMag - adj) ) & (mag < question))
#creating a table of star positions that are good (i.e.not cut off)
stars_tbl = Table()
stars_tbl['label'] = peaks_tbl['label'][mask]
stars_tbl['x'] = np.array([int(i) for i in x[mask]])
stars_tbl['y'] = np.array([int(j) for j in y[mask]])

print(stars_tbl)

#finds the indices of good stars to use for the PSF
goodStars = []
starLength = int(len(stars_tbl))
for i in range(0, (starLength) ):
    useful = True
    for j in range(0, starLength):
        if j != i:
            xdiff = stars_tbl[j]['x'] - stars_tbl[i]['x'] 
            ydiff = stars_tbl[j]['y'] - stars_tbl[i]['y']
            diffMod = np.sqrt(xdiff**2 + ydiff**2)
        
            if diffMod <= np.sqrt(2)*(size/2):
                useful = False
        
            if not useful:
                break

    if useful:
        goodStars.append(i)
    
    if not useful:
        continue
    
#convert the good stars indices into a numpy array
goodStars = np.asarray(goodStars)
goodStars = np.array([int(x) for x in goodStars])
goodStars = np.unique(goodStars)

print(goodStars.size)

#apply the mask that gets rid of stars with nearby sources!!!
stars_tbl = stars_tbl[goodStars]

print(stars_tbl)

rotatedData = rotatedData.astype('float64')
nddata = NDData(data=rotatedData)

#find the transformed coordinates of the good stars in stars_tbl
newCoords = []
for i in stars_tbl:
    newXY = rot(data_red, rotatedData, np.array([i['x'], i['y']]), thetaDeg)
    newCoords.append(newXY)

newCoords = np.asarray(newCoords).T
stars_tbl_rot = Table()
stars_tbl_rot['label'] = stars_tbl['label']
stars_tbl_rot['x'] = np.array([float(x) for x in newCoords[0]])
stars_tbl_rot['y'] = np.array([float(y) for y in newCoords[1]])

print(stars_tbl_rot)

#TEMP LINE TO GIVE THE EXTRACT STARS AN IMAGE AND LIST OF STUFF THAT ISN'T ROTATED!!!
nddata_unrot = NDData(data = data_red)

#extract our stars in the table
stars = extract_stars(nddata, stars_tbl_rot, size=size)

#look at first 25 stars
nrows = 5
ncols = 5

#plotting the cutouts we have created, of the first 25 stars
fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 20), squeeze=True)
ax = ax.ravel()

for i in range(nrows * ncols):
    norm = simple_norm(stars[i], 'log', percent=99.0)
    ax[i].imshow(stars[i], norm=norm, origin='lower', cmap='viridis')

fig.suptitle("Sample of Background Sources Selected for PSF\n (Plate {0})\n".format(plateNum), fontsize=55)
plt.savefig(path + "\Plate_{0}_1st_25_stars_stack".format(plateNum))
plt.show()

#building the point spread function (PSF) for each star by inputting each of the stars we extracted earlier.
#this is a class that we are using to create a function, then input our table of stars
epsf_builder = EPSFBuilder(oversampling=2, maxiters=10, smoothing_kernel='quadratic', progress_bar=False)
epsf, fitted_stars = epsf_builder(stars)
print(epsf.data.shape)

#visualise and plot the PSF we have created
norm = simple_norm(epsf.data, 'log', percent=99.0)

plt.imshow(epsf.data, norm=norm, origin='lower', cmap='inferno')
#col = plt.colorbar(ticks = [np.amin(epsf.data), np.amax(epsf.data)])
#col.ax.tick_params(labelsize=8)
plt.title("PSF of Stacked Background Stars (Plate No.{0})\n Composed of {1} Stars".format(plateNum, len(stars_tbl_rot)))
plt.figsize=(16, 9)
plt.savefig(path + "\Plate_{0}_stacked_PSF".format(plateNum))
plt.show()

j1 = int(input("What index to slice the PSF on the left? "))
j2 = int(input("What index to slice the PSF on the right? "))

plt.imshow((epsf.data.T[j1:(j2 + 1)]).T, norm=norm, origin='lower', cmap='inferno')
plt.colorbar()
plt.figsize=(16, 9)
plt.show()

#plotting the background star psf
fig, (ax) = plt.subplots(1, 1, figsize=(16, 9))
#summing across the x axis 
psf = (epsf.data.T[j1:(j2 + 1)]).T.sum(axis=1)

psf = psf/np.max(psf)
psf_x = np.linspace(-psf.size/2, psf.size/2, psf.size)
ax.plot(psf_x, psf, linewidth=8, color='#5c00a3')
ax.grid(True, linestyle='dashed')
ax.xaxis.set_label_text('Pixel Width of Fitted PSF', fontsize=14)
ax.yaxis.set_label_text('Normalised Flux Value', fontsize=14)
ax.set_title("Normalised PSF of Background Stars Summed Across X-Axis\n (Plate No.{0})".format(plateNum), fontsize=25)
fig.savefig(path + "\Plate_{0}_summed_LSF_normalized".format(plateNum))
plt.show()

import scipy.optimize
from scipy import signal

def psf_model(x, h, x0, b, x_model, y_model):
    """
    
    Parameters
    ----------
    x : x-data to interpolate 
    y : y-data to interpolate 
    
    Returns
    -------
    Function to insert any x value and give the approx y value

    """
    #resample the oversampld PSF data at the original pixel size
    y_model = signal.resample(y_model, size)
    xnew = np.linspace(-size/2, size/2, size)
    #xnew = x_model
    
    xbound1 = x0 - xnew.size/2
    xbound2 = x0 + xnew.size/2
    
    interpolater = scipy.interpolate.CubicSpline(xnew, y_model)                  
    interpFunc = x*0
    interpFunc[(x < xbound1)] = h*y_model[0] + b
    interpFunc[(x >= xbound1) & (x <= xbound2)] = h*interpolater(x - x0)[(x >= xbound1) & (x <= xbound2)] + b
    interpFunc[(x > xbound2)] = h*y_model[-1] + b
    
    return interpFunc

def psf_deriv(x, x0, x_model, y_model):
    """
    derivative of the interpolator function
    """
    #resample the oversampld PSF data at the original pixel size
    y_model = signal.resample(y_model, size)
    xnew = np.linspace(-size/2, size/2, size)
    #xnew = x_model
    
    xbound1 = x0 - xnew.size/2
    xbound2 = x0 + xnew.size/2
    
    #the interpoator
    interp = scipy.interpolate.CubicSpline(xnew, y_model)             
    
    interpFunc = x*0
    interpFunc[(x < xbound1)] = y_model[0]
    interpFunc[(x >= xbound1) & (x <= xbound2)] = interp(x - x0)[(x >= xbound1) & (x <= xbound2)]
    interpFunc[(x > xbound2)] = y_model[-1]
    
    #the derivative of the interpolator
    interpDeriv = scipy.interpolate.CubicSpline(xnew, y_model).derivative()                  
    
    interpFuncDeriv = x*0
    interpFuncDeriv[(x < xbound1)] = 0
    interpFuncDeriv[(x >= xbound1) & (x <= xbound2)] = interpDeriv(x - x0)[(x >= xbound1) & (x <= xbound2)]
    interpFuncDeriv[(x > xbound2)] = 0
    
    return interpFunc, interpFuncDeriv

def errors(h, x0, b, f, fDeriv, cov):
    """
    Function to propagate the errors on Y, based on the errors of the best fit params.
    
    """
    #calculating the error propagation from the components of the covariance matrix
    dy_sq = (  (f**2)*cov[0][0] + h*f*fDeriv*cov[0][1] + f*cov[0][2]
             + h*f*fDeriv*cov[1][0] + (h**2)*(fDeriv**2)*cov[1][1] 
             + h*fDeriv*cov[1][2] + f*cov[2][0] + h*fDeriv*cov[2][1] + cov[2][2]  )
    
    #square rooting
    dy = np.sqrt(dy_sq)
    
    return dy

def chi_squared(residuals, rms):

    normalised = (residuals/rms)**2
    return np.sum(normalised)
    
#setting up optimisation initial guesses
#the background guess is obtianed by multiplying the number of columns which the asteroid LSF summed over
bguess = np.min(ast_psf)
x0guess = np.argmax(ast_psf)
hguess = (np.max(ast_psf) - bguess)

#printing the initial guesses
print("h guess is ", hguess)
print("X0 GUESS IS ", x0guess)
print("b guess is ", bguess)

#creating an array of the sigma value we found earlier
fit_sigma_arr = fit_sigma*np.ones(ast_x.size)
print(fit_sigma_arr)
#using scipy to fit the right function to the asteroid data
fitParams = scipy.optimize.curve_fit( (lambda x, h, x0, b: psf_model(x, h, x0, b, psf_x, psf)), ast_x , ast_psf, [hguess, x0guess, bguess], sigma = fit_sigma_arr, absolute_sigma=True, method='lm')

#printing the calculated best parameters
besth = fitParams[0][0] #amplitude of the PSF
bestx0 = fitParams[0][1] #x coordinate of peak
bestb = fitParams[0][2] #background

print("The best Amplitude Value is :", besth)
print("The best x0 Value is :", bestx0)
print("The best Background Value is :", bestb)

#extracting the covariance matrix of the fit
cov = fitParams[1]
print(cov)

#extracts errors on the best fit params
params_errs = np.sqrt(np.diag(cov))

#evaluating each "predicted" data point with the best parameters
fit = psf_model(ast_x, fitParams[0][0], fitParams[0][1], fitParams[0][2], psf_x, psf)

#smooth model linspace fit evaluated for plotting
smoovX = np.linspace(ast_x[0], ast_x[-1], 1000)
smoothFit = psf_model(smoovX, fitParams[0][0], fitParams[0][1], fitParams[0][2], psf_x, psf)

#residual of model and data
residual = ast_psf - fit

#degrees of freedom are number of data points minus number of fitted parameters
dof = ast_x.size - 3

print("DEGREES OF FREEDOM", dof)

#NOW WE WILL DO ERROR PROPAGATION ON MY POINTS

#function and derivative of function at the x coordinates of the asteroid LSF
f, fDeriv = psf_deriv(ast_x, bestx0, psf_x, psf)

#using the error propagation function
dy = errors(besth, bestx0, bestb, f, fDeriv, cov)

print(dy)

chiSq = chi_squared(residual, fit_sigma)
print("CHI2 is ", chiSq)

reducedChiSq = chiSq/dof
print("REDUCED CHI2 ", reducedChiSq)

from scipy.stats import chi2
pvalue = chi2.sf(chiSq, dof)
print("THE P-VALUE IS ", pvalue)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(46, 14))

ax1.plot(smoovX, smoothFit, linewidth=8, color ='#490675', label="Background Star PSF Model")
ax1.plot(ast_x, ast_psf, linewidth = 4, color="#ff6a00", label="Asteroid Line Profile" )
ax1.grid(True, linestyle='dashed')
ax1.xaxis.set_label_text('Distance From Centroid (Pixel Number)', fontsize=30)
ax1.yaxis.set_label_text('Summed Flux Value', fontsize=30)
ax1.fill_between(ast_x, (fit - 2*dy), (fit + 2*dy), color='b', alpha=.1)
ax1.set_title("Asteroid '{0}' PSF With Best Fit Model (Plate No.{1})".format(name, plateNum), fontsize=36)
#ax1.suptitle("Plate Number {0}".format(plateNum), fontsize=27)
ax1.tick_params(axis='both', which='major', labelsize=20)
ax1.legend(fontsize=28)


ax2.plot(ast_x, residual, linewidth = 8, color="#fc03ca")
ax2.plot(ast_x, np.zeros(len(ast_x)), linestyle = "dashed")
ax2.grid(True, linestyle='dashed')
ax2.xaxis.set_label_text('Distance From Centroid (Pixel Number)', fontsize=30)
ax2.yaxis.set_label_text('Flux Residuals (Observed - Model)', fontsize=30)
ax2.set_title("Residuals of {0} Profile and Model".format(name), fontsize=36)
ax2.tick_params(axis='both', which='major', labelsize=20)
#plt.xlim(0, 50)
#plt.ylim(-2000, 16000)
plt.show()

#saving the fit and residual graph to desired directory
fig.savefig(path + "\FitAndResidual.png")

#writes the best fit parameters and their calculated uncertainties to file
best_params_file = open(path + '\model_params.txt','w')
best_params_file.write('%lf %lf %lf %lf %lf %lf\n'%(besth, bestx0, bestb, params_errs[0], params_errs[1], params_errs[2]))
best_params_file.close()

#writes the asteroid line profile to file
ast_file = open(path + '\{0}_profile.txt'.format(name),'w')
for i in range(ast_psf.size):
    ast_file.write('%lf %lf\n'%(ast_x[i], ast_psf[i]))
ast_file.close()

#writes the normalised stacked LSF of the background star profile to file
star_lsf = open(path + '\star_psf.txt','w')
for i in range(psf.size):
    star_lsf.write('%lf %lf\n'%(psf_x[i], psf[i]))
star_lsf.close()

#writes the best fit model with the best fit parameters and corresponding
#model uncertainty to file (allowing plotting of shaded confidence interval region)
star_model = open(path + '\star_psf_model.txt','w')
for i in range(ast_x.size):
    star_model.write('%lf %lf %lf\n'%(ast_x[i], fit[i], dy[i]))
star_model.close()

smooth_model = open(path + '\star_psf_model_SMOOTH.txt','w')
for i in range(smoovX.size):
    smooth_model.write('%lf %lf\n'%(smoovX[i], smoothFit[i]))
smooth_model.close()

#writes the residuals (asteroid profile - fit) to file
residual_file = open(path + '\{0}_residuals.txt'.format(name),'w')
for i in range(residual.size):
    residual_file.write('%lf %lf\n'%(ast_x[i], residual[i]))
residual_file.close() 

#writes the stats stuff, degrees of freedom, chi squared, reduced chi2, and the p-value
stats_file = open(path + '\statistics.txt','w')
stats_file.write('%lf %lf %lf %lf\n'%(dof, chiSq, reducedChiSq, pvalue))
stats_file.close() 




