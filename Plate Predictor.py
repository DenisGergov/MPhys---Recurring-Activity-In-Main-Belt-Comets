"""
Script for searching through the UK Schmidt Telescope ROE Plate Archive for a specific asteroid.
Functions include:
- Converting stellar coordinates of ROE Plates from the old B1950 epoch to the J2000 epoch.
- Converting the RA and DEC into decimal degrees.
- Converting the date (UTC) and time (Local Sidereal Time) of each plate exposure to MJD
- Converting the exposure times of the ROE plates to Minuutes

Then, the program reads in an ephemeris (or multiple) of an asteroid or any other stellar object
This ephemeris is generated and downloaded from the NASA Horizons system https://ssd.jpl.nasa.gov/horizons/app.html

The program then uses interpolation methods to determine the position of the asteroid at each plate exposure
And then checks whether that position lies within that plate.

Result: a .txt file with every playe that the asteroid is located on, with X and Y coordinates in mm from
the bottom left edge of the plate.

Some of the algorithms were provided by 
Duffet'Smith's "Practical Astronomy with your Calculator or Spreadsheet"

MPhys Project
Denis Gergov S1839787
2022/23

"""
import numpy as np
import math
import datetime as dt
from astropy.time import Time
import scipy.interpolate
from scipy.interpolate import interp1d

def convert(ra, dec):
    
    #converting to rectangular coordinates
    pos = ra_dec_to_xyz(ra, dec)
    
    #defining the coordinate transformation matrix and applying it to the posititon:
    epoch_matrix = np.array([[0.999926, -0.011179, -0.004859],
                             [0.011179, 0.999938, -0.000027],
                             [0.004859, -0.000027, 0.999988]])
    
    pos_new = np.matmul(epoch_matrix, pos)
    
    return xyz_to_ra_dec(pos_new)

def xyz_to_ra_dec(pos):
    #finally, we need to convert our transformed rectangular coordinate back into the standard form of RA and DEC
    r = math.atan(pos[1]/pos[0])
    
    dec_new = math.asin(pos[2])
    ra_new = 0
    
    if pos[0] > 0 and pos[1] > 0:
        ra_new += r
        
    elif pos[0] > 0 and pos[1] < 0:
        ra_new += r + 2*math.pi
    
    elif pos[0] < 0:
        ra_new += r + math.pi
      
    
    #ra in hours, dec in degrees
    ra_new = ((180/math.pi)*ra_new)/15
    dec_new = (180/math.pi)*dec_new

    #this deals with obtaining the minutes
    frac1, whole1 = math.modf(ra_new)
    frac2, whole2 = math.modf(dec_new)
    
    minutes1 = 60 * abs(frac1)
    minutes2 = 60 * abs(frac2)
    
    #this deals with obtaining the seconds
    frac12, whole12 = math.modf(minutes1)
    frac22, whole22 = math.modf(minutes2) 
    
    seconds1 = 60 * frac12
    seconds2 = 60 * frac22
    
    #generating array of final form for ra and dec
    ra = np.array([whole1, whole12, seconds1])
    dec = np.array([whole2, whole22, seconds2])
    
    return ra, dec
    
def hmsm_to_days(hour=0,mins=0,sec=0,micro=0):
    """
    Convert hours, minutes, seconds, and microseconds to fractional days.
    
    Parameters
    ----------
    hour : int, optional
        Hour number. Defaults to 0.
    
    mins : int, optional
        Minute number. Defaults to 0.
    
    sec : int, optional
        Second number. Defaults to 0.
    
    micro : int, optional
        Microsecond number. Defaults to 0.
        
    Returns
    -------
    days : float
        Fractional days.
        
    Examples
    --------
    >>> hmsm_to_days(hour=6)
    0.25
    
    """
    hours = sec + (micro / 1.e6)
    
    hours = mins + (hours / 60.)
    
    hours = hour + (hours / 60.)
    
    return hours

def days_to_hmsm(hr):
    """
    Convert fractional days to hours, minutes, seconds, and microseconds.
    Precision beyond microseconds is rounded to the nearest microsecond.
    
    Parameters
    ----------
    days : float
        A fractional number of days. Must be less than 1.
        
    Returns
    -------
    hour : int
        Hour number.
    
    min : int
        Minute number.
    
    sec : int
        Second number.
    
    micro : int
        Microsecond number.
        
    Raises
    ------
    ValueError
        If `days` is >= 1.
        
    Examples
    --------
    >>> days_to_hmsm(0.1)
    (2, 24, 0, 0)
    
    """
    hours = hr
    hours, hour = math.modf(hours)
    
    mins = hours * 60.
    mins, min = math.modf(mins)
    
    secs = mins * 60.
    secs, sec = math.modf(secs)
    
    micro = round(secs * 1.e6)
    
    return int(hour), int(min), int(sec), int(micro)

def lst_to_gst(hours, minutes, long):
    
    #converting LST to decimal hours
    decimal = hmsm_to_days(hours, minutes)
    
    #dividing longitude by 15 to convert that into decimal hours
    hours = long/15
    
    #bringing the range to 0 - 24
    result = decimal - hours
    if result < 0:
        result += 24
    elif result >= 24:
        result -+ 24
    hour, mins, sec, micro = days_to_hmsm(result)
    return np.array([hour, mins])

def gst_to_utc(date, hours, minutes, seconds=0):
    """
    Converts times from Greenwich Sidereal Time (GST) to Universal Time (UT)
                                                                        
    Parameters
    ----------
    date : Numpy array of format yymmdd
    hours : Scalar value of the hours in the time (GST)
    minutes : Scalar value of the minutes of the time
    seconds : Scalar value of the seconds The default is 0.

    Returns
    -------
    UT : Hours, Minutes and Seconds in Universal Time

    """
    
    jdFormat = str(date[0]) + "-" + str(date[1])+ "-" + str(date[2]) + "T" + "00:00:00.0"
    t = Time(jdFormat, format='isot', scale='utc')
    jd = t.jd
    s = jd - 2451545.0
    T = s/36525.0
    T0 = 6.697374558 + (2400.051336*T) + (0.000025862*(T**2))
    
    if T0 >= 24 or T0 < 0:
       T0 = T0%24

    gst = hmsm_to_days(hours, minutes, seconds)
    gstT0 = gst - T0

    if gstT0 >= 24 or gstT0 < 0:
       gstT0 = gstT0%24

    UT = 0.9972695663*gstT0
    UT = days_to_hmsm(UT)
    
    return UT

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

def ra_dec_to_degrees(ra, dec):
    #must convert ra and dec into decimal coordinates
    raN = (ra[0] + ra[1]/60 + ra[2]/3600)*15 
    decN = abs(dec[0]) + abs(dec[1])/60 + abs(dec[2])/3600 
        
    if dec[0] < 0 or dec[1] < 0 or dec[2] < 0:
        decN = (-1)*decN
    
    return raN, decN

def ra_dec_to_xyz(ra, dec):
    
    #converting the RA from hours, minutes, seconds and the DEC from degrees, minutes, seconds
    #into decimal degrees
    ra, dec = ra_dec_to_degrees(ra, dec)

    #must convert to radians
    ra_rad = (math.pi/180)*ra
    dec_rad = (math.pi/180)*dec
       
    #then, we must convert to rectangular coordinates in order to apply a coordinate transformation
    x = math.cos(ra_rad)*math.cos(dec_rad)
    y = math.sin(ra_rad)*math.cos(dec_rad)
    z = math.sin(dec_rad)
    pos = np.array([x, y, z])
    
    return pos

def deg_to_xyz(ra, dec):
    
    #must convert to radians
    ra_rad = (math.pi/180)*ra
    dec_rad = (math.pi/180)*dec
       
    #then, we must convert to rectangular coordinates in order to apply a coordinate transformation
    x = math.cos(ra_rad)*math.cos(dec_rad)
    y = math.sin(ra_rad)*math.cos(dec_rad)
    z = math.sin(dec_rad)
    pos = np.array([x, y, z])
    
    return pos

def readHorizons(file):
    
    file = file.readlines()
    #print(file)
    index = file.index("$$SOE\n")
    print(index)
    
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

def LocalPlane(tangentA, tangentD, ra, dec):
    
    tangentA = (math.pi/180)*tangentA
    tangentD = (math.pi/180)*tangentD
    
    ra = (math.pi/180)*ra
    dec = (math.pi/180)*dec
    
    cos_theta = np.sin(dec)*np.sin(tangentD) + np.cos(dec)*np.cos(tangentD)*np.cos(ra - tangentA)
    cos_theta_deg = (180/math.pi)*np.arccos(cos_theta)
    
    zeta = (np.cos(dec)*np.sin(ra - tangentA))/(cos_theta)
    eta = (np.sin(dec)*np.cos(tangentD) - np.cos(dec)*np.sin(tangentD)*np.cos(ra - tangentA))/(cos_theta)
    
    zeta = (180/math.pi)*zeta
    eta = (180/math.pi)*eta
    
    return zeta, eta, cos_theta_deg

def emul_filt_expTime():
    expTimeDict = {'IIaO  UG1   ': 180,
                   'IIIaJ UG1   ': 180,
                   '4415  UG1   ': 180,
                   'IIaO  GG 385': 60,
                   'IIIaJ GG 395': 60,
                   'IIaD  GG 495': 60,
                   'IIIaF OG 590': 60,
                   'IIIaF RG 630': 90,
                   '4415  OG 590': 60, 
                   '4415  OG590 ': 60,
                   '4415  HA659 ': 180,
                   'IVN   RG 715': 90,
                   'IIIaJ WG 3051': 45,
                   'IIIaF WG 3051': 20,
                   'IIIaF WG 3053': 20,
                   'IIIaF GG 4553': 35,
                   'IIIaJ WG 3053': 45,
                   'IIIaJ GG 3953': 60,
                   'IIIaJ GG 4553': 90}
    return expTimeDict

def emul_filt_LimMag():
    LimMagDict = {'IIaO  UG1   ': 21.0,
                  'IIIaJ UG1   ': 21.0,
                  '4415  UG1   ': 21.0,
                  'IIaO  GG 385': 21.0,
                  'IIIaJ GG 395': 22.5,
                  'IIaD  GG 495': 21.0,
                  'IIIaF OG 590': 21.5,
                  'IIIaF RG 630': 21.5,
                  '4415  OG 590': 22.5, 
                  '4415  OG590 ': 22.5,
                  '4415  HA659 ': 21.5,
                  'IVN   RG 715': 19.5,
                  'IIIaJ WG 3051': 20.0,
                  'IIIaF WG 3051': 18.5,
                  'IIIaF WG 3053': 17.5,
                  'IIIaF GG 4553': 18.5,
                  'IIIaJ WG 3053': 18.0,
                  'IIIaJ GG 3953': 18.5,
                  'IIIaJ GG 4553': 19.5}
    return LimMagDict


def mean_square(interp, testx, testy, ret):
    ret += np.mean((interp(testx) - testy)**2)
    return ret

def solve_task(x, y, xnew):
    """
    This function needs select the best interpolation method for provided data  
    and return the numpy array of interpolated values at the locations specified in test.txt

    """
    nsplit = 3
    N = len(x)
    pos = np.arange(len(x))
    ret1= 0
    ret2 = 0
    sParams = np.linspace(0, 5, 100)
    ret3 = np.zeros((len(sParams)))
    
    for i in range(nsplit):
        testsubset = pos%nsplit == i 
        fitsubset = ~testsubset
        curx = x[fitsubset]
        cury = y[fitsubset]
        testx = x[testsubset]
        testy = y[testsubset]
        
        linear = scipy.interpolate.interp1d(curx, cury, fill_value="extrapolate")
        cubic = scipy.interpolate.CubicSpline(curx, cury)
        
        ret1 = mean_square(linear, testx, testy, ret1)
        ret2 = mean_square(cubic, testx, testy, ret2)
        #ret3 = mean_square(smoothing, testx, testy, ret3)
        
        for j in range(len(sParams)):
            smoothing = scipy.interpolate.UnivariateSpline(curx, cury, s=sParams[j])
            ret3[j] = mean_square(smoothing, testx, testy, ret3[j])
        
    ret1 = ret1/nsplit
    ret2 = ret2/nsplit
    ret3 = ret3/nsplit
    
    dict = {
      "Linear": ret1,
      "Cubic": ret2,
      "Smoothing": ret3,
    }
    
    ms = np.concatenate((np.array([ret1, ret2]), ret3))
    ind = np.argsort(ms)
    ms = ms[ind]
    
    if ms[0] == dict["Linear"]:
        #return scipy.interpolate.interp1d(x, y, fill_value="extrapolate")(xnew)
        return "LINEAR IS BEST"
    
    elif ms[0] == dict["Cubic"]:
        #return scipy.interpolate.CubicSpline(x, y)(xnew)
        return "CUBIC IS BEST"
    
    elif ms[0] in dict["Smoothing"]:
        index = np.where(dict["Smoothing"] == ms[0])
        index = int(index[0])
        print("The best parameter is ", sParams[index])
        #return scipy.interpolate.UnivariateSpline(x, y, s=sParams[index])(xnew)
        return sParams[index]

def RA_DEC_degToMinsHrs(ra, dec):
    
    ra/=15
    
    fracRA, wholeRA = math.modf(ra)
    fracDEC, wholeDEC = math.modf(dec)
    
    fracMinsRA = 60 * abs(fracRA)
    fracMinsDEC = 60 * abs(fracDEC)
    
    #this deals with obtaining the seconds
    fracMinsRA, wholeMinsRA = math.modf(fracMinsRA)
    fracMinsDec, wholeMinsDEC = math.modf(fracMinsDEC) 
    
    secsRA = 60 * fracMinsRA
    secsDEC = 60 * fracMinsDec
    
    #generating array of final form for ra and dec
    raN = np.array([wholeRA, wholeMinsRA, secsRA])
    decN = np.array([wholeDEC, wholeMinsDEC, secsDEC])
    
    return raN, decN
    
ra = np.array([0, 0, 0])
print(ra)
dec = np.array([0, 0, 0])
print(dec)

ra_n, dec_n = convert(ra, dec)

print(ra_n)
print(dec_n)

"""
#TESTING AREA
#THIS IS WHERE I INITIALLY TEST FUNCTIONS FOR A SPECIFIC CASE BEFORE APPLYING THEM TO THE FILE    

ra = np.array([0, 40, 0])
print(ra)
dec = np.array([-90, 0, 0])
print(dec)

ra_n, dec_n = convert(ra, dec)
print("The RA in J2000 epoch is : ", ra_n[0], ";", ra_n[1], ";", round(ra_n[2], 3))
print("The DEC in J2000 epoch is : ", dec_n[0], ";", dec_n[1], ";", round(dec_n[2], 3))

times = np.array(['1999-1-1T5:00:00.123456789', '2010-01-01T23:44:0'])
t = Time(times, format='isot', scale='utc')
print(t.mjd)

y = lst_to_gst(0, 24.08717, -64)
print("LST TO GST TEST " + str(y))

date = np.array([1980, 4, 22])
hours = 4
minutes = 40
seconds = 5.23

b = gst_to_utc(date, hours, minutes, seconds)
print("THE UT IS " + str(b))

"""

def PlatePredict(ephemeris):
    
    #opens UKST data file of the plate archive
    df = open('ukst.txt')
    df = df.readlines()

    #creates lists that will be filled with the transformed coordinates (lists easier to append)
    #(later converted to np arrays)
    ra_list = []
    dec_list = []
    longitude = 149.07

    platNum = []
    dateList = []
    timeList = []
    emulFiltList = []

    expTimeList = []

    for line in df: 
        
        #turns every line into a string from which the characters can be picked out to slice out needed data
        ints = str(line)
        
        #deals with extracting the plate number, date of exposure, and LST (sidereal time) of exposure
        platNum.append(int(ints[2:7]))
        date = np.array([ints[30:32], ints[32:34], ints[34:36]])
        date = np.array([int(x) for x in date])
        time = np.array([ints[36:38], ints[38:40]])
        
        if len(ints) >= 63 and (ints[61] == '1' or ints[61] == '2' or ints[61] == '3' or ints[61] == '4'):
            emulFiltList.append(ints[40:52] + ints[61])
        
        else:
            emulFiltList.append(ints[40:52])
        
        #time to fill in the gaps that the really, REALLY inconvenient formatting of the plates archive features
        #wtf is time that is listed as "24"??
        if date[0] < 4:
            date[0] = int(str(200) + str(date[0]))
        
        elif date[0] > 4:
            date[0] = int(str(19) + str(date[0]))
        
        if str(time[0]) == "  ":
            time[0] = "0"
        
        if str(time[0]) == "24":
            time[0] = "0"
        
        #fills empty spaces in data with zeroes
        
        #converting time array elements to integers
        time = np.array([int(x) for x in time])
        
        #extracts the exposure time for each frame
        expTime = float(ints[52] + ints[53] + ints[54] + ints[55])
        
        #converts the exposure time from tenths of a minute to minutes
        expTime = expTime/10
        
        #we are interested in the time mid exposure, so this must be adjusted in the time array
        #this is done by dividing the exposure time by two and adding into the start of exposure
        midExp = expTime/2
        time[1] = (time[1]) + midExp
        
        if time[1] >= 60:
            remainder = (time[1])%60
            time[0] += ((time[1]) - midExp)/60
            time[1] = remainder
        else:
            time[1] = time[1]
        
        #saving the exposure time for later
        expTimeList.append(expTime)
        
        #converting the LST to GST using our function
        time = lst_to_gst(time[0], time[1], longitude)
        
        #converting the GST to UT to be used to be converted to MJD
        time = gst_to_utc(date, time[0], time[1])
        
        #appending to the created lists    
        dateList.append(date)
        timeList.append(time)
        
        #slices out the relevant columns from the data for the RA and DEC
        ra = np.array([ints[20:22], ints[22:24], ints[24]])
        dec = np.array([ints[25:28], ints[28:30], '0'])
        sign = ints[25]
        
        #converting RA and Dec into integers
        ra = np.array([int(x) for x in ra])
        dec = np.array([int(x) for x in dec])
        
        if sign == '-' and dec[0] == 0:
            dec[1] = (-1)*dec[1]
            
        #scaling factor that deals with the last element in ra representing tenths of a minute (converting to seconds)
        ra[2] = (ra[2])*(60/10)
        
        #applies conversion algorithm from above to transform from B1950 to J2000 epoch
        ra, dec = convert(ra, dec)
        
        #appends to the lists above
        ra_list.append(ra)
        dec_list.append(dec)


    #converts RA and DEC from a list of arrays to an array of arrays
    platNum = np.asarray(platNum)
    dateList = np.asarray(dateList)
    timeList = np.asarray(timeList)
    ra_list = np.asarray(ra_list)
    dec_list = np.asarray(dec_list)
    expTimeList = np.asarray(expTimeList)

    #writes a file of the extracted relevant data from the plate catalogue, ready to be read back into Astropy
    file = open("transformed.txt", "w")
    for i in range(len(ra_list)):
        file.write(str(platNum[i]) + " " + str(dateList[i][0]) + "-" + str(dateList[i][1])+ "-" + str(dateList[i][2]) + "T" + str(int(timeList[i][0])) + ":" + str(int(timeList[i][1])) + ":0" +  " " + str(ra_list[i][0]) + ";" + str(ra_list[i][1]) + ";" + str(round(ra_list[i][2], 3)) + " " + str(dec_list[i][0]) + ";" + str(dec_list[i][1]) + ";" + str(round(dec_list[i][2], 3)) + "  " + "\n")

    #creating an array to be written to the final text file which will have time in MJD
    plates = np.zeros((len(ra_list), 2))

    #index to be used in converting stuff to mjd, don't ask why it didn't work any other way
    count = 0

    #opening the newly created file from above to perform the MJD conversion
    file = open("transformed.txt", "r")
    for x in file.readlines():
        column = x.split(' ')
        plates[count][0] = column[0]
        t = Time(column[1])
        plates[count][1] = t.mjd
        count += 1
    file.close()

    #writing the MJD conversion from the Plates array
    outputMJD = open("MJD.txt", "w")
    for i in range(len(ra_list)):
        outputMJD.write(str(int(plates[i][0])) + " " + str(plates[i][1]) + "\n")
    outputMJD.close()
    
    #reads in a txt file of an ephemeris of a specific body, generated for the range of dates of the plate archive
    #this uses our horizons data file read function
    horizons = open(ephemeris, 'r')
    mjd, ra_h, dec_h, APmag_h = readHorizons(horizons)
    positions = np.zeros((len(mjd), 2))

    for i in range(mjd.size):
        positions[i] = ra_dec_to_degrees(ra_h[i], dec_h[i])

    positions = positions.T
    #creates a scipy interpolation of the read in time, three times, for x, y and z coordinates seperately
    #this is done using the CubicSpline method
    """
    intRA = scipy.interpolate.CubicSpline(mjd, positions[0])
    intDEC = scipy.interpolate.CubicSpline(mjd, positions[1])
    """

    """
    intRA = scipy.interpolate.UnivariateSpline(mjd, positions[0], s=2) #2.25
    intDEC = scipy.interpolate.UnivariateSpline(mjd, positions[1], s=2)
    """
    #interpolating functions from the read in ephemeris file:
    intRA = scipy.interpolate.interp1d(mjd, positions[0], fill_value="extrapolate")#2.25
    intDEC = scipy.interpolate.interp1d(mjd, positions[1],  fill_value="extrapolate")

    intAPmag = scipy.interpolate.CubicSpline(mjd, APmag_h)

    #array of the times (in MJD) of all the plates in the plate archive
    plates = plates.T
    plateNames = plates[0]
    plateTimes = plates[1]

    
    outputEMULS = open("EMULS.txt", "w")
    for i in range(len(plateNames)):
        outputEMULS.write(str(plateNames[i]) + " " + str(emulFiltList[i]) + '**' + "\n")
    outputMJD.close()

    #performs interpolation for each time in the plates archive, to predict the position of the object at each time
    positionPredict = np.array([intRA(plateTimes), intDEC(plateTimes)])
    positionPredict = positionPredict.T

    #this performs interpolation of the apparant magnitudes - this will be used to compare with the 
    #plate's limiting magnitude to determine if the asteroid is too dim for an emulsion
    APMagPredict = intAPmag(plateTimes)

    #then we also need to generate the emulsion and filter dictionaries
    #to check the exposure times and the limiting mags
    expTimeDict = emul_filt_expTime()
    limMagDict = emul_filt_LimMag()

    #each plate covers a square of the sky, 6.4 x 6.4 degrees in size
    #the plate position lists the coordinates of the centre of the plate
    plateSize = 6.4

    #output = 'plates_493Griseldis.txt'
    output = "plateOutput_" + ephemeris[9:]
    #finalPredictions = open(output, 'w')
    
    finderChart = "FCfile_" + ephemeris[9:]

    #header = "Number     X (mm)     Y (mm)     RA(object)     DEC(object)     RA(plate)     DEC(plate)"
    header = '{a:<11}{b:<10}{c:<13}{d:<11}{e:<15}{f:<11}{g:<11}'.format(a = 'Number', b = 'X(mm)', c = 'Y(mm)', d = 'RA(obj)', e = 'DEC(obj)', f = 'RA(plate)', g = 'DEC(plate)')

    with open(output, 'w') as finalPredictions:
       finalPredictions.write(header + "\n" + "---------------------------------------------------------------------------------"+ "\n")
    finalPredictions.close()
    
    #this will be a text file with the RA and DEC in the format of the SuperCOSMOS survey to get finder charts
    fc = open(finderChart, 'w')
    
    with open(output, 'a') as finalPredictions:
    #we will now check whether the predicted position of the object,
    #at the time of each plate exposure, fits into that respective plate of size 6.4 x 6.4 degrees
        for i in range(plateTimes.size):
            
            #converting RA and DEC of a plate into degrees
            ra_float, dec_float = ra_dec_to_degrees(ra_list[i], dec_list[i])
            
            #calling the predicted position of the object (RA and DEC)
            #also converting them from hours, minutes, seconds to decimal degrees
            raPred, decPred = positionPredict[i][0], positionPredict[i][1]
            
            zeta, eta, cosDeg = LocalPlane(ra_float, dec_float, raPred, decPred)
            
            #the plate scale is in mm per angular unit (plate width per angle)
            #this is the same for both x and y as the plates are square
            plateScale = 3600/67.12
            if abs(zeta) < 3.2 and abs(eta) < 3.2 and cosDeg < 90:
                
                #this gives the coordinates of the given object with the centre of the plate as the origin
                zeta_mm = plateScale * zeta
                eta_mm = plateScale * eta
                
                #we must translate these coordinates and make the origin the bottom left plate corner
                zeta_mm += (356/2)
                eta_mm += (356/2)
                
                #writes a text file with the relevant information to be able to identify the plate in the ROE basement
                #this includes the number of the plate, the coordinates of the object in mm with the origin set as the bottom left corner of the plate.
                
                line = '{a:<11}{b:<10}{c:<13}{d:<11}{e:<15}{f:<11}{g:<11}'.format(a = str(int(plateNames[i])), b = str(round(zeta_mm,3)), c = str(round(eta_mm,3)), d = str(round(raPred, 5)), e = str(round(decPred, 5)), f = str(round(ra_float,5)), g = str(round(dec_float,5)))
                
                ra_format, dec_format = RA_DEC_degToMinsHrs(raPred, decPred)
                
                if dec_format[0] < 0:
                    fcLine = '{a:<3}{b:<3}{c:<7}{d:<4}{e:<3}{f:<7}'.format(a = str(int(ra_format[0])).zfill(2), b = str(int(ra_format[1])).zfill(2), c = str('{:.3f}'.format(ra_format[2])).zfill(6), d = str(int(dec_format[0])).zfill(2), e = str(int(dec_format[1])).zfill(2), f = str('{:.3f}'.format(dec_format[2])).zfill(6))
                
                else:
                    fcLine = '{a:<3}{b:<3}{c:<7}{d:^4}{e:<3}{f:<7}'.format(a = str(int(ra_format[0])).zfill(2), b = str(int(ra_format[1])).zfill(2), c = str('{:.3f}'.format(ra_format[2])).zfill(6), d = str(int(dec_format[0])).zfill(2), e = str(int(dec_format[1])).zfill(2), f = str('{:.3f}'.format(dec_format[2])).zfill(6))
 
                #print("THIS IS A TEST", plateNames[i], " ", expTimeList[i])
                """
                print("PREDICTED MAG", APMagPredict[i])
                print("THIS IS A TEST", plateNames[i], " ", emulFiltList[i], " ", expTimeList[i], " ", limMagDict[emulFiltList[i]] - 2.5*math.log10(expTimeDict[emulFiltList[i]]/expTimeList[i]))
                """

                if emulFiltList[i] in expTimeDict and expTimeList[i] > 0 and APMagPredict[i] < limMagDict[emulFiltList[i]] - 2.5*math.log10(expTimeDict[emulFiltList[i]]/expTimeList[i]):
                    
                    finalPredictions.write("\n" + line + "\n")
                    fc.write(fcLine + '\n')
                    
                elif emulFiltList[i] not in expTimeDict:
                    
                    finalPredictions.write("\n" + line + "\n")
                    fc.write(fcLine + '\n')
                    
                #finalPredictions.write("\n" + line + "\n")
                
    finalPredictions.close()
    fc.close()
    
    
#input the name of the ephemeris of desired body. Will be used later

ephemeris = input("What is the name of the ephemeris text file? ")
PlatePredict(ephemeris)

"""
THIS IS A BATCH RUNNER 
THIS RUNS A BUNCH OF EPHEMERIS FILES ONE BY ONE

ephemeris = open("AsteroidNames.txt", 'r')
count = 1
for line in ephemeris.readlines():
    print("Working on Asteroid number ", count)
    print("........")
    PlatePredict(line[:-1])
    print("........")
    print("Asteroid Number ", count, " finished running!")
    count += 1

"""
    
print("Operation 100% Complete!")