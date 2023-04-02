"""
This program plots orbits of asteroids based on the input ephemeris obtained 
from NASA's HORIZONS tool.

Also plots the point of perihelion, as well as the point at which plate exposures 
captured it.

MPhys
2022/23
Denis Gergov S1839787
"""


import numpy as np
import scipy.interpolate

import matplotlib.pyplot as plt
#mpl.rcParams['legend.fontsize'] = 10
#plt.rcParams['figure.dpi'] = 250
plt.rcParams['figure.dpi'] = 230

def readHorizons(file):
    file = file.readlines()
    #print(file)
    index = file.index("$$SOE\n")
    index_end = file.index("$$EOE\n")
    
    #all end at -160: aprt from vesta which is at -169
    file = file[(index + 1):index_end-1]
    jd = []
    x_list = []
    y_list = []
    z_list = []
    
    for i in range(0, int(len(file)), 4):
        
        jd_number = float(file[i][0:17])
        jd.append(jd_number)
        
        coords = file[i + 1]
        
        x = float(coords[4:26])
        x_list.append(x)
        
        y = float(coords[30:52])
        y_list.append(y)
        
        z = float(coords[56:-1])
        z_list.append(z)
    
    jd = np.asarray(jd)
    x_list = np.asarray(x_list)
    y_list = np.asarray(y_list)
    z_list = np.asarray(z_list)
    
    return jd, x_list, y_list, z_list

def findPoint(t, jd, x, y, z):
    """
    TIME IS IN JD
    """
    
    interpX = scipy.interpolate.interp1d(jd, x, fill_value="extrapolate")#2.25
    interpY = scipy.interpolate.interp1d(jd, y,  fill_value="extrapolate")
    interpZ = scipy.interpolate.interp1d(jd, z,  fill_value="extrapolate")
    
    return np.array([interpX(t), interpY(t), interpZ(t)])
    
    
name = "7968 Elst-Pizarro"
perihelionJD = 2456331.8164059711
mjd_plate = np.array([5215, 5244, 5250, 10468, 14423])

#name = "233P La Sagra"
#perihelionJD = 2457198.8311056532
#mjd_plate = np.array([3639, ])

#name = "259PGarradd"
#perihelionJD = 2456317.7461155881
#mjd_plate = np.array([7165, 10244, 13849])

#name = "432P-PANSTARRS"
#perihelionJD = 2459457.5342270886
#mjd_plate = np.array([12659, 13198])

#name = "6478 Gault"
#perihelionJD = 2458851.3724058005
#mjd_plate = np.array([9004, 14099])

#name = "118401 LINEAR"
#perihelionJD = 2457824.8145411033
#mjd_plate = np.array([14142, 16964])

#go back to this one
#name = "248370 2005 QN173"
#perihelionJD = 2459349.9437462082
#mjd_plate = np.array([10674, 15560])

#name = "279870 2001 NL19"
#perihelionJD = 2458251.8681119257
#mjd_plate = np.array([4232, 11586, 12541])

#name = "300163 2006 VW139"
#perihelionJD = 2457700.5713323741
#mjd_plate = np.array([7632, 15440, 15445])

#name = "P2013 R3"
#perihelionJD = 2456509.6602348089
#mjd_plate = np.array([17616])

#name = "P2016 J1-A"
#perihelionJD = 2457563.7133528781
#mjd_plate = np.array([12565])

#name = "2675 Tolkien"
#perihelionJD = 2457678.1637576544
#mjd_plate = np.array([4845, 13992])

#name = "Hale-Bopp"
#perihelionJD = 2450537.1349071441
#mjd_plate = np.array([19011, 19310])

test = open(name + ".txt", "r")
#test = open("6478 Gault.txt", "r")

#reading in the 
info = readHorizons(test)
jd = info[0]
x = info[1]
y = info[2]
z = info[3]

#importing and plotting Earth's orbit for comparison
earth = open("earth.txt", "r")
e = readHorizons(earth)

#interpolating perihelion coordinates
perihelion_coords = findPoint(perihelionJD, jd, x, y, z)

#setting up the plot
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

#plotting an orbit
ax.plot(x, y, z, label='Asteroid Orbit', color="#294096")
ax.plot(e[1], e[2], e[3], label="Earth's Orbit", color="#8a1a34")
ax.scatter(perihelion_coords[0], perihelion_coords[1], perihelion_coords[2], marker = "^", label="Perihelion", color="orange", linewidth=4)

#finding point of orbit when exposure was taken
mjd_file = np.loadtxt("MJD.txt").T

color_list = ['orchid','tomato', 'gold', '#ad0057',  '#009c00', '#000dc2' ]
k = 0
for q in mjd_plate:

    index = np.where(mjd_file[0] == q)
    row = mjd_file.T[index]
    plateMJD = row[0][1]

    #coordinates of point in question!!
    plateMJD += 2400000.5
    plate_coords = findPoint(plateMJD, jd, x, y, z)
    ax.scatter(plate_coords[0], plate_coords[1], plate_coords[2], label=str(q), color=color_list[k])
    k+=1

ax.set_title("Orbit of {0}".format(name))
ax.set_xlabel("x (km)")
ax.set_ylabel("y (km)")
ax.set_zlabel("z (km)")
ax.legend(loc='center left', bbox_to_anchor=(1.07, 0.5), fontsize=7)
plt.show()
"""
#setting up the plot - 2D X-Y PLANE
fig = plt.figure()
ax = fig.add_subplot()

#plotting an orbit
ax.plot(x[0:35000], y[0:35000], label='Asteroid Orbit', color="#294096", zorder=0)
ax.plot(e[1], e[2], label="Earth's Orbit", color="#8a1a34")
ax.scatter(perihelion_coords[0], perihelion_coords[1], marker = "^", label="Perihelion", color="orange", linewidth=4)

#finding point of orbit when exposure was taken
mjd_file = np.loadtxt("MJD.txt").T

color_list = ['orchid','tomato', 'gold', '#ad0057',  '#009c00', '#000dc2' ]
k = 0
for q in mjd_plate:

    index = np.where(mjd_file[0] == q)
    row = mjd_file.T[index]
    plateMJD = row[0][1]

    #coordinates of point in question!!
    plateMJD += 2400000.5
    plate_coords = findPoint(plateMJD, jd, x, y, z)
    ax.scatter(plate_coords[0], plate_coords[1], label=str(q), color=color_list[k])
    k+=1


ax.set_title("{0} Orbit".format(name))
ax.set_xlim(-6e8, 6e8)
ax.set_ylim(-6e8, 6e8)
ax.set_xlabel("x (km)")
ax.set_ylabel("y (km)")
ax.legend(loc='center left', bbox_to_anchor=(1.07, 0.5), fontsize=7)
plt.gca().set_aspect('equal')
plt.tight_layout()
plt.show()
"""