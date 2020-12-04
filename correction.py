
'''
This code does the correction of mapping the tags position from w.r.t the aruco marker to w.r.t to the camera

'''
import os
import scipy.io
import getopt
import sys
import math
import numpy as np
from cv2 import cv2
import glob
from datetime import datetime
import csv
import numpy as np


######### converts from cartesian to spherical co-ordinate system #######
def cart2sph(x, y, z):
    hxy = np.hypot(x, y)
    r = np.hypot(hxy, z)
    el = np.arctan2(z, hxy)
    az = np.arctan2(y, x)
    return np.degrees(az), np.degrees(el), r

########## converts from csv to mat file format ###########
def csvToMat(fileName):
    csvMatrix=np.genfromtxt(fileName+'.csv', delimiter=',', dtype=None, encoding='utf-8')
    
    csvMatrix = np.delete(csvMatrix, 0, axis=0)
    csvMatrix=np.delete(csvMatrix, -1, axis=1)
    csvMatrix=csvMatrix.astype(np.float)

    scipy.io.savemat(fileName+'.mat', {'csvmatrix':csvMatrix}) 


###### Read the inputs from the user ######
fileName=input("Enter the name of the folder: ")


####### The file that has the csv files #######
files = glob.glob(fileName+'/*.csv')

####### x-y-z cordinates of the tag w.r.t to the aruco marker #########
x=float(input("Input the x-coordinate in cm: "))
y=float(input("Input the y-coordinate in cm: "))
z=float(input("Input the z-coordinate in cm: "))

#### do the correction in each csv file and store it in a new csv and mat file ############
for inputFile in files:
    readings=np.genfromtxt(inputFile, delimiter=',', dtype=None, encoding='utf-8')
    readings = np.delete(readings, 0, axis=0)

    line=np.array([0,0,0,1])
    tagLoc=np.array([-x/100,y/100,z/100,1])

    inputFile=os.path.splitext(inputFile)[0]
    newFileName=inputFile+'-corrected'

    with open(newFileName+'.csv', 'w', newline='') as csvfile:
        recordWriter=csv.writer(csvfile,delimiter=',')
        recordWriter.writerow(['Distance r(in m)','Azimuth(in deg)','Elevation(in deg)','x-axis','y-axis','z-axis','Rvec1','Rvec2','Rvec3','TimeStamp(UTC)','TimeStamp(Datetime)'])
        for i,row in enumerate(readings):
            tvec1=-float(row[3])
            tvec2=-float(row[4])
            tvec3=float(row[5])

            translation=np.array([tvec1,tvec2,tvec3])
            translation=translation.reshape(3,1)
            rotation=np.array([float(row[6]),float(row[7]),float(row[8])])

            ###### Obtain the rotation matrix from rotation vector #######
            [rotMatrix,jacobian] = cv2.Rodrigues(rotation)

            ##### rot-tran Matrix ##########
            rotMatrix=np.hstack([rotMatrix,translation])
            rotMatrix=np.vstack([rotMatrix,line])

            ####### Calculate the relative position of the tag w.r.t the camera ######
            relPosition=np.matmul(rotMatrix,tagLoc)
            relPosition=np.delete(relPosition,-1)

            relX=-relPosition[0]
            relY=-relPosition[1]
            relZ=relPosition[2]
            az,el,r= cart2sph(relZ,relX,relY)
            recordWriter.writerow([(str)(r),(str)(az),(str)(el),(str)(relX),(str)(relY),(str)(relZ),(str)(row[6]),(str)(row[7]),(str)(row[8]),(str)(row[9]),(str)(row[10])])
        csvToMat(newFileName)


