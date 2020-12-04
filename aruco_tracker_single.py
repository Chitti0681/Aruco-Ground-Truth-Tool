
import scipy.io
import getopt
import sys
import math
import numpy as np
from cv2 import cv2
import cv2.aruco as aruco
import glob
from datetime import datetime
import csv
import numpy as np

###### converts from cartesian to spherical co-ordinates ########


def cart2sph(x, y, z):
    hxy = np.hypot(x, y)
    r = np.hypot(hxy, z)
    el = np.arctan2(z, hxy)
    az = np.arctan2(y, x)
    return np.degrees(az), np.degrees(el), r


def parseArgument(argv):
    """So there are 6 input argumnents
        1st is the tagSize
        2nd is the output file name

        this is how we are supposed to pass the arguments
    """
    fileName = ''
    tagSize = None

    try:
        opts, args = getopt.getopt(
            argv, "ht:o:", ["tagSize=", "fileName"])
    except getopt.GetoptError:
        print('test.py -t <tagSize> -o <output filename>')
        sys.exit(2)
    for opt, arg in opts:
        # help argument
        if opt == '-h':
            print('test.py -t <tagSize> -o <output filename>')
            sys.exit()
        # input file
        elif opt in ("-t", "--tagSize"):
            tagSize = (float)(arg)

        elif opt in ("-o", "--fileName"):
            fileName = arg

    return fileName, tagSize


########## converts csv to mat file ###############
def csvToMat(fileName):
    with open(fileName+str(time)+'.csv') as f:
        reader = csv.reader(f)
        i = 0
        data = []

        for row in reader:
            rowData = []
            if i != 0:
                for j, elem in enumerate(row):

                    if j < 9:
                        rowData.append(float(elem))
                    if j == 9:
                        rowData.append(int(float(elem)))
                data.append(rowData)
            i += 1
    matrix = np.array(data)

    scipy.io.savemat(fileName+str(time)+'.mat', {'csvmatrix': matrix})

##### read fileName and tagSize#####


tagSize = float(input("Enter the tag Size: "))
fileName = input("Enter the file Name")
xTag = float(input("Input the x-coordinate in cm: "))
yTag = float(input("Input the y-coordinate in cm: "))
zTag = float(input("Input the z-coordinate in cm: "))


######## Location of tag ##############
tagLoc = np.array([-xTag/100, yTag/100, zTag/100, 1])


# import rectangleArea as ra
cv2.namedWindow("preview")
cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)


###################  Hard Coded calibration matrix for the calibration images in the image folder #########
mtx = np.array([[1.16853450e+03, 0.00000000e+00, 9.45814963e+02],
                [0.00000000e+00, 1.15460499e+03, 5.37990908e+02],
                [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

dist = np.array([[2.49754342e-01, -6.08282123e-01, -
                  3.96471621e-04, 4.01400822e-03, 3.04199503e-01]])


flag = True
time = datetime.now()
# ------------------ ARUCO TRACKER ---------------------------
with open(fileName+str(time)+'.csv', 'w', newline='') as csvfile:
    recordWriter = csv.writer(csvfile, delimiter=',')
    recordWriter.writerow(['Distance r(in m)', 'Azimuth(in deg)', 'Elevation(in deg)', 'x-axis',
                           'y-axis', 'z-axis', 'Rvec1', 'Rvec2', 'Rvec3', 'TimeStamp(UTC)', 'TimeStamp(Datetime)'])
    while (flag):
        ret, frame = cap.read()

        # operations on the frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # set dictionary size depending on the aruco marker selected
        aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)

        # detector parameters can be set here (List of detection parameters[3])
        parameters = aruco.DetectorParameters_create()
        parameters.adaptiveThreshConstant = 10

        # lists of ids and the corners belonging to each id
        corners, ids, rejectedImgPoints = aruco.detectMarkers(
            gray, aruco_dict, parameters=parameters)

        # font for displaying text (below)
        font = cv2.FONT_HERSHEY_SIMPLEX

        # check if the ids list is not empty
        # if no check is added the code will crash
        if np.all(ids != None):

            # estimate pose of each marker and return the values
            # rvet and tvec-different from camera coefficients

            ######## adding an offset of 1 cm and obtaining the rvec and tvec ############
            rvec, tvec, _ = aruco.estimatePoseSingleMarkers(
                corners, (tagSize/100)-0.01, mtx, dist)

            # (rvec-tvec).any() # get rid of that nasty numpy value array error
            # print(rvec)

            dateTimeObj = datetime.now()

            #### cartesian coordinate system###
            x = -tvec[0][0][0]
            y = -tvec[0][0][1]
            z = tvec[0][0][2]

            az, ele, r = cart2sph(z, x, y)
            recordWriter.writerow([(str)(r), (str)(az), (str)(ele), (str)(x), (str)(y), (str)(z), (str)(
                rvec[0][0][0]), (str)(rvec[0][0][1]), (str)(rvec[0][0][2]), (str)(dateTimeObj.timestamp()), (str)(dateTimeObj)])
            matArray = [r, az, ele, x, y, z, rvec[0][0][0], rvec[0]
                        [0][1], rvec[0][0][2], dateTimeObj.timestamp()]
            matArray = np.array(matArray)
            scipy.io.savemat(fileName+str(time)+'.mat',
                             {'csvmatrix': matArray})
            ######## Apppling Rodrigues method on the 3x1 rotational vectors to obtain the 3x3 rotation matrix ############
            [rotation, jacobian] = cv2.Rodrigues(rvec)
            print("Rotation: ", rotation)
            print("Re-test: ", cv2.Rodrigues(rotation))
            print("Translation:", x, y, z)
            print("rotational:", rvec)
            tvec1 = tvec.reshape(3, 1)
            print(tvec.shape)
            print(rotation.shape)
            print("Rotaion of z axis:", np.degrees(rvec[0][0][2]))
            line = np.array([0, 0, 0, 1])

            ###### Rel position of tag w.r.t marker Test marker position ######
            testMarker = tagLoc

            ##### rot-tran Matrix ##########
            matrixRot = np.hstack([rotation, tvec1])
            matrixRot = np.vstack([matrixRot, line])

            #### rel position of the tag w.r.t camera #####
            relPosition = np.matmul(matrixRot, testMarker)
            relPosition = np.delete(relPosition, -1)

            print("Relative_Position", relPosition)
            print(az, ele, r)
            print(cart2sph(relPosition[2], -relPosition[0], -relPosition[1]))
            for i in range(0, ids.size):

                # draw axis for the aruco markers

                aruco.drawAxis(frame, mtx, dist, rvec[i], tvec[i], 0.1)
                cv2.putText(frame, "r= %.3f m az= %.0f deg  el= %.0f deg" % (
                    r, az, ele), (0, 230), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (244, 0, 0), 2)

                flag = False
             # draw a square around the markers
            aruco.drawDetectedMarkers(frame, corners)

            # code to show ids of the marker found
            strg = ''
            for i in range(0, ids.size):
                strg += str(ids[i][0])+', '

            cv2.putText(frame, "Id: " + strg, (0, 64), font,
                        1, (0, 255, 0), 2, cv2.LINE_AA)

        else:
            # code to show 'No Ids' when no markers are found
            cv2.putText(frame, "No Ids", (0, 64), font,
                        1, (0, 255, 0), 2, cv2.LINE_AA)

        # display the resulting frame
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
