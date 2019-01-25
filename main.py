import cv2
import numpy as np
import time
import argparse

global cam_0_idx
global cam_1_idx
global cam_2_idx
global cam_3_idx
global cam_4_idx
global image_width
global image_height
global fps

image_width = 320
image_height = 240
num_cams = 6
saturation = 50
fps = 30

def CAPTURE_FRAME(cam_0_idx,cam_1_idx,cam_2_idx,cam_3_idx,cam_4_idx,cam_5_idx):

	print("[INFO] Capturing images")
	start = time.time()
	cam_0 = cv2.VideoCapture(cam_0_idx)
	cam_3 = cv2.VideoCapture(cam_3_idx)
	CAMERA_SETUP(cam_0,cam_0_idx);
	CAMERA_SETUP(cam_3,cam_3_idx);
	_,frame_0 = cam_0.read()
	_,frame_3 = cam_3.read()
	cam_0.release();
	cam_3.release();

	cam_1 = cv2.VideoCapture(cam_1_idx)
	cam_4 = cv2.VideoCapture(cam_4_idx)
	CAMERA_SETUP(cam_1,cam_1_idx);
	CAMERA_SETUP(cam_4,cam_4_idx);
	_,frame_1 = cam_1.read()
	_,frame_4 = cam_4.read()
	cam_1.release();	
	cam_4.release();

	cam_2 = cv2.VideoCapture(cam_2_idx)
	cam_5= cv2.VideoCapture(cam_5_idx)
	CAMERA_SETUP(cam_2,cam_2_idx);
	CAMERA_SETUP(cam_5,cam_5_idx);
	_,frame_2 = cam_2.read()
	_,frame_5 = cam_5.read()
	cam_2.release();
	cam_5.release();

	end = time.time()
	print("[INFO] Time taken to capture {} images = {}".format(num_cams, end-start))

	frames = (frame_0, frame_1, frame_2, frame_3, frame_4, frame_5)
	return frames


def CAPTURE_FRAME_LOW_RES(cam_0_idx,cam_1_idx,cam_2_idx,cam_3_idx,cam_4_idx,cam_5_idx):

	print("[INFO] Capturing images")
	start = time.time()
	
	_,frame_0 = cam_0.read()
	_,frame_3 = cam_3.read()
	_,frame_1 = cam_1.read()
	_,frame_4 = cam_4.read()
	_,frame_2 = cam_2.read()
	_,frame_5 = cam_5.read()
	
	end = time.time()
	print("[INFO] Time taken to capture {} images = {}".format(num_cams, end-start))

	frames = (frame_0, frame_1, frame_2, frame_3, frame_4, frame_5)
	return frames


def CAMERA_SETUP(cam, camID):
	
	ret1 = cam.set(cv2.CAP_PROP_FRAME_HEIGHT, image_height)
	ret2 = cam.set(cv2.CAP_PROP_FRAME_WIDTH, image_width)
	ret3 = cam.set(cv2.CAP_PROP_SATURATION, saturation)
	ret4 = cam.set(cv2.CAP_PROP_FPS, fps)
	if ret1:
                print("[INFO] Camera {} frame height set successfully to {}".format(camID, image_height))
        else:
                print("[ERROR] Camera {} failed to set frame height to {}".format(camID, image_height))
        if ret2:
                print("[INFO] Camera {} frame width set successfully to {}".format(camID, image_width))
        else:
                print("[ERROR] Camera {} failed to set frame width to {}".format(camID, image_width))

        if ret3:
                print("[INFO] Camera {} saturation set successfully to {}".format(camID, saturation))
        else:
                print("[ERROR] Camera {} failed to set saturation to {}".format(camID, saturation))

        if ret4:
                print("[INFO] Camera {} FPS set successfully to {}".format(camID, fps))
        else:
                print("[ERROR] Camera {} failed to set FPS to {}".format(camID, fps))




def ROTATE_IMAGE(image):
	M = cv2.getRotationMatrix2D((image_width/2,image_height/2),-90,1)

	# rotation calculates the cos and sin, taking absolutes of those.
	cos = np.abs(M[0, 0])
	sin = np.abs(M[0, 1])

	# find the new width and height bounds
	nW = int((image_height * sin) + (image_width * cos))
	nH = int((image_height * cos) + (image_width * sin))

	# adjust the rotation matrix to take into account translation
	M[0, 2] += (nW / 2) - image_width/2
	M[1, 2] += (nH / 2) - image_height/2

	# rotate image with the new bounds and translated rotation matrix
	rotated_mat = cv2.warpAffine(image, M, (nW, nH))

	return rotated_mat



if __name__ == "__main__":

        # construct the argument parser and parse the arguments
        ap = argparse.ArgumentParser()
        ap.add_argument("-iw", "--integratedwebcam", required=True,
                help="Is there an integrated webcam? 1 for yes, 0 for no")
        ap.add_argument("-lr", "--lowres", required=True,
                help="Low res mode. 1 for enable, 0 for disable")
        args = vars(ap.parse_args())

        # arguments returned as strings
        isIntegratedWebcam = args["integratedwebcam"]
        tryLowRes = args["lowres"]
        print("[INFO] IntegratedWebcam = %s" % isIntegratedWebcam)
        print("[INFO] LowRes = %s" % tryLowRes)
        
	# only used if using Python2.7 and have an integrated webcam on the PC
        if bool(int(isIntegratedWebcam)):
                for x in range(1, num_cams+1):
                        cam = cv2.VideoCapture(x)
                        CAMERA_SETUP(cam,x)
                        _, frame = cam.read()
                        rotated_image = ROTATE_IMAGE(frame)
                        cv2.imshow('cam_rotated', rotated_image)
                        cv2.waitKey(5)
                        Camera=input("[INPUT] Which camera is this? (1-6)")

                        # overloading 'Camera' is done here as input is read as integer in Python2.7
                        # in Python3.7 it is already read as a string
                        if(str(Camera)=='1'):
                                cam_0_idx = x
                        elif(str(Camera)=='2'):
                                cam_1_idx = x
                        elif(str(Camera)=='3'):
                                cam_2_idx = x
                        elif(str(Camera)=='4'):
                                cam_3_idx = x
                        elif(str(Camera)=='5'):
                                cam_4_idx = x
                        elif(str(Camera)=='6'):
                                cam_5_idx = x
                        cam.release()

        else:
                for x in range(0, num_cams):
                        cam = cv2.VideoCapture(x)
                        CAMERA_SETUP(cam,x)
                        _, frame = cam.read()
                        rotated_image = ROTATE_IMAGE(frame)
                        cv2.imshow('cam_rotated', rotated_image)
                        cv2.waitKey(5)
                        Camera=input("[INPUT] Which camera is this? (0-5)")

                        # overloading 'Camera' is done here as input is read as integer in Python2.7
                        # in Python3.7 it is already read as a string
                        if(str(Camera)=='1'):
                                cam_0_idx = x
                        elif(str(Camera)=='2'):
                                cam_1_idx = x
                        elif(str(Camera)=='3'):
                                cam_2_idx = x
                        elif(str(Camera)=='4'):
                                cam_3_idx = x
                        elif(str(Camera)=='5'):
                                cam_4_idx = x
                        elif(str(Camera)=='6'):
                                cam_5_idx = x
                        cam.release()

        cv2.destroyWindow('cam_rotated')
        blank_border = np.zeros((image_width,image_height,3))

        if bool(int(tryLowRes)):
                print("trying low res")
                image_width = 160
                image_height = 120
                cam_0 = cv2.VideoCapture(cam_0_idx)
                cam_3 = cv2.VideoCapture(cam_3_idx)
                CAMERA_SETUP(cam_0,cam_0_idx);
                CAMERA_SETUP(cam_3,cam_3_idx);
                cam_1 = cv2.VideoCapture(cam_1_idx)
                cam_4 = cv2.VideoCapture(cam_4_idx)
                CAMERA_SETUP(cam_4,cam_4_idx);
                CAMERA_SETUP(cam_1,cam_1_idx);
                cam_2 = cv2.VideoCapture(cam_2_idx)
                cam_5= cv2.VideoCapture(cam_5_idx)
                CAMERA_SETUP(cam_2,cam_2_idx);
                CAMERA_SETUP(cam_5,cam_5_idx);
                
                while(1):
                        frames_r = [0] * num_cams
                        frames = CAPTURE_FRAME_LOW_RES(cam_0_idx,cam_1_idx,cam_2_idx,cam_3_idx,cam_4_idx,cam_5_idx)

                        for x in range(0, num_cams):
                                frames_r[x] = ROTATE_IMAGE(frames[x])

                        #concatenated_image = np.concatenate((frames_r[0],blank_border,frames_r[1],blank_border,frames_r[2],blank_border,frames_r[3],blank_border,frames_r[4],blank_border,frames_r[5]), axis=1)
                        concatenated_image = np.concatenate((frames_r[0],frames_r[1],frames_r[2],frames_r[3],frames_r[4],frames_r[5]), axis=1)

                        cv2.namedWindow('concatenated_image')
                        cv2.moveWindow("concatenated_image", 0,0)
                        cv2.imshow('concatenated_image', concatenated_image)

                        # Wait 5ms and see if 'ESC' key was pressed
                        k = cv2.waitKey(5) & 0xFF
                        if k == 27:
                                break
                
                
        else:
                while(1):
                        frames_r = [0] * num_cams
                        frames = CAPTURE_FRAME(cam_0_idx,cam_1_idx,cam_2_idx,cam_3_idx,cam_4_idx,cam_5_idx)

                        for x in range(0, num_cams):
                                frames_r[x] = ROTATE_IMAGE(frames[x])

                        #concatenated_image = np.concatenate((frames_r[0],blank_border,frames_r[1],blank_border,frames_r[2],blank_border,frames_r[3],blank_border,frames_r[4],blank_border,frames_r[5]), axis=1)
                        concatenated_image = np.concatenate((frames_r[0],frames_r[1],frames_r[2],frames_r[3],frames_r[4],frames_r[5]), axis=1)

                        cv2.namedWindow('concatenated_image')
                        cv2.moveWindow("concatenated_image", 0,0)
                        cv2.imshow('concatenated_image', concatenated_image)

                        # Wait 5ms and see if 'ESC' key was pressed
                        k = cv2.waitKey(5) & 0xFF
                        if k == 27:
                                break

	



