from argparse import ArgumentParser
from multiprocessing import Process, Queue

import cv2
import glob
import os
import cvlib as cv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


from mark_detector import MarkDetector
from os_detector import detect_os
from pose_estimator import PoseEstimator
from stabilizer import Stabilizer

print("OpenCV version: {}".format(cv2.__version__))

# multiprocessing may not work on Windows and macOS, check OS for safety.
detect_os()

CNN_INPUT_SIZE = 128



def get_face(detector, img_queue, box_queue):
    """Get face from image queue. This function is used for multiprocessing"""
    while True:
        image = img_queue.get()
        box = detector.extract_cnn_facebox(image)
        box_queue.put(box)

ims = []    



videos_path = [f for f in glob.glob("data/*.mp4")]

for video in videos_path:
    video_src = video
    cap = cv2.VideoCapture(video_src)
    
    #create folders for saving images
    path = "path name" + video_src[5:-10]
    try:
        os.mkdir(path)
    except OSError:
        print ("Creation of the directory %s failed" % path)
    else:
        print ("Successfully created the directory %s " % path)
    
    
    if video_src == 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        
    _, sample_frame = cap.read()

    # Introduce mark_detector to detect landmarks.
    mark_detector = MarkDetector()

    # Setup process and queues for multiprocessing.
    img_queue = Queue()
    box_queue = Queue()
    img_queue.put(sample_frame)
    box_process = Process(target=get_face, args=(mark_detector, img_queue, box_queue,))
    box_process.start()

    # Introduce pose estimator to solve pose. Get one frame to setup the
    # estimator according to the image size.
    height, width = sample_frame.shape[:2]
    pose_estimator = PoseEstimator(img_size=(height, width))

    # Introduce scalar stabilizers for pose.
    pose_stabilizers = [Stabilizer(
        state_num=2,
        measure_num=1,
        cov_process=0.1,
        cov_measure=0.1) for _ in range(6)]

    tm = cv2.TickMeter()
    ind = 0
    lc = 0.9

    while True:
        ind += 1
        # Read frame, crop it, flip it, suits your needs.
        frame_got, frame = cap.read()
        if frame_got is False:
            break

        # Crop it if frame is larger than expected.
        # frame = frame[0:480, 300:940]

        # If frame comes from webcam, flip it so it looks like a mirror.
        if video_src == 0:
            frame = cv2.flip(frame, 2)

        # Pose estimation by 3 steps:
        # 1. detect face;
        # 2. detect landmarks;
        # 3. estimate pose

        # Feed frame to image queue.
        img_queue.put(frame)

        # Get face from box queue.
        facebox = box_queue.get()
        # print(facebox)
        if facebox is not None:
            # Detect landmarks from image of 128x128.
            face_img = frame[facebox[1]: facebox[3],
                                facebox[0]: facebox[2]]
            face_img = cv2.resize(face_img, (CNN_INPUT_SIZE, CNN_INPUT_SIZE))
            face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)

            tm.start()
            marks = mark_detector.detect_marks([face_img])
            tm.stop()

            # Convert the marks locations from local CNN to global image.
            marks *= (facebox[2] - facebox[0])
            marks[:, 0] += facebox[0]
            marks[:, 1] += facebox[1]

            # Uncomment following line to show raw marks.
            # mark_detector.draw_marks(
            #     frame, marks, color=(0, 255, 0))

            # Uncomment following line to show facebox.
            # mark_detector.draw_box(frame, [facebox])

            # Try pose estimation with 68 points.
            pose = pose_estimator.solve_pose_by_68_points(marks)

            # Stabilize the pose.
            steady_pose = []
            pose_np = np.array(pose).flatten()
            for value, ps_stb in zip(pose_np, pose_stabilizers):
                ps_stb.update([value])
                steady_pose.append(ps_stb.state[0])
            steady_pose = np.reshape(steady_pose, (-1, 3))

            points2d = pose_estimator.get_point_2d(frame, steady_pose[0], steady_pose[1])
            for i in  range(4):
                frame = cv2.circle(frame, (int(points2d[i][0][0]),int(points2d[i][0][1])), radius=0, color=(0, 0, 255), thickness=-1)

            AX,AY = points2d[0][0]
            BX,BY = points2d[1][0]
            CX,CY = points2d[2][0]
            DX,DY = points2d[3][0]
            
            ##################################### Homography #####################################
            faces, confidences = cv.detect_face(frame)
            # loop through detected faces and add bounding box
            im_face = frame
            for face in faces:
                (startX,startY) = face[0],face[1]
                (endX,endY) = face[2],face[3]


            # Four corners of the face in source image
            pts_src = np.array([[BX, BY], [CX, CY], [DX, DY],[AX, AY]], dtype=float)


            # Read destination image.
            im_dst = frame
            # Four corners of the face in destination image.
            #pts_dst = lc*np.array([[BX0, BY0], [CX0, CY0], [DX0, DY0],[AX0, AY0]], dtype=float) + (1-lc) * pts_src
            pts_dst = lc*np.array([[min(AX,BX), min(BY,CY)], [max(CX,DX), min(CY,BY)], [max(DX,CX), max(DY,AY)],[min(AX,BX), max(AY,DY)]], dtype=float) + (1-lc) * pts_src

            # Calculate Homography
            h, status = cv2.findHomography(pts_src, pts_dst)

            # Warp source image to destination based on homography
            im_out = cv2.warpPerspective(frame, h, (im_dst.shape[1],im_dst.shape[0]))

            # Display images
            #fig=plt.figure(figsize=(16, 16))

            #fig.add_subplot(1, 2, 1)
            #plt.imshow(frame)
            #fig.add_subplot(1, 3, 2)
            #plt.imshow(im_face)
            #fig.add_subplot(1, 2, 2)
            #plt.imshow(im_out)
            #################################################################################
            
            # save rectified imgage
            cv2.imwrite(os.path.join(path , str(ind).zfill(6) + '.jpg'), im_out)
            
            
    ind = 0
    # Clean up the multiprocessing process.
    box_process.terminate()
    box_process.join()

