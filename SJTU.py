import cv2
import numpy as np
import pandas as pd
import os




def HSV_threshold(image):
    #apply threshold to RGB image
    image_cpy = image.copy()
    #convert the image to HSV color space to detect yellow lines better.
    HSV_image = cv2.cvtColor(image_cpy,cv2.COLOR_BGR2HSV)

    #set lower bound and upper bound for the white lane color
    lower_limit_w= np.array([0,0,110])
    upper_limit_w= np.array([200,80,255])
    #create the threshold for the white lane
    whitelane_mask = cv2.inRange(HSV_image,lower_limit_w,upper_limit_w)
    #create upper and lower bounnds for yellow lane color
    lower_limit_y= np.array([12,80,140])
    upper_limit_y= np.array([30,255,255])
    #create threshold for yellow lane
    yellowlane_mask = cv2.inRange(HSV_image,lower_limit_y,upper_limit_y)
    #combine the two masks
    combined = cv2.bitwise_or(whitelane_mask, yellowlane_mask)
    
    return combined
    # we are assuming the camera is always fixed at the same position. 
def ROI_mask(binary_image):
    #apply triangular mask on thresholded image to get the area of interest
    #make a black mask of the binary image
    black_mask= np.zeros_like(binary_image) 
    #get the height and the width of the image
    im_height = binary_image.shape[0]
    im_width = binary_image.shape[1]
    #create triangular mask. Note: you can not have half a pixel, thus use int to make any possible floats int
    triangle = np.array([[(0,im_height),(int(im_width/2), int( im_height/2.2)),(im_width,im_height)]])    
    # we do not want to overwrite useful pixel data, so we will create a mask to apply to the image. Use cv2.fillpolly to create triangle shape.
    #Apply mask(white triangle on image) into the black mask
    cv2.fillPoly(black_mask, [triangle],(255))
    #apply the mask to the binary image
    #extract common white pixels
    lane_line = cv2.bitwise_and(binary_image,black_mask)

    return(lane_line)

    
    

def frame_extractor(input_video, output_video):

    #opening the video file and then extract the frames.
    # check if vdeo exists  

    if os.path.exists(input_video)== False:
        print("Video file does not exist.")
        return(None)
    else:
        print("Video upload successfully.")
    # Open the video file
    video_cap = cv2.VideoCapture(input_video)
    #ret True if frame is read successfully. When False, video file ended. Now read the video frame by frame.
    ret, image = video_cap.read()
    #check if the first image is read
    if ret == False:
        print("image can not be read")
    # use cv2.VideoWriter to create output video file
    output= cv2.VideoWriter(output_video,cv2.VideoWriter_fourcc(*'mp4v'),video_cap.get(cv2.CAP_PROP_FPS), (image.shape[1],image.shape[0]))
    while ret:

        #push image through threshold function
        HSV_threshold_image = HSV_threshold(image)

        #apply region of interest mask after pushing the image through the threshold functiion
        ROI = ROI_mask(HSV_threshold_image) 
        #Below are image displaying/ testing commands. Uncomment for use.
        #cv2.imshow("Threshold Mask", HSV_threshold_image)
        #cv2.waitKey(0)  
        #cv2.destroyAllWindows()

        #ROI is in one channel, convert back to 3 channels to write into VideoWriter
        colored_ROI = cv2.cvtColor(ROI,cv2.COLOR_GRAY2BGR)
        #Convert white lanes into red lanes
        colored_ROI[np.where((colored_ROI == [255,255,255]).all(axis = 2))] = [0,0,255]
        #overlay the original video with the colored_ROI
        final_image = cv2.addWeighted(image,0.8,colored_ROI,2,0)
        #write the images to the output file
        output.write(final_image )
        ret,image = video_cap.read()
    cv2.destroyAllWindows()
    video_cap.release()
    output.release()
    return(output)

        

    
print(frame_extractor(r"C:\Users\tess_\Downloads\road lane detection\video.mp4", r"C:\Users\tess_\Downloads\road lane detection\test1-3.mp4"))




