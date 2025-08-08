import cv2
import numpy as np
import pandas as pd
import os
import math



def gauss_blur(image, kernel_size, sigma):
    #Use Gaussian blur on image to eliminate noise
    #sigma shows how much the neighboring pixels influence the pixel value
    #have odd kernel size so that there is actually a center kernel pixel. Now, lets make the kernel
    #find the mean(or as I ike to think it, the center) of the kernel
    mean = (kernel_size -1)/2
    #use fromfunction() to execute the gaussian fucntion on the kernel. Use lamba to make a function inline (easier than creating a seperate funcion)
    kernel =np.fromfunction(lambda m,n: (1/np.sqrt(2*math.pi*(sigma**2)))*np.exp(-((n-mean)**2 +(m-mean)**2)/(2*(sigma**2))),(kernel_size, kernel_size))
    #find weighted average of kernel
    kernel = kernel/np.sum(kernel)
    #apply the kernel to the image. (-1) means to keep depth of the image.
    blurred_image = cv2.filter2D(image,-1,kernel)
    return(blurred_image)


def sobel_operator(image, kernel_size):
    #find gradient of image with sobel operator
    #find the gradient in the x directiom
    edge_x = cv2.Sobel(image,cv2.CV_64F,1,0,kernel_size)
    #find the gradient in the y direction
    edge_y = cv2.Sobel(image,cv2.CV_64F,0,1,kernel_size)
    #find the gradient magnitude
    grad_magnitude = np.sqrt(edge_x**2 + edge_y**2)
    #angle of gradient(for the direcion). it will have the form of m by n.
    grad_direction = np.arctan2(edge_y,edge_x)
    return(grad_magnitude, grad_direction)

def maxima_suppression(gradient, direction):
    #supress pixels that are not close to the maxima gradient
    m,n = direction.shape
    pi = np.pi
    # create black image same demension size
    new_image = np.zeros_like(gradient)
    # make a loop to get each angle in the m by n grid. Go through each pixel
    for i in range(1, m-1): # we do this to look at neighbors
        for j in range(1,n-1):
            angle = direction[i,j]
            # group into 4 directions based on angle: 0,pi/4,pi/2,3pi/4,pi. it is much faster than looping through each angle!
            new_angle = np.floor((angle + pi/8)/(pi/4))*(pi/4)
            if new_angle == 0 or angle == pi:
                #look at slopes of neigboring pixels. we will lookng at the pixels along the same angle line
                #for 0 and pi we are looking at the horizontal neigboring pixels
                # (note: cause the angles are horizontal. think of a 5 by 5 grid to visualize)
                n1 = gradient[i,j-1]
                n2 = gradient[i,j+1]
            elif new_angle == (pi/4):
                #now look at pixels along the pi/4 angle in the grid.
                n1 = gradient[i+1,j-1]
                n2 = gradient[i-1,j+1]
            elif new_angle == (pi/2):
                n1 = gradient[i-1,j]
                n2 = gradient[i+1,j]
            else:
                # this is for the 3pi/4 angle. look at neigboring pixels along this angle
                n1 = gradient[i+1,j+1]
                n2 = gradient[i-1,j-1]

            # check to see if the neighboring slopes are greater or not greater than selected point
            if gradient[i,j] >= n1 or gradient[i,j]>= n2:
                #insert the gradient into the black image if gradient of seleted point is greater than neighbors
                new_image[i,j]= gradient[i,j]
    # return a newly altered image
    return(new_image)



def edge_thresh(image,min,max):
    # uses min annd max threshold values to flter out strng edges from weak edges. 
    #gradients with max values above threshold are kept (make white), while below threshold are discarded (make black)
    # reduces  noise
    new_image = np.zeros_like(image)
    m,n = image.shape
    #the boarders do not matter too much
    for i in range(1,m-1):
        for j in range(1,n-1):
            pixel = image[i,j]
            if pixel >= max:
                # if pixel is above the threshold make it white
                new_image[i,j] = 255
                #append location of white pixes to a list to use for hough transformation
            elif pixel <= min:
                #if pixel is below threshold make black
                new_image[i,j] = 0
            elif np.any(image[i-1:i+2,j-1:j+2]>= max):
                #at this point of the algorithm we are left the weak points
                #the above condition checks if the neigboring pixels are connected to a max pixel we keep,
                new_image[i,j] = 255
            else:
                #we discard everything else
                new_image[i,j] = 0
                # weak edges in the threshold
    return(new_image)
                




    

def CannyEdge(image, kernel_size, sigma, thresh_max,thresh_min):
    # find edges of the image
    image_cpy = image.copy()
    # Step 1: grayscale the image
    #first, split the image into r,g,b channels
    r,g,b = cv2.split(image_cpy)
    gray_image = (0.299*r) + (0.587*g) + (0.114*b)
    # Step 2: use Gaussian blur to lessen the noise from the image
    gaussian_blur = gauss_blur(gray_image, kernel_size, sigma)
    gradient_mag, gradient_direct = sobel_operator(gaussian_blur, kernel_size)
    #step 3: non maxima suppression. Use magnitude and direction to make the edges thinner
    suppressed_image = maxima_suppression(gradient_mag, gradient_direct)
    #step 4: hysteresis thresholding function. filter strong and weak edges
    thresh_image = edge_thresh(suppressed_image,thresh_min,thresh_max) 
    return(thresh_image)

#ROI mask function only wors for grayscale image
def ROI_mask(image):
    #get the height and the width of the image
    
    #apply triagular mask on image to get the area of interest
    im_height = image.shape[0]
    im_width = image.shape[1]
    #check if it is a 3 channel image
    if len(image.shape)==3:
        black_mask = np.zeros((im_height,im_width),np.uint8)
    #make a black mask of the binary image
    else:
        black_mask= np.zeros_like(image) 
    #create quad mask. Note: you can not have half a pixel, thus use int to make any possible floats int
    quad =  np.array([[(0,im_height),(int(im_width *0.4), int(im_height* 0.7)),( int(im_width *0.7),int( im_height*0.7)),(im_width,im_height)]])    
    # we do not want to overwrite useful pixel data, so we will create a mask to apply to the image. Use cv2.fillpolly to create quad shape.
    #Apply mask(white triangle on image) into the black mask
    cv2.fillPoly(black_mask, [quad],(255))
    #apply the mask to the binary image
    #do the following step only if 3 channel image
    if len(image.shape)== 3:
        black_mask = cv2.cvtColor(black_mask,cv2.COLOR_GRAY2BGR)
    lane_line = cv2.bitwise_and(image,black_mask)

    return(lane_line)


def hough(image):
    #hough transformation for lane line extraction  
    m,n = image.shape
    #we dont do 0 -> pi because we don't want double angles
    #lets make a grid for the accumulator
    angles = np.arange(-np.pi/2,np.pi/2,np.pi/180)
    accum_col = len(angles)
    rhos = int(np.ceil(np.sqrt((m**2)+(n**2))))
    #the following line of code is borrowed Lane Detection: Implementing a canny Edge Detector and Hough Transform from  Scratch
    all_rhos = np.linspace(-rhos,rhos,2*rhos +1)
    accum_row =len(all_rhos)
    # accumulator that looks like a grid with votes. More votes signal there is a line
    white_pixels_accum = np.zeros((accum_row, accum_col))
    for i in range(m):
        for j in range(n):
            if image[i,j] == 255:
                for theta_indx in range(accum_col):
                    #p is the prependicular distance from origin
                    #use theta_indx to gget the angle 
                    theta = angles[theta_indx]
                    rho = (j*np.cos(theta))+ (i*np.sin(theta))
                    #this is an interesting cocept but, you take rho and subtract it from all elements in rho
                    #you taake the absolute vale after. Once that , you will see that whichever number in the array is closest to rho will have 
                    #a small value. thus do argmin to extract index.
                    rho_indx = int(np.round(rho +rhos))#np.argmin(np.abs(rho - all_rhos))
                    #make a vote at this index
                    white_pixels_accum[rho_indx,theta_indx]+= 1
    
    return(white_pixels_accum,angles,all_rhos)





def frame_extractor(input_video, output_video, kernel_size = 3,sigma = 1, thresh_max = 150, thresh_min = 50 ):

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
    frame_count = 0
    while ret:
        print(f"Processing frame {frame_count}")
        HSV_image = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
        lower_limit_y= np.array([12,80,140])
        upper_limit_y= np.array([35,255,255])
        #create threshold for yellow lane since yellow lines are otherwise not detected
        yellowlane_mask = cv2.inRange(HSV_image,lower_limit_y,upper_limit_y)

        #push image through threshold function
        CE = CannyEdge(image, kernel_size, sigma, thresh_max, thresh_min)
        CE = CE.astype(np.uint8)
        combined = cv2.bitwise_or(CE, yellowlane_mask)
        ROI_masked_im = ROI_mask(combined)
        accumulator, angles, all_rho = hough(ROI_masked_im)
        #I am using this idea from chatGPt to find the thresh min by take the max and then taking a percent of it for the min. I thought it was very versatile
        vote_min = 0.75* np.max(accumulator)
         #np.where returns a tuple of rows and columns. unpact tuple
        rhos_indx, thetas_indx = np.where(accumulator >= vote_min)
        #we are now going to draw our red lines on black image
        line_mask = np.zeros_like(image)
        for z in range(len(rhos_indx)):
            #we want to draw lines now. so convert to cartesian ccoordinates

            a = np.cos(angles[thetas_indx[z]])
            b = np.sin(angles[thetas_indx[z]])
            c = all_rho[rhos_indx[z]] 
            x_0 = a* c # we are converting froom polar to cartesian coordinates. ex: x = rho * cos(theta)
            y_0 = b * c
            #rotate 90 degrees to get the line we draw along. will you -sin for x and  cos for y
            x_1 = int(x_0 + (2000* (-b)))
            y_1 = int(y_0 +(2000*(a)))
            x_2 = int(x_0 - (2000*(-b)))
            y_2 = int(y_0 - (2000*a))
            cv2.line(line_mask, (x_1, y_1), (x_2, y_2), (0, 0, 250), thickness=9)           

        roi_mask_2 = ROI_mask(line_mask)
        final_image = cv2.addWeighted(image,0.8,roi_mask_2,2,0)

        #Below are image displaying/ testing commands. Uncomment for use.
        #cv2.imshow("Threshold Mask", HSV_threshold_image)
        
        
        output.write(final_image )
        ret,image = video_cap.read()
        frame_count += 1
    cv2.destroyAllWindows()
    video_cap.release()
    output.release()
    return(output)

print(frame_extractor(r"C:\Users\tess_\Downloads\SJTU task 2\road lane detection\input.mp4", r"C:\Users\tess_\Downloads\SJTU task 2\task2test5_2.mp4", 3, 1, 150, 50))
