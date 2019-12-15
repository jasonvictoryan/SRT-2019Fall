import cv2
import pyrealsense2 as rs
import numpy as np
import matplotlib.pyplot as plt
from decimal import Decimal

def Remove_outliers(Depth):
	AllNums=[]
	for i in range(len(Depth)):
		for j in range(len(Depth[i])):
			AllNums.append(Depth[i][j])
	AllNums.sort()
	return np.array(AllNums[int(0.4*len(AllNums)):int(0.6*len(AllNums))])

def discern(img,depth_img,aligned_depth_frame):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cap = cv2.CascadeClassifier(
        r"./cascade.xml"
    )
    Rects = cap.detectMultiScale(
        gray, scaleFactor=1.2, minNeighbors=3, minSize=(30, 30))
    if len(Rects):
        for Rect in Rects:
            x, y, w, h = Rect
            cv2.rectangle(img, (x, y), (x + h, y + w), (0, 255, 0), 2)
            cv2.rectangle(depth_img, (x, y), (x + h, y + w), (0, 255, 0), 2)
			
			# get dist
            depth = np.asanyarray(aligned_depth_frame.get_data())
            depth = depth[x:x+h,y:y+w].astype(float)
			
			# Remove_outliers
            Processed_depth=Remove_outliers(depth)
			
            depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
            Processed_depth = Processed_depth * depth_scale
			
            dist, _, _, _ = cv2.mean(Processed_depth)
            dist_temp = Decimal(dist).quantize(Decimal('0.000'))
            print("Detected a {0} {1:.3} meters away.".format('Target', dist_temp))
			
			# put text
            cv2.putText(img, str(Decimal(dist).quantize(Decimal('0.00')))+"m",
                (x, y - 5),
                cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,255,0))
				
    images = np.hstack((img, depth_img))
    cv2.imshow("Image", images)
    

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)  #(Type, camera resolution_x, camera resolution_y, etc.)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
 
# Start streaming
profile = pipeline.start(config)

try:
    while True:

        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
		
		# Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.05), cv2.COLORMAP_JET)
		
		# Create alignment primitive with color as its target stream:
        colorizer = rs.colorizer()
        align = rs.align(rs.stream.color)
        frames = align.process(frames)

        # Update color and depth frames:
        aligned_depth_frame = frames.get_depth_frame()
        colorized_depth = np.asanyarray(colorizer.colorize(aligned_depth_frame).get_data())
		
		#discern
        discern(color_image,colorized_depth,aligned_depth_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:

    # Stop streaming
    pipeline.stop()
    cv2.destroyAllWindows()
