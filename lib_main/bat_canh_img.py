import cv2
import imutils
import argparse
import numpy as np
import points_line #list_points
import lib_icp

def bat_canh(image,list_data,list_add = [],arr_goc = np.array([[],[],[]]),point = [-1,-1]):
    min_x = 5000
    max_y = 0
    rmse = 100
    h1,h2,s1,s2,v1,v2 = list_data
    h_img,w_img,_ = image.shape
    img = np.ones((h_img,w_img,3),dtype=np.uint8)*255
    # detect black shape using HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower = np.array([h1,s1,v1])
    upper = np.array([h2,s2,v2])
    mask = cv2.inRange(hsv, lower, upper)
    # contour
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    # print amount of number of black objects
    list_points = []
    px = []
    py = []
    arr = np.array([[],[],[]])
    for c in cnts:
        cv2.drawContours(img, [c], -1, (0,0,0), 1)
        cv2.drawContours(image, [c], -1, (0,255,0), 1)
    if len(list_add) > 0:
        target_value = [0,0,0]
        list_poi = np.argwhere(np.all(img == target_value, axis=-1))
        for i1 in range(0,len(list_poi)):
            x = int(list_poi[i1][1])
            y = int(list_poi[i1][0])
            if arr_goc[0].shape[0] != 0:
                px.append(x)
                py.append(y)
            for i2 in range(0,len(list_add)):
                x1 = int(list_add[i2][0])
                y1 = int(list_add[i2][1])
                x2 = int(list_add[i2][2])
                y2 = int(list_add[i2][3])
                if x >= x1 and x <= x2 and y >= y1 and y <= y2:
                    list_points.append([x,y])
                    if arr_goc[0].shape[0] == 0:
                        px.append(x)
                        py.append(y)
                        if min_x > x:
                            min_x = x
                        if max_y < y:
                            max_y = y
    px = np.array(px)
    py = np.array(py)
    arr = np.ones((3,px.shape[0]))
    arr[0,:] = px
    arr[1,:] = py
    arr[2,:] = np.ones(px.shape[0])
    diem1 = []
    diem2 = []
    if arr_goc[0].shape[0] != 0:
        arr_goc = np.array(arr_goc)
        (r, t, k,rmse) = lib_icp.IterativeClosestPoint(arr_goc,arr)
        new_arr = lib_icp.ApplyTransformation(arr_goc,r,t)
        for l in range(0,int(new_arr[0].shape[0])):
            cv2.circle(image, (int(new_arr[0,l]),int(new_arr[1,l])),1,(255,0,255), 1) 
            cv2.circle(image, (int(arr_goc[0,l]),int(arr_goc[1,l])),1,(0,0,255), 1) 
        if point[0] != -1:
            cv2.circle(image, (int(arr_goc[0,point[0]]),int(arr_goc[1,point[0]])),1,(255,255,0), 5) 
            cv2.circle(image, (int(new_arr[0,point[0]]),int(new_arr[1,point[0]])),1,(255,255,0), 5) 
            diem1 = [int(new_arr[0,point[0]]),int(new_arr[1,point[0]])]
        if point[1] != -1:
            cv2.circle(image, (int(arr_goc[0,point[1]]),int(arr_goc[1,point[1]])),1,(255,255,0), 5) 
            cv2.circle(image, (int(new_arr[0,point[1]]),int(new_arr[1,point[1]])),1,(255,255,0), 5) 
            diem2 = [int(new_arr[0,point[1]]),int(new_arr[1,point[1]])]
    else:
        for l in range(0,len(list_points)):
            if len(list_points[l]) > 0:
                cv2.circle(image, (int(list_points[l][0]),int(list_points[l][1])),1,(255,255,0), 1) 
    # if point != -1:
    #     cv2.circle(image, (int(arr_goc[0,point]),int(arr_goc[1,point])),1,(255,0,0), 5) 
    #     cv2.circle(image, (int(new_arr[0,point]),int(new_arr[1,point])),1,(255,0,0), 5) 
    return image, arr,min_x,max_y,rmse,diem1,diem2

