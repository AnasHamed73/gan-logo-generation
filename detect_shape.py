#!/usr/bin/env python

import cv2
import argparse
import numpy as np
import os
import shutil



import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np

IMG_SIZE = 32
DEBUG = True

#imgs_path = "/home/kikuchio/Documents/courses/gan-seminar/logan-code/LLD_favicons_clean_png/all"
imgs_path = "/home/kikuchio/Documents/courses/gan-seminar/logan-code/tmp"
encoding_file = "one_hot_encoding"


def show_img(img, name="title", delay=500):
    if not DEBUG:
        return
    cv2.imshow("img", img)
    cv2.waitKey(delay)
    cv2.imwrite(name+".png", img)


def show_text(text, title, delay=1000):
    if not DEBUG:
        return
    img = np.zeros((42, 42, 3), dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (2 if text == "square" else 5,25)
    fontScale = 0.4 if text == "other" else 0.35
    fontColor = (255, 255, 255)
    lineType = 2
    cv2.putText(img, text, bottomLeftCornerOfText, font, fontScale, fontColor, 1, cv2.LINE_AA)
    show_img(img, "shape", delay)


def one_pad(img, pwx, pwy):
    """Pads a given image with zero at the border.
    """
    padded_img = np.full((img.shape[0]+2*pwy, img.shape[1]+2*pwx, img.shape[2]), 255, dtype=np.uint8)
    for i in range(0, img.shape[0]):
        padded_img[pwy+i][pwx: img.shape[1]+pwx][:] = img[i][:][:]
    return padded_img     


def direction(frm, to):
    frm_row = frm[0]
    frm_col = frm[1]
    to_row = to[0]
    to_col = to[1]
    if frm_row < to_row:  # down
        if frm_col < to_col: # down-right
            #print("going down right")
            return 315
        elif frm_col == to_col: # down-straight
            #print("going down straight")
            return 270
        elif frm_col > to_col: # down-left
            #print("going down left")
            return 225
    elif frm_row > to_row:  # up
        if frm_col < to_col: # up-right
            #print("going up right")
            return 45
        elif frm_col == to_col: # up-straight
            #print("going up straight")
            return 90
        elif frm_col > to_col: # up-left
            #print("going up left")
            return 135
    elif frm_row == to_row:
        if frm_col > to_col: # left-straight
            #print("going left straight")
            return 180
        elif frm_col < to_col: # right-straight
            #print("going right straight")
            return 0
    pass


def get_neighbor(p, contour, traversed, radius):
    cur_row = p[0]
    cur_col = p[1]
    for i in range(0, max(1, radius)):
        row = cur_row+radius-i
        if contour.get(row) is not None:
            for col in contour[row]:
                if (row, col) not in traversed and (row, col) != p and abs(cur_col - col) <= radius:
                #    print("Found one in next row")
                    return (row, col)
    if contour.get(cur_row) is not None:
        for col in contour[cur_row]:
            #print("checking in current row of ", cur_row)
            if (cur_row, col) not in traversed and (cur_row, col) != p and col != cur_col and abs(cur_col - col) <= radius:
               # print("Found one in current row")
                return (cur_row, col)
    #print("contour.get(cur_row-radius) is not None: ", contour.get(cur_row-radius) is not None)
    #print("point to check: ", p)
    for i in range(0, max(1, radius)):
        row = cur_row-radius+i
        #print("row: ", row)
        if contour.get(row) is not None:
            for col in contour[row]:
                #print("point to check: ", p)
                #print("checking ", (row, col))
                #print("(row, col) not in traversed: ", (row, col) not in traversed)
                #print("(row, col) != p:", (row, col) != p)
                #print("abs(cur_col - col) <= radius", abs(cur_col - col) <= radius)
                #print("radius: ", radius)
                #print("available points in row: ", contour.get(cur_row-radius))
                #print("********************")
                if (row, col) not in traversed and (row, col) != p and abs(cur_col - col) <= radius:
                 #   print("found one: ", (row, col))
                    return (row, col)
    return None


def closest_neighbor(current, contour, traversed, max_radius):
    for r in range(1, max_radius+1):
        neighbor = get_neighbor(current, contour, traversed, r)
        if neighbor is not None:
            return neighbor
    return None


def get_directions(contour):
    dir_vals = [0, 45, 90, 135, 180, 225, 270, 315]
    dirs_hisogram = {k: 0 for k in dir_vals}
    traversed = []
    con_keys = list(contour.keys())
    con_vals = list(contour.values())
    init_point = (con_keys[0], contour[con_keys[0]][0])
    current = init_point
    neighbor = closest_neighbor(current, contour, traversed, IMG_SIZE)
    while neighbor is not None:
        #print(f"neighbor of {current} is {neighbor}")
        cur_dir = direction(current, neighbor)
        if cur_dir is not None:
            dirs_hisogram[cur_dir] += 1
        traversed.append(current)
        current = neighbor
        neighbor = closest_neighbor(current, contour, traversed, IMG_SIZE)
    return dirs_hisogram


def is_circle(sorted_d):
    min_freq = sorted_d[0][1]
    max_freq = sorted_d[-1][1]
    if DEBUG:
        print("min freq: ", min_freq)
        print("max freq: ", max_freq)
    if abs(max_freq - min_freq) <= 11:
        return True
    return False


def is_square(sorted_d):
    top_4 = sorted_d[-4:]
    top_4_keys = [t[0] for t in top_4]
    top_4_vals = [t[1] for t in top_4]
    bottom_4_vals = [t[1] for t in sorted_d[0:4]]
    bottom_4_vals_sum = sum(bottom_4_vals)
    rng = abs(top_4_vals[-1] - top_4_vals[0])
    sq_dirs = sorted(top_4_keys) == [0, 90, 180, 270] or sorted(top_4_keys) == [45, 135, 225, 315]
    if DEBUG:
        print("sorted(top_4_keys): ", sorted(top_4_keys))
        print("bottom 4 sum: ", bottom_4_vals_sum)
        print("square rng: ", rng)
        print("is sq dirs: ", sq_dirs)
    if sq_dirs and rng <= 9 and sorted_d[-5][1] < 9 and bottom_4_vals_sum < 28:
        return True
    return False


def inc_val(d, key):
    if d.get(key) is None:
        d[key] = 0
    d[key] += 1


def similar_borders(img):
    print("image shape: ", img.shape)
    pix_hist = {}
    for pix in img[0]:
        inc_val(pix_hist, (pix[0], pix[1], pix[2]))
    for pix in img[-1]:
        inc_val(pix_hist, (pix[0], pix[1], pix[2]))
    for pix in img[:][0]:
        inc_val(pix_hist, (pix[0], pix[1], pix[2]))
    for pix in img[:][-1]:
        inc_val(pix_hist, (pix[0], pix[1], pix[2]))
    max_freq = sorted(list(pix_hist.values()))[-1]
    max_precent = max_freq / (img.shape[0] * 4)
    if DEBUG:
        print("hist: ", pix_hist)
        print("max freq precent: ", max_precent)
    return False



def get_shape(cnt_points_dict, img):
    dirs = get_directions(cnt_points_dict)
    sorted_d = sorted(dirs.items(), key=lambda kv: kv[1])
    if DEBUG:
        print("directions: ", dirs)
    if is_square(sorted_d):
        return "square"
    elif is_circle(sorted_d):
        return "circle"
    return "other"


def to_binary(img):
    h, w, _ = img.shape
    border_pix = img[0][5]
    res = np.ones((h, w))
    for r in range(h):
        for c in range(w):
            if img[r][c][0] != border_pix[0] or img[r][c][1] != border_pix[1] or img[r][c][2] != border_pix[2]: 
                res[r][c] = 0
    return res


def thin(cnt):
    cpy = cnt[:]
    h, w = cnt.shape
    res = np.zeros((h, w))
    print("copy shape ", cpy.shape)
    for i in range(h):
        nonzeros = np.nonzero(cpy[i])[0]
        min_nz = nonzeros.min()
        max_nz = nonzeros.max()
        res[i][min_nz] = 255
        res[i][max_nz] = 255
    cpy_t = np.transpose(cpy)
    res = np.transpose(res)
    for i in range(h):
        nonzeros = np.nonzero(cpy[i])[0]
        min_nz = nonzeros.min()
        max_nz = nonzeros.max()
        res[i][min_nz] = 255
        res[i][max_nz] = 255
    return np.transpose(res)


def one_hot(shape):
    if shape == "square":
        return "1,0,0"
    elif shape == "circle":
        return "0,1,0"
    else:
        return "0,0,1"


def get_contour_binary(img, img_in):
    img, contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    areas = [cv2.contourArea(x) for x in contours]
    if areas == []:
        return None
    max_index, max_area = max(enumerate(areas), key = lambda x: x[1])
    max_contour = contours[max_index]
    
    img_out = img_in
    w, h, c = img_in.shape 
    o = cv2.drawContours(img_out, [max_contour], 0, (0, 0, 255), 1)
    
    cnt = np.zeros((h, w, 3), dtype=np.uint8)
    for c in max_contour:
        for p in c:
            cnt[p[1]][p[0]][0] = 255
    kernel = np.ones((1, 1), np.uint8)
    cnt = cv2.dilate(cnt, kernel, 1)
    cnt = cv2.erode(cnt, kernel, 1)
    show_img(cnt, "contour_closed")
    
    cnt_bin = cnt[:, :, 0]
    show_img(cnt_bin, "contour")
    return cnt_bin


def points_to_dict(cnt_bin):
    nz = np.nonzero(cnt_bin)
    cnt_bin_points = [(x, y) for x, y in zip(nz[0], nz[1])]
    cnt_point_dict = {x: [] for x in set(nz[0])}
    for x, y in cnt_bin_points:
        cnt_point_dict[x].append(y)
    return cnt_point_dict


def get_image_shape(img_path):
    img_in = cv2.imread(img_path)
    img_name = img_path.split("/")[-1]
    if DEBUG:
        print("Image name: ", img_name)

    padded = one_pad(img_in, 5, 5)
    show_img(padded, "padded")
    img_in = padded
    
    w, h, c = img_in.shape 
    
    img = cv2.Canny(img_in, 50, 50)
    show_img(img, "canny")
    
    kernel = np.ones((4, 4), np.uint8)
    img = cv2.dilate(img, kernel, 1)
    show_img(img, "canny_dilated")
    
    cnt_bin = get_contour_binary(img, img_in)
    if cnt_bin is None:
        print("found logo with no contour")
        return "other"
    
    cnt_point_dict = points_to_dict(cnt_bin)
    
    shape = get_shape(cnt_point_dict, img_in)
    show_text(shape, "shape", delay=1000)

    return shape


######MAIN


for img in sorted(os.listdir(imgs_path)):
    shape = get_image_shape(os.path.join(imgs_path, img))
    if DEBUG:
        print("shape: ", shape)
    #ohe = one_hot(shape)
    #print(f"{img}:{ohe}")
    #with open(encoding_file, "a+") as f:
    #    f.write(img+ ":" + ohe+"\n")
    #with open(shape+"_imgs", "a+") as f:
    #    f.write(img+"\n")

#img_dest = os.path.join(shape, img_name)
#if not DEBUG:
#    shutil.move(img_src, img_dest)

if DEBUG:
    print("________________________")
