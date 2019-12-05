#!/usr/bin/env python

import os
import shutil

#count_limit = 55000
count_limit = 10

square_imgs="./square_imgs"
circle_imgs="./circle_imgs"
other_imgs="./other_imgs"

output_dir="./partitioned"
square_imgs_output_dir = os.path.join(output_dir, "square")
circle_imgs_output_dir = os.path.join(output_dir, "circle")
other_imgs_output_dir = os.path.join(output_dir, "other")

original_imgs_dir = "/home/kikuchio/Documents/courses/gan-seminar/logan-code/LLD_favicons_clean_png/all"


def copy_imgs_to_partition(imgs, dest_dir, limit):
    with open(imgs, "r") as f:
        for i, img in enumerate(f.readlines(), 0):
            if i >= limit:
                break
            img_src = os.path.join(original_imgs_dir, img.strip())
            img_dest = os.path.join(dest_dir, img.strip())
            print(f"moving {img_src} to {img_dest}")
            shutil.copy(img_src, img_dest)
            
##### MAIN

copy_imgs_to_partition(circle_imgs, circle_imgs_output_dir, count_limit)
copy_imgs_to_partition(square_imgs, square_imgs_output_dir, count_limit)
copy_imgs_to_partition(other_imgs, other_imgs_output_dir, count_limit)
    


