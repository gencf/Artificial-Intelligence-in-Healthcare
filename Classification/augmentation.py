import cv2
import os
import numpy as np

def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result

def augmentation(img_dirs):
    for dir in img_dirs:
        files = os.listdir("rsna")
        if os.path.exists(dir) == False:
            os.mkdir(dir)
        else:
            print(dir + " directory already exits.")
            #continue
        for i, filename in enumerate(files):
            input_path = os.path.join("rsna",filename)
            output_path = os.path.join(dir,filename)
            img = cv2.imread(input_path, 0)
            if dir == "rsna_rotated":
                image = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            elif dir == "rotated45":
                image = rotate_image(img, 45)
            elif dir == "rotated315":
                image = rotate_image(img, -45)
            cv2.imwrite(output_path, image)
            print(i)
            

img_dirs = ["rsna_rotated"]
augmentation(img_dirs)