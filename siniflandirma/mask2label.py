import cv2
import os

class_num = [1,2]
mask_path_list = ["/content/DATASET/ISKEMI_MASK","/content/DATASET/KANAMA_MASK"]
mask_label_path_list = ["/content/DATASET/ISKEMI_MASK_LABEL","/content/DATASET/KANAMA_MASK_LABEL"] 
# You don't have to create a folder, the code will do it.
                        
for mask_path, mask_label_path, class_number in zip(mask_path_list,mask_label_path_list,class_num):
    try:
        os.mkdir(mask_label_path)
        print("Directory ", mask_label_path, " Created ")
    except FileExistsError:
        print("Directory ", mask_label_path, " already exists")

    for img_name in os.listdir(mask_path):
        # print(img_name)
        img = cv2.imread(mask_path + "/" + img_name)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h_img, w_img = gray.shape
        objects = []
        ROI_number = 0
        cnts = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            # cv2.rectangle(img, (x, y), (x + w, y + h), (26, 245, 55), 1)
            objects.append([(x+w/2)/w_img, (y+h/2)/h_img, w/w_img, h/h_img])
            # ROI is to take the image inside the box.
            # ROI = img[y:y+h, x:x+w]  # if you want 1 channel (gray image) - ROI = gray[y:y+h, x:x+w]
            # cv2.imwrite('ROI_{}.png'.format(ROI_number), ROI)
            # ROI_number += 1
        file = open(mask_label_path + "/" + img_name[:-4] + ".txt", "w")
        for obj in objects:
            a = str(class_number) + " " + " ".join([str(x) for x in obj])
            file.write(a)
            if obj == objects[-1]:
                break
            file.write("\n")
        file.close()
