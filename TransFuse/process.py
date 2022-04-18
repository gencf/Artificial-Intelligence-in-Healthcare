import numpy as np
import cv2
import os
import shutil 
from windowing import output
import albumentations as A
from windowing import output

def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result

def augmentation(image, atype):

    if atype == "rotated":
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif atype == "rotated45":
        image = rotate_image(image, 45)
    elif atype == "hflip":
        image = cv2.flip(image,0)
    elif atype == "vflip":
        image = cv2.flip(image,1)
    elif atype == "equalize":
        transform = A.Compose([
            A.Equalize(p=1)
        ])
        image = transform(image=image)["image"]

    return image


def dcm_loader(path, atype=0):
    pixels = output(path).astype(np.uint8)
    pixelse = augmentation(pixels.copy(), "equalize")    
    return [pixels, pixelse]

def binary_loader(path, atype=0):
    img = cv2.imread(path,0)
    return [img, img]


def process(paths, npy_path): # process 2
    height = 192
    width = 256

    for path in paths:

        count = 0
        length = len(os.listdir(path))
        imgs = np.uint8(np.zeros([length, height, width, 3]))
        file_names = np.uint8(np.zeros([length, height, width, 3]))

        img_path = os.path.join(path, "PNG")

        for i,f in enumerate(os.listdir(img_path)):
            file_name = os.path.join(img_path, f)
            img = cv2.imread(file_name)
            img = cv2.resize(img, (width, height))


            imgs[count] = img
            file_names.append(file_name)
            count +=1 

        if "iskemi" in path.lower():
            np.save(os.path.join(npy_path, 'data_iskemi.npy'), imgs)
            np.save(os.path.join(npy_path, 'mask_iskemi.npy'), masks)
        
        elif "kanama" in path.lower():
            np.save(os.path.join(npy_path, 'data_kanama.npy'), imgs)
            np.save(os.path.join(npy_path, 'mask_kanama.npy'), masks)
 
 
root = 'new_dataset/' # change to your data folder path
data_f = ['ISKEMI/train/PNG/','ISKEMI/test/PNG/','KANAMA/train/PNG/','KANAMA/test/PNG/']
mask_f = ['ISKEMI/train/MASKS/','ISKEMI/test/MASKS/','KANAMA/train/MASKS/','KANAMA/test/MASKS/']
save_name = ['iskemi_train', 'iskemi_test','kanama_train', 'kanama_test']

height = 192
width = 256

new_dataset = "/content/new_dataset/"

for index in range(2):
    for j in range(2*index, 2*index+2):
        print('processing ' + data_f[j] + '......')
        count = 0
        
        path = root + data_f[j]
        mask_p = root + mask_f[j]

        save_path = os.path.join(new_dataset, data_f[j])
        mask_path = os.path.join(new_dataset, mask_f[j])
        os.makedirs(save_path, exist_ok=True)
        os.makedirs(mask_path, exist_ok=True)
        
        length = len(sorted(os.listdir(root + data_f[j])))
        
        if "train" in path:
            length *= 2
        
        imgs = np.uint8(np.zeros([length, height, width, 3]))
        masks = np.uint8(np.zeros([length, height, width]))

        for i in range(len(os.listdir(path))):
            print(i+1, "/" , len(os.listdir(path)))
            dcm_path = path + str(i) + ".dcm"
            m_path = mask_p + str(i) + ".png"

            # img = pydicom.dcmread(dcm_path).pixel_array

            if "test" in path:
                myimgs = [output(dcm_path)]
                mymasks = [cv2.imread(m_path, 0)]
            else:
                myimgs = dcm_loader(dcm_path)
                mymasks = binary_loader(m_path)

            for idx,img in enumerate(myimgs):
                img_resized = cv2.resize(img.copy(), (width, height))
                mask_resized = cv2.resize(mymasks[idx].copy(), (width, height))

                imgs[count] = img_resized.astype(np.uint8)
                masks[count] = mask_resized

                cv2.imwrite(os.path.join(save_path, str(count) + ".png"), img.copy().astype(np.uint8))
                cv2.imwrite(os.path.join(mask_path, str(count) + ".png"), mymasks[idx])

                count +=1 

        print(imgs.shape) 
        print(masks.shape)
        np.save('/content/TransFuse/data_{}.npy'.format(save_name[j]), imgs)
        np.save('/content/TransFuse/mask_{}.npy'.format(save_name[j]), masks)

    print("done")

path = "/content/TransFuse"
iskemi_path = "/content/TransFuse/ISKEMI_npy_files"
kanama_path = "/content/TransFuse/KANAMA_npy_files"
save_path = "/content/drive/MyDrive/İNAN/SağlıktaYapayZeka/TransFuse/dicom"
os.makedirs(iskemi_path, exist_ok=True)
os.makedirs(kanama_path, exist_ok=True)

for f in os.listdir(path):
    if "iskemi" in f and ".npy" in f:
        shutil.move(os.path.join(path,f), os.path.join(iskemi_path,f))
    if "kanama" in f and ".npy" in f:
        shutil.move(os.path.join(path,f), os.path.join(kanama_path,f))
