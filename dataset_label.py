import os
import cv2

for dir in ["KANAMA", "ISKEMI", "INMEYOK"]:
    files = os.listdir(os.path.join("TRAINING", dir, "PNG"))
    for i, filename in enumerate(files):
        print(filename)
        input_path = os.path.join("TRAINING", dir, "PNG", filename)
        output_path = os.path.join("TRAINING1", dir + str(i) + ".png")
        print(input_path, output_path)
        img = cv2.imread(input_path, 0)
        cv2.imwrite(output_path, img)
        k = cv2.waitKey(0)
        if k == ord("q"):
            break

cv2.destroyAllWindows()