import os
from PIL import Image

lbl_cont = {}
with open("/home/yuan/data/HisMap/SMOLdata/train_hameln2.txt", 'r') as f:
    lines = f.readlines()
    for line in lines:
        if 'c3821_2002' in line:
            lbl_file = os.path.join("/home/yuan/data/HisMap/SMOLdata/hameln2/lbls", line.strip().split('.')[1] + '.png')
            lbl = Image.open(lbl_file)
            lbl = lbl.convert('L')

            lbl_pixels = lbl.getdata()
            unique_pixels = set(lbl_pixels)
            for pixel in unique_pixels:
                if pixel in lbl_cont:
                    lbl_cont[pixel] += 1
                else:
                    lbl_cont[pixel] = 1

print(lbl_cont)