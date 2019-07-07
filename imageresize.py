import os
import PIL
from PIL import Image

src = "./datav1" 
dst = "./resizedData" 

os.mkdir(dst)

counter = 1;
for each in os.listdir(src):
    if each == '.DS_Store':
        continue;
    img = Image.open('datav1/' + each)
    img = img.resize((128, 128), PIL.Image.ANTIALIAS)
    img.save('resizedData/' + str(counter) + '.jpg')
    counter+=1
    
print("Done")