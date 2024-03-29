from PIL import Image
import os
src = "resizedData"
dst = "resized_black/"

os.mkdir(dst)

for each in os.listdir(src):
    png = Image.open(os.path.join(src,each))
    if png.mode == 'RGBA':
        png.load()
        background = Image.new("RGB", png.size, (0,0,0))
        background.paste(png, mask=png.split()[3]) # 3 is the alpha channel
        background.save(os.path.join(dst,each.split('.')[0] + '.jpg'), 'JPEG')
    else:
        png.convert('RGB')
        png.save(os.path.join(dst,each.split('.')[0] + '.jpg'), 'JPEG')

print("Done")