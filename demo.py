import os
from PIL import Image
import matplotlib.pyplot as  plt

image = Image.open('/home/lzhpersonal/darknet/data/img/0101_1857_001_left.jpg')
img = image.crop((1000,100,5000,5000))
plt.imshow(img)
plt.show()