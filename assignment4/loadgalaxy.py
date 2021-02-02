import os

import numpy as np
from PIL import Image


def LoadDir(dirname):
  imgs = []
  for imgname in os.listdir(dirname):
    img = Image.open(os.path.join(dirname, imgname))
    img = img.convert('LA')  # conver to grayscale
    img = img.resize([20, 20])
    
    img = np.squeeze(np.array(img)[:, :, 0])
    imgs.append(img)

  return np.array(imgs)

train_imgs = LoadDir('galaxy/train')
print(train_imgs.shape)
val_imgs = LoadDir('galaxy/val')
print(val_imgs.shape)
test_imgs = LoadDir('galaxy/test')
print(test_imgs.shape)
