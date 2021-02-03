import os

import numpy as np
from PIL import Image

def LoadDir(dirname):
  num_files = len(os.listdir(dirname))
  imgs = []
  
  count = 0
  for imgname in os.listdir(dirname):
    print(f'  {count/num_files*100:6.3}', end='\r')
    img = Image.open(os.path.join(dirname, imgname))
    img = img.convert('LA')  # conver to grayscale
    img = img.resize([20, 20])
    
    img = np.squeeze(np.array(img)[:, :, 0])
    imgs.append(img)
    count += 1

  # convert to numpy
  imgs_arr = np.array(imgs)

  # reshape to be 1D vectors
  imgs_vect = np.reshape(imgs_arr,(int(imgs_arr.size/400), 400))
  return imgs_vect

def LoadData():
  print('Loading Training Data...')
  train_imgs = LoadDir('galaxy/train')
  print(train_imgs.shape)

  print('Loading Validation Data...')
  val_imgs = LoadDir('galaxy/val')
  print(val_imgs.shape)

  print('Loading Test Data...')
  test_imgs = LoadDir('galaxy/test')
  print(test_imgs.shape)

  return train_imgs, val_imgs, test_imgs