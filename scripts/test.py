# import pandas as pd
# import numpy as np
# import PIL
# import matplotlib.pyplot as plt
# from  PIL import Image
# from get_albumentation import get_train_transform
# import cv2
#
# for i in range(10):
#     plt.figure(12)
#     img = cv2.imread('../dataset/train/1.jpg', 1)
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     # print(img)
#
#     plt.subplot(1, 2, 1)
#     plt.imshow(img)
#     plt.subplot(1, 2, 2)
#     transform = get_train_transform()
#     img = np.array(img)
#     img = transform(image=img)['image']
#     plt.imshow(img)
#     plt.show()
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
pw = Image.open('../dataset/train/1.jpg').convert('RGB')
transform = transforms.RandomHorizontalFlip(p=0.7)
plt.figure(12)
plt.subplot(121)
plt.imshow(pw)
plt.subplot(122)
pw = transform(pw)
plt.imshow(pw)
plt.show()