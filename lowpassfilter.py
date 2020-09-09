import cv2
import numpy as np
from matplotlib import pyplot as plt
#https://blog.csdn.net/wumu720123/article/details/89930745

#读取图像
img = cv2.imread("lena.png", cv2.IMREAD_GRAYSCALE)

#傅里叶变换
dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
dftShift = np.fft.fftshift(dft)

#设置高通滤波器
rows,cols = img.shape
crow,ccol = int(rows/2), int(cols/2)
mask = np.zeros((rows, cols, 2), np.uint8)
mask[crow-30:crow+30, ccol-30:ccol+30] = 1

#傅里叶逆变换
fshift = dftShift * mask
ishift = np.fft.ifftshift(fshift)
iimg = cv2.idft(ishift)
iimg = cv2.magnitude(iimg[:,:,0], iimg[:,:,1])

#显示原始图像和高通滤波处理图像
plt.subplot(121), plt.imshow(img, 'gray'), plt.title('Original Image')
plt.axis('off')
plt.subplot(122), plt.imshow(iimg, 'gray'), plt.title('Result low Image')
plt.axis('off')
plt.show()