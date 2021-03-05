# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 10:44:36 2019
created 2 branch
@author: User
"""

import numpy as np
from matplotlib import pylab as plt

import cv2 as cv

from skimage.feature import register_translation
from scipy.ndimage import fourier_shift

amp_img = open('D:/OF IRG/TEST subj/TEST001-D/2018-05-10/TEST001-D_Angio (3mmx3mm)(1536x300x1203)_2018-05-10_06-22-14-PM_OS_amplitude.bin', "rb")
#phase_img = open('MIOP2878_Angio (3mmx3mm)(1536x300x1203)_2018-05-09_10-40-04-AM_OD_phase.bin', "rb")
phase_img = open('D:/OF IRG/TEST subj/TEST001-D/2018-05-10/TEST001-D_Angio (3mmx3mm)(1536x300x1203)_2018-05-10_06-22-14-PM_OS_phase.bin', "rb")

###############################################################################
"""
Reading image and storing it in array
"""

imA = np.zeros((1200,1536,300), dtype=np.uint8)
for i in range(1200):
    imH = amp_img.read(1536*300)
    im_hex2dec = np.frombuffer(imH, np.uint8)
    imA[i] = im_hex2dec.reshape(300,1536).T
    
imP = np.zeros((1200,1536,300), dtype=np.uint8)
for i in range(1200):
    imH = phase_img.read(1536*300)
    im_hex2dec = np.frombuffer(imH, np.uint8)
    imP[i] = im_hex2dec.reshape(300,1536).T


plt.figure()
plt.imshow(imA[1,:,:])

###############################################################################
"""
Averaging 4 repeatative images
"""
def avg_img(img_ssoct):
#    avgimg = np.zeros((300,1536,300), dtype=np.uint8)
    img_ssoct = img_ssoct.astype('double')
    avgimg = np.zeros((300,1536,300))
    for i in range(1200):
#        print(i)
        if ((i+1)%4) == 0:
            new_base = round((i)/4) - 1
#            print(i, i-1, i-2, i-3)
            print(new_base)
            avgimg[new_base] = ((img_ssoct[i-3] + img_ssoct[i-2] + img_ssoct[i-1] + img_ssoct[i])/4)
    return avgimg

avg_imA = avg_img(imA)
vessal_avg_img = np.max(avg_imA[:,600:800,], axis=1)
plt.figure()
plt.imshow(vessal_avg_img)
plt.title('OMAG Amplitude Map (Average)')
###############################################################################
"""
Efficient subpixel image registration by cross-correlation with subpixel precision
"""
#def image_registration(img_ssoct):
##    print(recons_image.shape)
#    shift_vals = []
#    for i in range(1200-1):
#        shift, error, diffphase = register_translation(img_ssoct[0,500:900,:], img_ssoct[i+1,500:900,:], 10)
#        print("Detected subpixel offset %d (y, x): {}".format(shift) %(i+1))
#        shift_vals.append(shift)
#    return shift_vals
def image_registration(img_ssoct):
#    print(recons_image.shape)
    shift_vals = []
    for i in range(1200-1):
        if ((i%4)==0):
            j = i
        shift, error, diffphase = register_translation(img_ssoct[j,500:900,:], img_ssoct[i+1,500:900,:], 10)
#        print(i,j)
        print("Detected subpixel offset %d (y, x): {}".format(shift) %(i+1))
        shift_vals.append(shift)
    return shift_vals

def image_translation(img_ssoct, shift):
    recons_image = np.zeros((1200,1536,300))
#    print(recons_image.shape)
    for i in range(1200-1):
        fft_img = fourier_shift(np.fft.fftn(img_ssoct[i+1]), shift[i])
        ifft_img = np.fft.ifftn(fft_img)
        recons_image[i+1] = ifft_img.real
        print("Readjusted Bscan frame %d" %(i+1))
    recons_image[0] = img_ssoct[0]
    return recons_image

#shiftA = image_registration(imA)
#corrA_img = image_translation(imA, shiftA)
#corrB_img = image_translation(imP, shiftA)
#corP_img,  = imageB

###############################################################################

"""
OMAG 
"""
def OMAG(img_ssoct):
    octaI = np.zeros((300,1536,300), dtype=np.uint8)
    img_ssoct = img_ssoct.astype('double')
    for i in range(1200):
        if ((i+1)%4) == 0:
            new_base = round((i+1)/4) - 1
#            print(i, i-1, i-2, i-3)
            print("OMAG base",new_base)
            octaI[new_base] = (abs(img_ssoct[i] - img_ssoct[i-1]) + abs(img_ssoct[i-1] - img_ssoct[i-2]) + abs(img_ssoct[i-2] - img_ssoct[i-3]))/3
    return octaI

#omag_amp = OMAG(corrA_img)

###############################################################################
"""
Adjacent A line difference
"""
imP_Adiff = np.zeros((1200, 1536,300), dtype='uint8')
for j in range(1200):
    for i in range(int(300-1)):
        print(i,j)
        imP_Adiff[j,:,i] = imP[j,:,i]-imP[j,:,i+1]
#        imP_Adiff[j,:,i] = corrB_img[j,:,i]-corrB_img[j,:,i+1]

###############################################################################
omag_phase = OMAG(imP_Adiff)

"""
Filterig
"""
def filter_speckle(img):
    dst = np.zeros((300,1536,300), dtype=np.uint8)
    for i in range(300):
#        dst[i] = cv.fastNlMeansDenoising(img[i,:,:],None,7,7,21)
        dst[i] = cv.GaussianBlur(img[i,:,:],(5,5),0)
        print("Filtering Image:",i)
    return dst

#omag_amp = filter_speckle(omag_amp)
#vessal_omag_amp = np.max(omag_amp[:,550:600,:], axis=1)
#plt.figure()
#plt.imshow(vessal_omag_amp)
#plt.title('OMAG Amplitude Map')
#plt.savefig("omag_amp_map_with_filter.pdf")
#omag_phase = OMAG(corrB_img)
vessal_omag_phase = np.mean(omag_phase[:,600:610,:], axis=1)
plt.figure()
plt.imshow(vessal_omag_phase)
plt.title('OMAG Phase Map')

###############################################################################
"""
Thresholding of images

from skimage import data
from skimage.filters import try_all_threshold

img = omag_phase[2,450:800,:]

fig, ax = try_all_threshold(img, figsize=(10, 8), verbose=False)
plt.show()
"""
###############################################################################
"""
Animation of B scans
"""
import matplotlib.animation as animation

fig = plt.figure()
ax = fig.add_axes([0.05,0.05,0.9,0.9])
ims = []
for i in range(300):
    ttl = plt.text(0.5, 0.95, "Frame {0}".format(i), horizontalalignment='center', 
                   verticalalignment='top',transform=ax.transAxes, color='red')
#    im = plt.imshow(corrB_img[i,400:900,:], animated=True)
    im = plt.imshow(omag_phase[i,400:800,:], animated=True)
#    plt.savefig('corr_img%d.png'%(i+1))
#    im = plt.imshow(avgIM[i,600:900,:], animated=True)
    ims.append([im, ttl])
    ttl = []
 #   ims.append([im])

ani = animation.ArtistAnimation(fig, ims, interval=100, blit=True,
                                repeat_delay=3000)
#ani.save('fig_amp.gif', writer='imagemagick')
#ani.save('sample.mp4', writer='ffmpeg')
#from matplotlib.animation import FFMpegWriter
#writer = FFMpegWriter(fps=15, metadata=dict(artist='Me'), bitrate=1800)
#ani.save("movie.mp4", writer=writer)
plt.show()
