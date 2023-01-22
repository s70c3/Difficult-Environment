import cv2
import numpy as np
import random

def change_light(image, coeff):
    image_HLS = cv2.cvtColor(image,cv2.COLOR_RGB2HLS) ## Conversion to HLS
    image_HLS = np.array(image_HLS, dtype = np.float64)
    image_HLS[:,:,1] = image_HLS[:,:,1]*coeff ## scale pixel values up or down for channel 1(Lightness)
    if(coeff>1):
        image_HLS[:,:,1][image_HLS[:,:,1]>255]  = 255 ##Sets all values above 255 to 255
    image_HLS = np.array(image_HLS, dtype = np.uint8)
    image_RGB = cv2.cvtColor(image_HLS,cv2.COLOR_HLS2RGB) ## Conversion to RGB
    return image_RGB

def brighten(img, coeff = None):
    if coeff is None:
        coeff = random.uniform(0,1)
    if coeff < 0:
        raise Exception("Coefficient should be above zero. ")
    coeff += 1
    image_RGB = change_light(img, coeff)
    return image_RGB

def darken(img, coeff = None):
    if coeff is None:
        coeff = random.uniform(0, 1)
    if coeff<0:
        raise Exception("Coefficient should be above zero. ")
    coeff= 1-coeff
    image_RGB = change_light(img, coeff)
    return image_RGB

def add_snow(image):
    image_HLS = cv2.cvtColor(image,cv2.COLOR_RGB2HLS)
    ## Conversion to HLS
    image_HLS = np.array(image_HLS, dtype = np.float64)
    brightness_coefficient = 2.5
    snow_point=140 ## increase this for more snow
    image_HLS[:,:,1][image_HLS[:,:,1]<snow_point] = image_HLS[:,:,1][image_HLS[:,:,1]<snow_point]*brightness_coefficient ## scale pixel values up for channel 1(Lightness)
    image_HLS[:,:,1][image_HLS[:,:,1]>255]  = 255 ##Sets all values above 255 to 255
    image_HLS = np.array(image_HLS, dtype = np.uint8)
    image_RGB = cv2.cvtColor(image_HLS,cv2.COLOR_HLS2RGB) ## Conversion to RGB
    return image_RGB

def generate_random_lines(imshape, slant, drop_length):
    drops=[]
    for i in range(1500): ## If You want heavy rain, try increasing this
        if slant<0:
            x= np.random.randint(slant,imshape[1])
        else:
            x= np.random.randint(0,imshape[1]-slant)
        y= np.random.randint(0,imshape[0]-drop_length)
        drops.append((x,y))
    return drops

def add_rain(image):
    imshape = image.shape
    image = image.copy()
    slant_extreme=10
    slant= np.random.randint(-slant_extreme,slant_extreme)
    drop_length=10
    drop_width=1
    drop_color=(200,200,200) ## a shade of gray
    rain_drops= generate_random_lines(imshape,slant,drop_length)
    for rain_drop in rain_drops:
        cv2.line(image,(rain_drop[0],rain_drop[1]),(rain_drop[0]+slant,rain_drop[1]+drop_length),drop_color,drop_width)
    image= cv2.blur(image,(7,7)) ## rainy view are blurry
    brightness_coefficient = 0.7 ## rainy days are usually shady
    image_HLS = cv2.cvtColor(image,cv2.COLOR_RGB2HLS) ## Conversion to HLS
    image_HLS[:,:,1] = image_HLS[:,:,1]*brightness_coefficient ## scale pixel values down for channel 1(Lightness)
    image_RGB = cv2.cvtColor(image_HLS,cv2.COLOR_HLS2RGB) ## Conversion to RGB
    return image_RGB


def generate_random_blur_coordinates(imshape, fog_coeff):
    blur_points=[]
    h, w = imshape[:2]
    for i in range(int(250*fog_coeff)):
        x= np.random.randint(0, w)
        y= np.random.randint(0, h)
        blur_points.append((x,y))
    return blur_points

def add_blur(image, x,y, hw, fog_coeff, type = 'fog'):
    overlay= image.copy()
    output= image.copy()
    alpha= 0.08*fog_coeff
    rad= hw//2
    point=(x+hw//2, y+hw//2)
    if type == 'fog':
        color=(255, 255, 255)
    else:
        c = random.randint(20, 120)
        color = (c, c, c)
    cv2.circle(overlay, point, int(rad),  color, -1)
    cv2.addWeighted(overlay, alpha, output, 1-alpha ,0, output)
    return output


def add_fog(image, coeff=random.uniform(0.1, 0.8)):

    if (coeff < 0.0 or coeff > 1.0):
        raise Exception("Fog strength coefficient should be between 0 and 1.")

    imshape = image.shape
    hw=int(imshape[1]//3*coeff)
    haze_list=generate_random_blur_coordinates(imshape,coeff)
    for haze_points in haze_list:
        image= add_blur(image, haze_points[0], haze_points[1], hw, coeff, 'fog')
    image = brighten(image, 0.1)
    image = cv2.blur(image, (hw//20,hw//20))
    image_RGB = image

    return image_RGB

def add_smoke(image, coeff=random.uniform(0.1, 0.8)):

    if (coeff < 0.0 or coeff > 1.0):
        raise Exception("Fog strength coefficient should be between 0 and 1.")

    imshape = image.shape
    hw=int(imshape[1]//3*coeff)
    haze_list=generate_random_blur_coordinates(imshape,coeff)
    for haze_points in haze_list:
        image= add_blur(image, haze_points[0], haze_points[1], hw, coeff, 'smoke')
    image = brighten(image, 0.1)
    image = cv2.blur(image, (hw//20,hw//20))
    image_RGB = image

    return image_RGB


def noisy(image, noise_type='gaussian'):
    if noise_type not in ['gaussian', 'poisson', 's&p', 'speckle']:
        raise Exception('Noise type should be one of these: gaussian, poisson, s&p, speckle.')
    from skimage.util import random_noise
    noise_img = random_noise(image, mode=noise_type)

    return np.array(255*noise_img, dtype = 'uint8')

import math
err_flare_circle_count="Numeric value between 0 and 20 is allowed"
def add_sun_process(image, point, radius, src_color):
    overlay= image.copy()
    output= image.copy()
    num_times=radius//10
    alpha= np.linspace(0.0, 0.8,num= num_times)
    rad= np.linspace(1,radius, num=num_times)
    for i in range(num_times):
        cv2.circle(overlay, point, int(rad[i]), src_color, -1)
        alp=alpha[num_times-i-1]*alpha[num_times-i-1]*alpha[num_times-i-1]
        cv2.addWeighted(overlay, alp, output, 1-alp ,0, output)
    return output

def add_sun_flare(image, flare_center=None, angle=None, src_radius=400, src_color=(255,255,255)):
    if angle:
        angle=angle%(2*math.pi)

    h, w=image.shape[:2]
    if angle is None:
        angle=random.uniform(0,2*math.pi)
        if angle==math.pi/2:
            angle=0

    if flare_center is None:
        flare_center=(random.randint(0,w),random.randint(0,h//2))

    output=add_sun_process(image, flare_center, src_radius, src_color)
    image_RGB = output
    return image_RGB


def posture(im, n=8):
    indices = np.arange(0, 256)  # List of all colors
    divider = np.linspace(0, 255, n + 1)[1]  # we get a divider
    quantiz = np.int0(np.linspace(0, 255, n))  # we get quantization colors
    color_levels = np.clip(np.int0(indices / divider), 0, n - 1)  # color levels 0,1,2..
    palette = quantiz[color_levels]  # Creating the palette
    im2 = palette[im]  # Applying palette on image
    im2 = cv2.convertScaleAbs(im2)  # Converting image back to uint8
    return  im2

def add_weighted(im1, im2, alpha):
    beta = (1.0 - alpha)
    dst = cv2.addWeighted(im1, alpha, im2, beta, 0.0)
    return dst
