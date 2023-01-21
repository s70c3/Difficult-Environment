import cv2
import numpy as np
import random

def add_brightness(image):
    image_HLS = cv2.cvtColor(image,cv2.COLOR_RGB2HLS)
    ## Conversion to HLS
    image_HLS = np.array(image_HLS, dtype = np.float64)
    random_brightness_coefficient = np.random.uniform()+0.5
    ## generates value between 0.5 and 1.5
    image_HLS[:,:,1] = image_HLS[:,:,1]*random_brightness_coefficient
    ## scale pixel values up or down for channel 1(Lightness)
    image_HLS[:,:,1][image_HLS[:,:,1]>255]  = 255
    ##Sets all values above 255 to 255
    image_HLS = np.array(image_HLS, dtype = np.uint8)
    image_RGB = cv2.cvtColor(image_HLS,cv2.COLOR_HLS2RGB)
    ## Conversion to RGB
    return image_RGB

def generate_shadow_coordinates(imshape, no_of_shadows=1):
    vertices_list=[]
    for index in range(no_of_shadows):
        vertex=[]
        for dimensions in range(np.random.randint(3,15)):
            ## Dimensionality of the shadow polygon
            vertex.append(( imshape[1]*np.random.uniform(),imshape[0]//3+imshape[0]*np.random.uniform()))
        vertices = np.array([vertex], dtype=np.int32)
        ## single shadow vertices
        vertices_list.append(vertices)
    return vertices_list ## List of shadow vertices

def add_shadow(image,no_of_shadows=1):
    image_HLS = cv2.cvtColor(image,cv2.COLOR_RGB2HLS)
    ## Conversion to HLS
    mask = np.zeros_like(image)
    imshape = image.shape
    vertices_list= generate_shadow_coordinates(imshape, no_of_shadows)
    #3 getting list of shadow vertices
    for vertices in vertices_list:
        cv2.fillPoly(mask, vertices, 255) ## adding all shadow polygons on empty mask, single 255 denotes only red channel
        image_HLS[:,:,1][mask[:,:,0]==255] = image_HLS[:,:,1][mask[:,:,0]==255]*0.5   ## if red channel is hot, image's "Lightness" channel's brightness is lowered
    image_RGB = cv2.cvtColor(image_HLS,cv2.COLOR_HLS2RGB)
    ## Conversion to RGB
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

def generate_random_lines(imshape,slant,drop_length):
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
    drop_length=20
    drop_width=2
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

def generate_random_blur_coordinates(imshape,hw):
    blur_points=[]
    midx= imshape[1]//2-2*hw
    midy= imshape[0]//2-hw
    index=1
    while(midx>-hw and midy>-hw):
        for i in range(hw//10*index):
            x= np.random.randint(midx,imshape[1]-midx-hw)
            y= np.random.randint(midy,imshape[0]-midy-hw)
            blur_points.append((x,y))
        midx-=3*hw*imshape[1]//sum(imshape)
        midy-=3*hw*imshape[0]//sum(imshape)
        index+=1
    return blur_points

def add_blur(image, x,y, hw, fog_coeff):
    overlay= image.copy()
    output= image.copy()
    alpha= 0.08*fog_coeff
    rad= hw//2
    point=(x+hw//2, y+hw//2)
    cv2.circle(overlay,point, int(rad), (255,255,255), -1)
    cv2.addWeighted(overlay, alpha, output, 1 -alpha ,0, output)
    return output


def add_fog(image, coeff=-1):
    if (coeff < 0.0 or coeff > 1.0) and coeff!=-1:
        raise Exception("Fog strength coefficient should be between 0 and 1. You can use -1 for random coefficient.")
    if coeff==-1:
        coeff=random.uniform(0.1,0.5)

    imshape = image.shape
    hw=int(imshape[1]//3*coeff)
    haze_list=generate_random_blur_coordinates(imshape,hw)
    for haze_points in haze_list:
        image= add_blur(image, haze_points[0], haze_points[1], hw, coeff)
    image = cv2.blur(image, (hw//10,hw//10))
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
def flare_source(image, point,radius,src_color):
    overlay= image.copy()
    output= image.copy()
    num_times=radius//10
    alpha= np.linspace(0.0,1,num= num_times)
    rad= np.linspace(1,radius, num=num_times)
    for i in range(num_times):
        cv2.circle(overlay,point, int(rad[i]), src_color, -1)
        alp=alpha[num_times-i-1]*alpha[num_times-i-1]*alpha[num_times-i-1]
        cv2.addWeighted(overlay, alp, output, 1 -alp ,0, output)
    return output

def add_sun_flare_line(flare_center,angle, w):
    x=[]
    y=[]
    i=0
    for rand_x in range(0,w,10):
        rand_y= math.tan(angle)*(rand_x-flare_center[0])+flare_center[1]
        x.append(rand_x)
        y.append(2*flare_center[1]-rand_y)
    return x,y

def add_sun_process(image, no_of_flare_circles,flare_center,src_radius,x,y,src_color):
    overlay= image.copy()
    output= image.copy()
    imshape=image.shape
    for i in range(no_of_flare_circles):
        alpha=random.uniform(0.05,0.2)
        r=random.randint(0, len(x)-1)
        rad=random.randint(1, imshape[0]//100-2)
        cv2.circle(overlay,(int(x[r]),int(y[r])), rad*rad*rad, (random.randint(max(src_color[0]-50,0), src_color[0]),random.randint(max(src_color[1]-50,0), src_color[1]),random.randint(max(src_color[2]-50,0), src_color[2])), -1)
        cv2.addWeighted(overlay, alpha, output, 1 - alpha,0, output)
    output= flare_source(output,(int(flare_center[0]),int(flare_center[1])),src_radius,src_color)
    return output
def add_sun_flare(image,flare_center=None, angle=None, no_of_flare_circles=8,src_radius=400, src_color=(255,255,255)):
    if angle:
        angle=angle%(2*math.pi)
    if not(0 <= no_of_flare_circles <= 20):
        raise Exception(err_flare_circle_count)

    h, w=image.shape[:2]
    if angle is None:
        angle=random.uniform(0,2*math.pi)
        if angle==math.pi/2:
            angle=0

    if flare_center is None:
        flare_center=(random.randint(0,w),random.randint(0,h//2))

    x,y= add_sun_flare_line(flare_center,angle, w)
    output= add_sun_process(image, no_of_flare_circles,flare_center,src_radius,x,y,src_color)
    image_RGB = output
    return image_RGB



def fill(img, h, w):
    img = cv2.resize(img, (h, w), cv2.INTER_CUBIC)
    return img


def horizontal_shift(img, ratio=0.0):
    if ratio > 1 or ratio < 0:
        print('Value should be less than 1 and greater than 0')
        return img
    ratio = random.uniform(-ratio, ratio)
    h, w = img.shape[:2]
    to_shift = w * ratio
    if ratio > 0:
        img = img[:, :int(w - to_shift), :]
    if ratio < 0:
        img = img[:, int(-1 * to_shift):, :]
    img = fill(img, h, w)
    return img

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