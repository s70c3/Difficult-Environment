
from compose import compose
from utils.weatheraug import *
import random
import cv2
def random_aug(img, n=1):

    all_funcs = [brighten,  darken, rain, fog, smoke, noise, sun]
    funcs = random.choices(all_funcs, k=n)
    result = list(map(compose(*funcs), img))
    return result

def apply_func(img, funcs):
    result = list(map(compose(*funcs), img))
    return result

def apply_to_folder(base_dir, target_dir, n=2):
    import os
    d = os.listdir(base_dir)
    for filename in d:
        img = cv2.imread(os.path.join(base_dir, filename))
        img_aug= random_aug([img], n=n)
        cv2.imwrite(os.path.join(target_dir, filename), img_aug)
def apply_to_stereo(base_dir, target_dir, n=2):
    import os
    d = os.listdir(os.path.join(base_dir, 'left'))
    for filename in d:
        img_l = cv2.imread(os.path.join(base_dir, 'left', filename))
        img_r = cv2.imread(os.path.join(base_dir, 'right', filename))
        img_l_aug, img_r_aug = random_aug([img_l, img_r], n=n)
        cv2.imwrite(os.path.join(target_dir, 'left', filename), img_l_aug)
        cv2.imwrite(os.path.join(target_dir, 'right', filename), img_r_aug)