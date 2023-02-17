
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
def apply_to_stereo(base_dir, target_dir, n=2, resize_size = None):
    import os
    d = os.listdir(os.path.join(base_dir, 'left'))
    for filename in d:
        img_l = cv2.imread(os.path.join(base_dir, 'left', filename))
        img_r = cv2.imread(os.path.join(base_dir, 'right', filename))
        if resize_size is not None:
            img_l = cv2.resize(img_l, resize_size)
            img_r = cv2.resize(img_r, resize_size)
        img_l_aug, img_r_aug = random_aug([img_l, img_r], n=n)
        cv2.imwrite(os.path.join(target_dir, 'left', filename), img_l_aug)
        cv2.imwrite(os.path.join(target_dir, 'right', filename), img_r_aug)


def random_combine(first_dir, second_dir, target_dir, var = 2):
    import os
    import shutil
    d = os.listdir(os.path.join(first_dir, 'left'))
    for filename in d:
        if random.randint(0, var)==0:
            shutil.copyfile(os.path.join(first_dir, 'left', filename), os.path.join(target_dir, 'left', filename))
            shutil.copyfile(os.path.join(first_dir, 'right', filename), os.path.join(target_dir, 'right', filename))
        else:
            shutil.copyfile(os.path.join(second_dir, 'left', filename), os.path.join(target_dir, 'left', filename))
            shutil.copyfile(os.path.join(second_dir, 'right', filename), os.path.join(target_dir, 'right', filename))

def regular_combine(first_dir, second_dir, target_dir, split = 0.5):
    import os
    import shutil
    d = os.listdir(os.path.join(first_dir, 'left'))
    for i in range(int(len(d)*split)):
        shutil.copyfile(os.path.join(first_dir, 'left', f'{i}.png'), os.path.join(target_dir, 'left', f'{i}.png'))
        shutil.copyfile(os.path.join(first_dir, 'right', f'{i}.png'), os.path.join(target_dir, 'right', f'{i}.png'))
    for i in range(int(len(d) * split), len(d)):
        shutil.copyfile(os.path.join(second_dir, 'left', f'{i}.png'), os.path.join(target_dir, 'left', f'{i}.png'))
        shutil.copyfile(os.path.join(second_dir, 'right', f'{i}.png'), os.path.join(target_dir, 'right', f'{i}.png'))
