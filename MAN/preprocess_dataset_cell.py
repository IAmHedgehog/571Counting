from scipy.io import loadmat
from PIL import Image
import numpy as np
import os
from glob import glob
import cv2
import argparse
import random
import pandas as pd


def cal_new_size(im_h, im_w, min_size, max_size):
    if im_h < im_w:
        if im_h < min_size:
            ratio = 1.0 * min_size / im_h
            im_h = min_size
            im_w = round(im_w*ratio)
        elif im_h > max_size:
            ratio = 1.0 * max_size / im_h
            im_h = max_size
            im_w = round(im_w*ratio)
        else:
            ratio = 1.0
    else:
        if im_w < min_size:
            ratio = 1.0 * min_size / im_w
            im_w = min_size
            im_h = round(im_h*ratio)
        elif im_w > max_size:
            ratio = 1.0 * max_size / im_w
            im_w = max_size
            im_h = round(im_h*ratio)
        else:
            ratio = 1.0
    return im_h, im_w, ratio


def find_dis(point):
    square = np.sum(point*points, axis=1)
    dis = np.sqrt(np.maximum(square[:, None] - 2*np.matmul(point, point.T) + square[None, :], 0.0))
    dis = np.mean(np.partition(dis, 3, axis=1)[:, 1:4], axis=1, keepdims=True)
    return dis

def generate_data(im_path):
    im = Image.open(im_path)
    im_w, im_h = im.size
    csv_path = im_path.replace('images', 'ground_truth').replace('.tiff', '.csv')
    df = pd.read_csv(csv_path)
    points = df.to_numpy().astype(np.float32)
    # points = loadmat(mat_path)['annPoints'].astype(np.float32)
    idx_mask = (points[:, 0] >= 0) * (points[:, 0] <= im_w) * (points[:, 1] >= 0) * (points[:, 1] <= im_h)
    points = points[idx_mask]
    im_h, im_w, rr = cal_new_size(im_h, im_w, min_size, max_size)
    im = np.array(im)
    if rr != 1.0:
        im = cv2.resize(np.array(im), (im_w, im_h), cv2.INTER_CUBIC)
        points = points * rr
    return Image.fromarray(im), points


def parse_args():
    parser = argparse.ArgumentParser(description='Test ')
    parser.add_argument('--origin-dir', default='IDCIAv2',
                        help='original data directory')
    parser.add_argument('--data-dir', default='cell_Train_Val_Test',
                        help='processed data directory')
    args = parser.parse_args()
    return args


def split_data(folder, num_fold):
    image_paths = glob(f"{folder}/images/*.tiff")
    print("----------------->", len(image_paths))
    random.shuffle(image_paths)
    val_num = len(image_paths) // num_fold
    # val_image_paths = image_paths[:val_num]
    # train_image_paths = image_paths[val_num:]
    # return train_image_paths, val_image_paths, val_image_paths
    data = []
    for i in range(num_fold):
        data.append(image_paths[val_num*i: val_num*(i+1)])
    return data


if __name__ == '__main__':
    args = parse_args()
    save_dir = args.data_dir
    min_size = 512
    max_size = 2048
    num_fold = 5
    img_folds = split_data(args.origin_dir, num_fold)
    for fold_idx in range(num_fold):
        # process fold fold_idx
        train_imgs = []
        for idx, fold in enumerate(img_folds):
            if idx != fold_idx:
                train_imgs.extend(fold)
        val_imgs = img_folds[fold_idx]
        test_imgs = val_imgs
        for phase in ['Train', 'Test']:
            if phase == 'Train':
                sub_phase_list = {'train': train_imgs, 'val': val_imgs}
                for sub_phase, sub_imgs in sub_phase_list.items():
                    sub_save_dir = os.path.join(f'{save_dir}_{fold_idx}', sub_phase)
                    if not os.path.exists(sub_save_dir):
                        os.makedirs(sub_save_dir)
                    for im_path in sub_imgs:
                        name = os.path.basename(im_path)
                        print(name)
                        im, points = generate_data(im_path)
                        if sub_phase == 'train':
                            dis = find_dis(points)
                            points = np.concatenate((points, dis), axis=1)
                        im_save_path = os.path.join(sub_save_dir, name).replace('tiff', 'jpg')
                        im.save(im_save_path)
                        gd_save_path = im_save_path.replace('jpg', 'npy')
                        np.save(gd_save_path, points)
            else:
                sub_save_dir = os.path.join(save_dir, 'test')
                if not os.path.exists(sub_save_dir):
                    os.makedirs(sub_save_dir)
            
                for im_path in test_imgs:
                    name = os.path.basename(im_path)
                    print(name)
                    im, points = generate_data(im_path)
                    im_save_path = os.path.join(sub_save_dir, name).replace('tiff', 'jpg')
                    im.save(im_save_path)
                    gd_save_path = im_save_path.replace('jpg', 'npy')
                    np.save(gd_save_path, points)
