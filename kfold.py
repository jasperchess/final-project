from sklearn.model_selection import StratifiedKFold, KFold
import pandas as pd
import numpy as np
import os
import shutil
import uuid
import cv2
import keras

def augment(img):
    augment = keras.Sequential([
        keras.layers.RandomFlip(),
        keras.layers.RandomRotation(0.2)
    ])

    return augment(img)
"""
    This utitlity function copies the images from the raw
    directories into their respective train and test folds.
    This is used in conjuction with the `fold_dataset` method
    below, inorder to evaluate models using K-Fold.
"""
def copy_files_to_fold(train_fold, test_fold, augment_train=None, augment_test=None, split_dir='./data/folds'):
    train_split = 'train'
    test_split = 'test'
    splits = [train_split, test_split]
    normal_class = 'normal'
    tb_class = 'tuberculosis'
    classes = [normal_class, tb_class]

    for i, _ in enumerate(train_fold):
        if augment_train[i] == True:
            split = train_fold[i].split(".")
            split[0] += '_AUGMENT'
            train_fold[i] = '.'.join(split)

    for i, _ in enumerate(test_fold):
        if augment_test[i] == True:
            split = test_fold[i].split(".")
            split[0] += '_AUGMENT'
            test_fold[i] = '.'.join(split)

    if os.path.exists(split_dir):
        print("Removing existing dataset")
        # If the split directory exists, remove it and so it can be regenerated
        shutil.rmtree(split_dir)

    for split in splits:
        for cls in classes:
            os.makedirs(os.path.join(split_dir, split, cls), exist_ok=True)

    def get_filters(fold):
        normal_filter = []
        tb_filter = []
        for filepath in fold:
            norm = "Normal" in filepath
            normal_filter.append(norm)
            tb_filter.append(not norm)

        return np.array(normal_filter), np.array(tb_filter)

    base_normal_filter, base_tb_filter = get_filters(train_fold)
    train_normal = train_fold[base_normal_filter]
    train_tb = train_fold[base_tb_filter]

    test_normal_filter, test_tb_filter = get_filters(test_fold)
    test_normal = test_fold[test_normal_filter]
    test_tb = test_fold[test_tb_filter]

    # Function to copy files to their respective directories
    def copy_files(file_list, dst_dir):
        for file in file_list:
            filename = uuid.uuid4().hex[:6].upper() + '.png'
            dest = os.path.join(dst_dir, filename)

            if '_AUGMENT' in file:
                tmp = file.replace('_AUGMENT', '')
                img = cv2.imread(tmp)
                img = augment(img)
                cv2.imwrite(dest, img.numpy())
            else:
                shutil.copy(file, dest)

    print("Copying train normal set...")
    copy_files(train_normal, os.path.join(split_dir, train_split, normal_class))
    print("Copying test normal set...")
    copy_files(test_normal, os.path.join(split_dir, test_split, normal_class))

    print("Copying train TB set...")
    copy_files(train_tb, os.path.join(split_dir, train_split, tb_class))
    print("Copying test TB set...")
    copy_files(test_tb, os.path.join(split_dir, test_split, tb_class))


"""
    This method uses the Scikit Learn Stratified K-Fold class
    to split the dataset into 5 folds. Stratified K-Fold is an
    adapted version of K-Fold which ensures that the class imbalance
    is represented in each fold. In this case 4:1 Normal:Tuberculosis

    In order to correctly split out the files this method returns the
    K-Fold split results, as well as the X and y arrays (images and labels)
"""
def fold_dataset(normal_dir='./data/Normal/', tb_dir='./data/Tuberculosis/', undersampling=False, oversampling=False):
    tb_metadata = pd.read_excel('./data/Tuberculosis.metadata.xlsx')
    norm_metadata = pd.read_excel('./data/Normal.metadata.xlsx')

    if undersampling == True:
        norm_metadata = norm_metadata[:len(tb_metadata)]

    if oversampling == True:
        tb_metadata = pd.concat([tb_metadata] * round(len(norm_metadata) / len(tb_metadata)), ignore_index=True)

    # print(len(tb_metadata))
    tb_metadata['tuberculosis'] = 1
    norm_metadata['tuberculosis'] = 0
    tb_metadata['FILE NAME'] = tb_dir + tb_metadata['FILE NAME']
    norm_metadata['FILE NAME'] = normal_dir + norm_metadata['FILE NAME']

    metadata = pd.concat([tb_metadata, norm_metadata])
    metadata.rename(columns={'FILE NAME': 'filename'}, inplace=True)
    metadata['filename'] = metadata['filename'] + '.png'
    metadata.drop(['FORMAT', 'SIZE'], axis=1, inplace=True)

    y = metadata['tuberculosis'].to_numpy()
    X = metadata['filename'].to_numpy()

    skf = StratifiedKFold(n_splits=5, shuffle=True)
    skf.get_n_splits(X, y)

    return skf.split(X, y), X, y

SEGMENTATION_FOLDS = './segmentation_data/folds'
def fold_segmentation_dataset(images_dir='./segmentation_data/images', masks_dir='./segmentation_data/masks'):
    X = np.array([f for f in os.listdir(images_dir) if os.path.isfile(os.path.join(images_dir, f))])
    y = np.array([f for f in os.listdir(masks_dir) if os.path.isfile(os.path.join(masks_dir, f))])

    kf = KFold(n_splits=5, shuffle=True)
    kf.get_n_splits(X)

    return kf.split(X), X, y

def copy_segmentation_dataset_to_folds(train_data, test_data, split_dir=SEGMENTATION_FOLDS, images_dir='./segmentation_data/images', masks_dir='./segmentation_data/masks'):
    if os.path.exists(split_dir):
        shutil.rmtree(split_dir)

    train_images_dir = os.path.join(split_dir, 'train', 'images')
    train_masks_dir = os.path.join(split_dir, 'train', 'masks')
    test_images_dir = os.path.join(split_dir, 'test', 'images')
    test_masks_dir = os.path.join(split_dir, 'test', 'masks')

    print("Creating fold dirs...")
    for dir in [train_images_dir, train_masks_dir, test_images_dir, test_masks_dir]:
        os.makedirs(dir)

    print("Copying the training data...")
    # Copying the training data into the fold
    for (image_name, mask_name) in list(zip(train_data['images'], train_data['masks'])):
        shutil.copy(os.path.join(images_dir, image_name), os.path.join(train_images_dir, image_name))
        shutil.copy(os.path.join(masks_dir, mask_name), os.path.join(train_masks_dir, mask_name))

    print("Copying the test data...")
    # Copying the training data into the fold
    for (image_name, mask_name) in list(zip(test_data['images'], test_data['masks'])):
        shutil.copy(os.path.join(images_dir, image_name), os.path.join(test_images_dir, image_name))
        shutil.copy(os.path.join(masks_dir, mask_name), os.path.join(test_masks_dir, mask_name))
