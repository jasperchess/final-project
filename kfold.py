from sklearn.model_selection import StratifiedKFold
import pandas as pd
import numpy as np
import os
import shutil
import uuid

def copy_files_to_fold(train_fold, test_fold, split_dir='./data/folds'):
    train_split = 'train'
    test_split = 'test'
    splits = [train_split, test_split]
    normal_class = 'normal'
    tb_class = 'tuberculosis'
    classes = [normal_class, tb_class]

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
            shutil.copy(file, os.path.join(dst_dir, filename))

    print("Copying train normal set...")
    copy_files(train_normal, os.path.join(split_dir, train_split, normal_class))
    print("Copying test normal set...")
    copy_files(test_normal, os.path.join(split_dir, test_split, normal_class))

    print("Copying train normal set...")
    copy_files(train_tb, os.path.join(split_dir, train_split, tb_class))
    print("Copying test normal set...")
    copy_files(test_tb, os.path.join(split_dir, test_split, tb_class))

def fold_dataset(normal_dir='./data/Normal/', tb_dir='./data/Tuberculosis/', oversampling=False, undersampling=False):
    tb_metadata = pd.read_excel('./data/Tuberculosis.metadata.xlsx')
    norm_metadata = pd.read_excel('./data/Normal.metadata.xlsx')

    tb_metadata['tuberculosis'] = 1
    norm_metadata['tuberculosis'] = 0
    tb_metadata['FILE NAME'] = tb_dir + tb_metadata['FILE NAME']
    norm_metadata['FILE NAME'] = normal_dir + norm_metadata['FILE NAME']

    if undersampling:
        norm_metadata = norm_metadata.head(len(tb_metadata))

    metadata = pd.concat([tb_metadata, norm_metadata])
    metadata.rename(columns={'FILE NAME': 'filename'}, inplace=True)
    metadata['filename'] = metadata['filename'] + '.png'
    metadata.drop(['FORMAT', 'SIZE'], axis=1, inplace=True)

    y = metadata['tuberculosis'].to_numpy()
    X = metadata['filename'].to_numpy()

    skf = StratifiedKFold(n_splits=5, shuffle=True)
    skf.get_n_splits(X, y)

    return skf.split(X, y), X, y