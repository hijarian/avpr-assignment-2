import os

current_dir = os.path.dirname(__file__)

annotations_path = os.path.join(current_dir, 'Stanford40/XMLAnnotations')
images_path = os.path.join(current_dir, 'Stanford40/JPEGImages')

given_splits_train = os.path.join(current_dir, 'data/Stanford40/given_splits/train')
given_splits_test = os.path.join(current_dir, 'data/Stanford40/given_splits/test')
random_splits_train = os.path.join(current_dir, 'data/Stanford40/random_splits/train')
random_splits_test = os.path.join(current_dir, 'data/Stanford40/random_splits/test')

models = os.path.join(current_dir, 'models')