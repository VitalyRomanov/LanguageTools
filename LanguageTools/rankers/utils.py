import os


def check_dir_exists(path):
    if not os.path.isdir(path):
        os.mkdir(path)
