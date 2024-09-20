
import os

def create_directory_if_not_exists(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        print(f"Directory '{dir_path}' created successfully!")
    else:
        print(f"Directory '{dir_path}' already exists.")