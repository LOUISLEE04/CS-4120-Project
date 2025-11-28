import os


def get_filepath(folder_name: str, file_name: str):
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # Build the path to the models folder
    path = os.path.join(root_dir, folder_name,file_name)
    return path

if __name__ == "__main__":
    print(get_filepath("models", "mlp(64,2)"))


