import os


def log_error(save_path, text, **kwargs):
    path = os.path.join(save_path, "log.txt")
    with open(path, 'a') as f:
        f.write(", ".join(f"{key}={value}" for key, value in kwargs.items()) + "\n")
        if text is not None:
            f.write("\n")
            f.write(text + "\n")