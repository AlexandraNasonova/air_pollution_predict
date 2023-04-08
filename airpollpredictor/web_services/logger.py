# pylint: disable=E0401, R0913, R0914, W0703

"""
Simple custom local logger
"""
import os


def log_error(save_path, text, **kwargs) -> None:
    """
    Log errors to the log.txt file
    @param save_path: Path to the log.txt file
    @param text: Error message
    @param kwargs: Parameters and their values for saving
    """
    path = os.path.join(save_path, "log.txt")
    with open(path, 'a', encoding="utf8") as file_stream:
        file_stream.write(", ".join(f"{key}={value}" for key, value in kwargs.items()) + "\n")
        if text is not None:
            file_stream.write("\n")
            file_stream.write(text + "\n")
