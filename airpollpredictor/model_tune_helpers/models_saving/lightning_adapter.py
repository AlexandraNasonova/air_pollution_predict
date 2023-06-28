import shutil, zipfile, os


def zip_best_version_folder(best_model_path: str, output_folder: str):
    name = best_model_path[:best_model_path.rindex('/checkpoints')]
    zip_name = os.path.join('output_folder', 'model.zip')

    with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zip_ref:
        for folder_name, subfolders, filenames in os.walk(name):
            for filename in filenames:
                file_path = os.path.join(folder_name, filename)
                zip_ref.write(file_path, arcname=os.path.relpath(file_path, name))
    zip_ref.close()


def unzip_best_version_folder(path_to_zip_file: str, extract_path: str):
    with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
        zip_ref.extractall(extract_path)


def copy_best_checkpoint(best_model_path, dist_path):
    #dist_path = "../../experiments_results/tft/6001/model.ckpt"
    shutil.copyfile(best_model_path, dist_path)
