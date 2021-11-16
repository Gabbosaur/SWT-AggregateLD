import os, shutil
IMAGE_FACES_PATH = 'images/faces'
for filename in os.listdir(IMAGE_FACES_PATH):
    file_path = os.path.join(IMAGE_FACES_PATH, filename)
    try:
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
    except Exception as e:
        print('Failed to delete %s. Reason: %s' % (file_path, e))