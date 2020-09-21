import os
from ocr import ocr
import time
import shutil
import numpy as np
from PIL import Image
from glob import glob


def single_pic_proc(image_file):
    image = np.array(Image.open(image_file).convert('RGB'))
    result, image_framed = ocr(image)
    return result,image_framed


if __name__ == '__main__':
    image_files = glob('./test_images/*.*')
    result_dir = './test_result'
    train_samples_list_path = os.path.join(result_dir, 'train_samples_list.csv')
    train_samples_list = open(train_samples_list_path, 'w')
    for index,image_file in sorted(image_files):
        train_samples_list.write(index+",train,"+image_file)
    train_samples_list.close()