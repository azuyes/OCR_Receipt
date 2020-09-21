import os
from ocr import ocr
import time
import shutil
import numpy as np
from PIL import Image
from glob import glob
from xml.etree import ElementTree as ET
import os


def single_pic_proc(image_file):
    image = np.array(Image.open(image_file).convert('RGB'))
    result, image_framed = ocr(image)
    return result,image_framed


if __name__ == '__main__':
    image_files = glob('./test_result_for_PICK/test_images/*.*')
    result_dir_ROOT = './test_result_for_PICK'
    boxes_and_transcripts_path = "./test_result_for_PICK/boxes_and_transcripts"
    train_samples_list_path = os.path.join(result_dir_ROOT, 'train_samples_list.csv')
    train_samples_list = open(train_samples_list_path, 'w')
    for index in range(len(sorted(image_files))):
        train_samples_list.write(str(index)+",train,"+image_files[index].split('/')[-1]+"\n")
        result, image_framed = single_pic_proc(image_files[index])
        boxs_and_transcripts = open(boxes_and_transcripts_path+"/"+image_files[index].split('/')[-1].replace(".jpg", ".tsv"), 'w')
        for idx in range(len(result)):
            array = result[idx][0]
            text = result[idx][1]
            box_str = ""
            for i in range(8):
                box_str += ","+str(array[i])
            newLine = str(index) + box_str + "," + text+"\n"
            boxs_and_transcripts.write(newLine)

    boxs_and_transcripts.close()
    train_samples_list.close()