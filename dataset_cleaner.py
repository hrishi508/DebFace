import os
import csv
import cv2
import glob
import shutil
import argparse

from utils.utils_config import get_config
from utils.utils_config import ConfigParams

class IMFDB_Cleaner():
    def __init__(self, cfg):
        print("Initializing the cleaner..")
        self.base_path = cfg.datasets_base_path
        self.data_path = self.base_path + "IMFDB_simplified/"
        self.target_path = self.base_path + "IMFDB_cleaned/"
        self.img_path = self.target_path + "images/"
        self.read_label_path = self.data_path + "labels.csv" 
        self.write_label_path = self.target_path + "labels.csv" 
        self.img_postfix = "**/*.jpg"
        self.img_files = []
        self.img_cnt = 0
        self.curr_idx = 0
        self.err_file = open("cleaner_errors.txt", "w")

        # Image width and height filter size
        self.height = cfg.height
        self.width = cfg.width

        # id mapping cleaning dictionary
        self.id_cnt = 0
        self.id_clean_dict = dict({10: 9,
                                   26: 24,
                                   29: 26,
                                   31: 27,
                                   34: 29,
                                   43: 37,
                                   51: 44,
                                   56: 48,
                                   72: 63,
                                   77: 67,
                                   79: 68,
                                   89: 77,
                                   90: 77,
                                   93: 79,
                                   95: 80
                                   })
        self.new_id_dict = dict()
        
        
        # label initializers
        self.img_name_index = 0
        self.gender_index = 1
        self.age_index = 2
        self.id_index = 3
        
        # temporary initializers
        self.tmp_id = None
        self.tmp_label = None
        self.tmp_img_name = None
        self.tmp_gender = None
        self.tmp_age = None
        self.tmp_img_path = None

        try:
            os.makedirs(self.img_path)

        except:
            pass

    def clean(self):
        print("Please wait while I clean the dataset..")
        self.img_files = glob.glob(self.data_path + self.img_postfix, recursive=True)
        self.img_files.sort()

        with open(self.read_label_path, "r") as f1:
            with open(self.write_label_path, "w") as f2:
                w1 = csv.writer(f2)
                r1 = csv.reader(f1)

                w1.writerow(["Image Name", "Gender Label", "Age Label", "ID Label"])
                next(r1) # skip the header row

                for label in r1:

                    # Handle image size issue
                    image = cv2.imread(self.img_files[self.curr_idx])
                    h, w, _ = image.shape
                    image = None

                    if h < self.height or w < self.width:
                        self.curr_idx += 1
                        continue

                    # Handle ID label issues
                    self.tmp_id = int(label[self.id_index])

                    if self.tmp_id in self.id_clean_dict.keys():
                        self.tmp_id = self.id_clean_dict[self.tmp_id]

                    if self.tmp_id not in self.new_id_dict.keys():
                        # print(self.tmp_id, self.id_cnt)
                        self.id_clean_dict[self.tmp_id] = self.id_cnt
                        self.tmp_id = self.id_cnt
                        self.new_id_dict[self.tmp_id] = 1
                        self.id_cnt += 1

                    else:
                        self.new_id_dict[self.tmp_id] += 1

                    # Perform sanity checks 
                    self.tmp_img_name = self.img_files[self.curr_idx].split('/')[-1]
                        
                    if self.tmp_img_name != label[self.img_name_index]:
                        self.err_file.write("Alert, image and label do not match!\n")
                        self.err_file.write(self.tmp_img_name + "\t" + label[self.img_name_index] + "\n")
                        self.err_file.write(", ".join(label) + "\n\n")
                        print("Alert, image and label do not match! Terminating..")
                        quit()

                    # Update cleaned image and label in new target directory
                    self.tmp_gender = int(label[self.gender_index])
                    self.tmp_age = int(label[self.age_index])

                    # print(self.tmp_img_name, self.img_cnt, self.tmp_id, self.tmp_gender, self.tmp_age)

                    self.tmp_img_name = str(self.img_cnt).zfill(5) + ".jpg"
                    self.tmp_img_path = self.img_path + self.tmp_img_name 
                    self.tmp_label = [self.tmp_img_name, self.tmp_gender, self.tmp_age, self.tmp_id]

                    w1.writerow(self.tmp_label)
                    shutil.copyfile(self.img_files[self.curr_idx], (self.img_path + self.tmp_img_name))

                    self.img_cnt += 1
                    self.curr_idx += 1

            f2.close()
        f1.close()

        print("Done!")
        print(f"Total no. of images in the cleaned dataset: {self.img_cnt}")
        print(f"Minimum dimension of each image in the cleaned dataset: {self.width} x {self.height}")
        print(f"Total no. of subjects in the cleaned dataset: {len(self.new_id_dict)}")
        # print(self.new_id_dict)

if __name__=="__main__":
    
    # get config
    parser = argparse.ArgumentParser(
        description="Dataset Cleaner")
    parser.add_argument("config", type=str, help="absolute path to the config file (config.ini)")
    args = parser.parse_args()
    str_type_cfg = get_config(args.config)
    cfg = ConfigParams(str_type_cfg)

    cleaner = IMFDB_Cleaner(cfg)
    cleaner.clean()

    print("IMFDB Dataset Cleaned successfully!")
    

