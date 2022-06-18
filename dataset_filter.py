import os
import csv
import glob
import shutil
import argparse

from utils.utils_config import get_config
from utils.utils_config import ConfigParams

class IMFDB_Filter():
    def __init__(self, cfg):
        print("Initializing the filter..")
        self.base_path = cfg.datasets_base_path
        self.errors_path = self.base_path + "errors/"
        self.data_path = self.base_path + "IMFDB_cleaned/"
        self.target_path = self.base_path + "IMFDB_final/"
        self.img_path = self.target_path + "images/"
        self.read_label_path = self.data_path + "labels.csv" 
        self.write_label_path = self.target_path + "labels.csv" 
        self.img_postfix = "**/*.jpg"
        self.img_files = []
        self.img_cnt = 0
        self.curr_idx = 0

        # No. of images per subject filter
        self.num_img_filter = cfg.num_img_filter

        # id mapping cleaning dictionary
        self.id_cnt = 0
        self.num_img_per_subj_dict = dict({0: 82, 1: 337, 2: 54, 3: 217, 4: 90, 5: 13, 6: 94, 7: 221, 8: 231, 9: 10, 10: 53, 11: 1, 12: 28, 13: 108, 14: 209, 15: 88, 16: 126, 17: 8, 18: 63, 19: 189, 20: 244, 21: 76, 22: 346, 23: 163, 24: 119, 25: 79, 26: 149, 27: 177, 28: 53, 29: 179, 30: 144, 31: 158, 32: 137, 33: 121, 34: 6, 35: 37, 36: 26, 37: 165, 38: 103, 39: 101, 40: 65, 41: 371, 42: 129, 43: 55, 44: 48, 45: 201, 46: 35, 47: 62, 48: 74, 49: 199, 50: 35, 51: 73, 52: 144, 53: 204, 54: 23, 55: 2, 56: 76, 57: 52, 58: 34, 59: 33, 60: 63, 61: 143, 62: 266, 63: 14, 64: 90, 65: 162, 66: 1, 67: 28, 68: 89, 69: 190, 70: 214, 71: 5, 72: 199, 73: 11, 74: 6, 75: 29, 76: 148, 77: 67, 78: 131, 79: 20, 80: 39, 81: 28, 82: 121, 83: 144, 84: 89, 85: 9, 86: 173, 87: 183, 88: 51})
        self.id_clean_dict = dict()
        self.remove_id_dict = dict()
        self.new_id_dict = dict()

        # initialize remove id dictionary
        for key in self.num_img_per_subj_dict:
            if self.num_img_per_subj_dict[key] < self.num_img_filter:
                self.remove_id_dict[key] = self.num_img_per_subj_dict[key]       
        
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
            os.mkdir(self.errors_path)

        except:
            pass

        self.err_file = open((self.errors_path + "filter_errors.txt"), "w")

    def filter(self):
        print("Please wait while I filter the dataset..")
        self.img_files = glob.glob(self.data_path + self.img_postfix, recursive=True)
        self.img_files.sort()

        with open(self.read_label_path, "r") as f1:
            with open(self.write_label_path, "w") as f2:
                w1 = csv.writer(f2)
                r1 = csv.reader(f1)

                w1.writerow(["Image Name", "Gender Label", "Age Label", "ID Label"])
                next(r1) # skip the header row

                for label in r1:
                                        
                    self.tmp_id = int(label[self.id_index])

                    # Handle ranimukerji gender mislabelling issue
                    if self.tmp_id == 61:
                        if label[self.gender_index] == '0':
                            self.curr_idx += 1
                            continue    

                    # Handle no. of images per subject issue
                    if self.tmp_id in self.remove_id_dict.keys():
                        self.curr_idx += 1
                        continue

                    if self.tmp_id in self.id_clean_dict.keys():
                        self.tmp_id = self.id_clean_dict[self.tmp_id]
                        self.new_id_dict[self.tmp_id] += 1

                    elif self.tmp_id not in self.new_id_dict.keys():
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
                        # print(self.tmp_img_name, self.img_name_index)
                        # print(label)
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

            f1.close()
        f2.close()

        print("Done!")
        print(f"Total no. of images in the filtered dataset: {self.img_cnt}")
        print(f"Minimum no. of images per subject in the filtered dataset: {self.num_img_filter}")
        print(f"Total no. of subjects in the filtered dataset: {len(self.new_id_dict)}")
        # print(self.new_id_dict)

if __name__=="__main__":
    
    # get config
    parser = argparse.ArgumentParser(
        description="Dataset Filter")
    parser.add_argument("config", type=str, help="absolute path to the config file (config.ini)")
    args = parser.parse_args()
    str_type_cfg = get_config(args.config)
    cfg = ConfigParams(str_type_cfg)

    filter = IMFDB_Filter(cfg)
    filter.filter()

    print("IMFDB Dataset Filtered successfully!")
    

