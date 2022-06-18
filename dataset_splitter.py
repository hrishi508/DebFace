import os
import csv
import glob
import shutil
import argparse

from utils.utils_config import get_config
from utils.utils_config import ConfigParams

class IMFDB_Splitter():
    def __init__(self, cfg):
        print("Initializing the splitter..")
        self.datasets_base_path = cfg.datasets_base_path
        self.info_file = open((self.datasets_base_path + "dataset_split_info.txt"), "w")

        self.train_path = self.datasets_base_path + "Train/"
        self.train_img_path = self.train_path + "images/"
        self.train_label_path = self.train_path + "labels.csv"
        self.train_img_cnt = 0
        self.train_id_cnt = 0

        self.test_path = self.datasets_base_path + "Test/"
        self.test_img_path = self.test_path + "images/"
        self.test_label_path = self.test_path + "labels.csv" 
        self.test_img_cnt = 0
        self.test_id_cnt = 0

        self.data_path = self.datasets_base_path + "IMFDB_final/"
        self.data_label_path = self.data_path + "labels.csv"
        self.img_cnt = 0

        self.img_postfix = "**/*.jpg"
        self.img_files = []

        # useful dictionaries
        self.test_subjects_id_dict = dict({2: 1, 8: 1, 23: 1, 62: 1, 44: 1, 35: 1, 37: 1})
        self.new_train_id_dict = dict()
        self.new_test_id_dict = dict()
        self.id_cnt_dict = dict()
        self.gender_cnt_dict = dict()
        self.gender_info_dict = dict()
        self.age_cnt_dict = dict()
        self.age_info_dict = dict()

        # label initializers
        self.id_index = 3
        self.gender_index = 1
        self.age_index = 2
        
        # temporary initializers
        self.tmp_id = None
        self.tmp_gender = None
        self.tmp_age = None
        self.tmp_label = None
        self.tmp_img_name = None
        self.tmp_img_path = None

        try:
            os.makedirs(self.train_img_path)
            os.makedirs(self.test_img_path)

        except:
            pass

    def split(self):
        print("Please wait while I split the dataset..")
        self.img_files = glob.glob(self.data_path + self.img_postfix, recursive=True)
        self.img_files.sort()

        with open(self.data_label_path, "r") as f1:
            r1 = csv.reader(f1)
            next(r1) # skip the header row

            with open(self.train_label_path, "w") as f2:
                with open(self.test_label_path, "w") as f3:
                    w1 = csv.writer(f2)
                    w2 = csv.writer(f3)

                    w1.writerow(["Image Name", "Gender Label", "Age Label", "ID Label"])
                    w2.writerow(["Image Name", "Gender Label", "Age Label", "ID Label"])

                    for label in r1:
                        self.tmp_gender = int(label[self.gender_index])
                        self.tmp_age = int(label[self.age_index])
                        self.tmp_id = int(label[self.id_index])

                        # Check if ID is in test subjects dict
                        if self.tmp_id in self.test_subjects_id_dict.keys():
                            if self.tmp_id not in self.new_test_id_dict.keys():
                                self.new_test_id_dict[self.tmp_id] = self.test_id_cnt
                                self.tmp_id = self.test_id_cnt
                                self.test_id_cnt += 1

                            else:
                                self.tmp_id = self.new_test_id_dict[self.tmp_id]

                            self.tmp_img_name = str(self.test_img_cnt).zfill(5) + ".jpg"
                            self.tmp_img_path = self.test_img_path + self.tmp_img_name 
                            self.tmp_label = [self.tmp_img_name, self.tmp_gender, self.tmp_age, self.tmp_id]

                            w2.writerow(self.tmp_label)
                            shutil.copyfile(self.img_files[self.img_cnt], self.tmp_img_path)

                            self.test_img_cnt += 1
                            self.img_cnt += 1

                        else:
                            if self.tmp_id not in self.new_train_id_dict.keys():
                                self.new_train_id_dict[self.tmp_id] = self.train_id_cnt
                                self.tmp_id = self.train_id_cnt
                                self.train_id_cnt += 1

                            else:
                                self.tmp_id = self.new_train_id_dict[self.tmp_id]

                            self.tmp_img_name = str(self.train_img_cnt).zfill(5) + ".jpg"
                            self.tmp_img_path = self.train_img_path + self.tmp_img_name 
                            self.tmp_label = [self.tmp_img_name, self.tmp_gender, self.tmp_age, self.tmp_id]

                            w1.writerow(self.tmp_label)
                            shutil.copyfile(self.img_files[self.img_cnt], self.tmp_img_path)

                            self.train_img_cnt += 1
                            self.img_cnt += 1
                f3.close()
            f2.close()
        f1.close()

        self.info_file.write(f"Training Set Info\n-----------------\n\n")
        self.info_file.write(f"Total no. of images: {self.train_img_cnt}\n")
        self.info_file.write(f"Total no. of subjects: {len(self.new_train_id_dict)}\n\n")
        self.info_file.write(f"{self.new_train_id_dict}\n\n")
        self.info_file.write(f"Test Set Info\n-------------\n\n")
        self.info_file.write(f"Total no. of images: {self.test_img_cnt}\n")
        self.info_file.write(f"Total no. of subjects: {len(self.new_test_id_dict)}\n\n")
        self.info_file.write(f"{self.new_test_id_dict}\n\n")
        
        print("Done!")

if __name__=="__main__":
    
    # get config
    parser = argparse.ArgumentParser(
        description="IMFDB Dataset Splitter")
    parser.add_argument("config", type=str, help="absolute path to the config file (config.ini)")
    args = parser.parse_args()
    str_type_cfg = get_config(args.config)
    cfg = ConfigParams(str_type_cfg)

    dataset_splitter = IMFDB_Splitter(cfg)
    dataset_splitter.split()

    print("IMFDB Dataset Split Successful!")
    

