import os
import csv
import glob
import shutil
import argparse

from utils.utils_config import get_config
from utils.utils_config import ConfigParams

class IMFDB_Organizer():
    def __init__(self, cfg):
        print("Initializing the organizer..")
        self.base_path = cfg.datasets_base_path
        self.data_path = self.base_path + "IMFDB/"
        self.target_path = self.base_path + "IMFDB_simplified/"
        self.img_path = self.target_path + "images/"
        self.label_path = self.target_path + "labels.csv" 
        self.img_postfix = "**/*.jpg"
        self.txt_postfix = "**/*.txt"
        self.img_files = []
        self.label_files = []
        self.img_cnt = 0
        self.err_file = open("organizer_errors.txt", "w")

        # class mapping dictionaries
        self.id_cnt = 0
        self.id_dict = dict()
        self.gender_dict = dict({
            "MALE": 0,
            "FEMALE": 1
        })
        self.age_dict = dict({
            "CHILD": 0,
            "YOUNG": 1,
            "MIDDLE": 2,
            "OLD": 3
        })
        
        # label initializers
        self.img_name_index = 2
        self.id_index = 9
        self.gender_index = 10
        self.age_index = 14
        
        # temporary initializers
        self.tmp_label = None
        self.tmp_img_name = None
        self.tmp_id = None
        self.tmp_gender = None
        self.tmp_age = None
        self.tmp_img_path = None

        try:
            os.makedirs(self.img_path)

        except:
            pass

    def organize(self):
        print("Please wait while I organize the dataset..")
        self.img_files = glob.glob(self.data_path + self.img_postfix, recursive=True)
        self.label_files = glob.glob(self.data_path + self.txt_postfix, recursive=True)

        self.img_files.sort()
        self.label_files.sort()

        with open(self.label_path, "w") as f1:
            w1 = csv.writer(f1)
            w1.writerow(["Image Name", "Gender Label", "Age Label", "ID Label"])
            
            for label_file in self.label_files:
                with open(label_file, "r") as f2:                        
                    r1 = csv.reader(f2, delimiter='\t')

                    for first_label in r1:
                        if first_label == [] or first_label == ['\ufeff']:
                            continue

                        try:
                            self.tmp_id = first_label[self.id_index].lower()

                        except:
                            print(self.id_index)
                            print(first_label)
                            quit()

                        if self.tmp_id not in self.id_dict.keys():
                            # print(self.tmp_id, self.id_cnt)
                            self.id_dict[self.tmp_id] = self.id_cnt
                            self.id_cnt += 1
                            self.tmp_id = None

                        break

                f2.close()
            
                with open(label_file, "r") as f2:
                    r1 = csv.reader(f2, delimiter='\t')

                    for label in r1:
                        if label == [] or label == ['\ufeff']:
                            continue

                        self.tmp_img_name = self.img_files[self.img_cnt].split('/')[-1]
                        
                        if self.tmp_img_name != label[self.img_name_index]:
                            # print(self.tmp_img_name, self.img_name_index)
                            # print(label)
                            self.err_file.write("Alert, image and label do not match!\n")
                            self.err_file.write(self.tmp_img_name + "\t" + label[self.img_name_index] + "\n")
                            self.err_file.write(", ".join(label) + "\n\n")
                            # print("Alert, image and label do not match! Terminating..")
                            continue
                        
                        # print(self.tmp_img_name, self.img_cnt, label[self.id_index].lower(), label[self.gender_index], label[self.age_index])
                        # print(self.id_dict)
                        
                        try:
                            try:
                                self.tmp_id = self.id_dict[label[self.id_index].lower()]

                            except:
                                self.img_cnt += 1
                                continue

                            self.tmp_gender = self.gender_dict[label[self.gender_index]]

                            if label[self.age_index] == "MIDDDLE":
                                self.tmp_age = self.age_dict["MIDDLE"]

                            else:
                                self.tmp_age = self.age_dict[label[self.age_index]]

                        except:
                            print("Key not found in dictionary!")
                            print(self.tmp_img_name, self.img_cnt, label[self.id_index].lower(), label[self.gender_index], label[self.age_index])
                            print(label)
                            quit()

                        self.tmp_img_name = str(self.img_cnt).zfill(5) + ".jpg"
                        self.tmp_img_path = self.img_path + self.tmp_img_name 
                        self.tmp_label = [self.tmp_img_name, self.tmp_gender, self.tmp_age, self.tmp_id]

                        w1.writerow(self.tmp_label)
                        shutil.copyfile(self.img_files[self.img_cnt], (self.img_path + self.tmp_img_name))

                        self.img_cnt += 1

                f2.close()        
        f1.close()
        
        print("Done!")
        print(f"Total no. of images in the reorganized dataset: {self.img_cnt}")
        print(f"Total no. of subjects in the reorganized dataset: {len(self.id_dict)}")
        # print(self.id_dict)

        # ID dictionary debug code
        a = [0 for i in range(len(self.id_dict))]
        for key in self.id_dict:
            a[self.id_dict[key]] = key

        for i in range(len(a)):
            if a[i] == 0:
                print(f"Alert! {i}\n")

if __name__=="__main__":
    
    # get config
    parser = argparse.ArgumentParser(
        description="Dataset Organizer")
    parser.add_argument("config", type=str, help="absolute path to the config file (config.ini)")
    args = parser.parse_args()
    str_type_cfg = get_config(args.config)
    cfg = ConfigParams(str_type_cfg)

    organizer = IMFDB_Organizer(cfg)
    organizer.organize()

    print("IMFDB Dataset Organized successfully!")
    

