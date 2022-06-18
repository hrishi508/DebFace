import csv
import argparse

from utils.utils_config import get_config
from utils.utils_config import ConfigParams

class IMFDB_Info():
    def __init__(self, cfg):
        print("Initializing..")
        self.datasets_base_path = cfg.datasets_base_path
        self.base_path = cfg.base_path
        self.data_path = self.datasets_base_path + "IMFDB_final/"
        self.target_path = self.datasets_base_path + "IMFDB_final_info.txt"
        self.label_path = self.data_path + "labels.csv" 
        self.img_cnt = 0
        self.info_file = open(self.target_path, "w")

        # class mapping dictionaries
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

    def info(self):
        print("Please wait while I extract information from the dataset..")

        with open(self.label_path, "r") as f1:
            r1 = csv.reader(f1)
            next(r1) # skip the header row

            for label in r1:
                self.tmp_gender = label[self.gender_index]
                self.tmp_age = label[self.age_index]
                self.tmp_id = label[self.id_index]

                # Gender Count
                if self.tmp_gender not in self.gender_cnt_dict.keys():
                    self.gender_cnt_dict[self.tmp_gender] = 1

                else:
                    self.gender_cnt_dict[self.tmp_gender] += 1
                
                # Gender Info
                if self.tmp_gender not in self.gender_info_dict.keys():
                    self.gender_info_dict[self.tmp_gender] = set()

                else:
                    self.gender_info_dict[self.tmp_gender].add(self.tmp_id)

                # Age Info
                if self.tmp_age not in self.age_info_dict.keys():
                    self.age_info_dict[self.tmp_age] = set()

                else:
                    self.age_info_dict[self.tmp_age].add(self.tmp_id)
                
                # Age Count
                if self.tmp_age not in self.age_cnt_dict.keys():
                    self.age_cnt_dict[self.tmp_age] = 1

                else:
                    self.age_cnt_dict[self.tmp_age] += 1
                
                # ID Count
                if self.tmp_id not in self.id_cnt_dict.keys():
                    self.id_cnt_dict[self.tmp_id] = 1

                else:
                    self.id_cnt_dict[self.tmp_id] += 1

                self.img_cnt += 1

        f1.close()

        self.info_file.write(f"Total no. of images: {self.img_cnt}\n")
        self.info_file.write(f"Total no. of subjects: {len(self.id_cnt_dict)}\n\n")
        self.info_file.write(f"No. of male subjects: {len(self.gender_info_dict['0'])}\n")
        self.info_file.write(f"No. of female subjects: {len(self.gender_info_dict['1'])}\n\n")
        self.info_file.write(f"No. of 'CHILD' subjects: {len(self.age_info_dict['0'])}\n")
        self.info_file.write(f"No. of 'YOUNG' subjects: {len(self.age_info_dict['1'])}\n")
        self.info_file.write(f"No. of 'MIDDLE' subjects: {len(self.age_info_dict['2'])}\n")
        self.info_file.write(f"No. of 'OLD' subjects: {len(self.age_info_dict['3'])}\n\n")
        self.info_file.write(f"Image Count distribution by subjects (key is subject ID)\n")
        self.info_file.write(f"--------------------------------------------------------\n\n")
        self.info_file.write(f"{self.id_cnt_dict}\n\n")
        self.info_file.write(f"Gender distribution by subjects ('0': Male, '1': Female)\n")
        self.info_file.write(f"--------------------------------------------------------\n\n")
        self.info_file.write(f"{self.gender_info_dict}\n\n")
        self.info_file.write(f"Age distribution by subjects ('0': CHILD, '1': YOUNG, '2': MIDDLE, '3': OLD)\n")
        self.info_file.write(f"----------------------------------------------------------------------------\n\n")
        self.info_file.write(f"{self.age_info_dict}\n\n")
        
        print("Done! All the information logged into 'dataset_info.txt'.")

if __name__=="__main__":
    
    # get config
    parser = argparse.ArgumentParser(
        description="IMFDB info extraction")
    parser.add_argument("config", type=str, help="absolute path to the config file (config.ini)")
    args = parser.parse_args()
    str_type_cfg = get_config(args.config)
    cfg = ConfigParams(str_type_cfg)

    info_extractor = IMFDB_Info(cfg)
    info_extractor.info()

    print("IMFDB Information Extraction Successful!")
    

