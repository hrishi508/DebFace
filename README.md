# DebFace
# Description
An implementation of the paper - "Jointly de-biasing face recognition and demographic attribute estimation" by Sixue Gong et al., 2020 [[1]](#1). This project is aimed at de-biasing and privacy preservation in biometric methods that use face verification.

# Environment Setup

## Dependencies
1. Python 3
2. PyTorch 
3. torchsummary
4. argparse
5. configparser
6. torchviz
7. OpenCV
8. glob
9. shutil
10. Pandas
11. NumPy
12. Pillow

## Using host OS environment:
1. Check to see if your Python installation has pip. Enter the following in your terminal:

        pip3 -h
        
     If you see the help text for pip then you have pip installed, otherwise [download and install pip](https://pip.pypa.io/en/latest/installing.html)

2. Clone the repo from GitHub and then install the various dependencies using pip

      Mac OS / Linux
        
        git clone https://github.com/hrishi508/DebFace.git
        cd DebFace/


## Using a virtual environment:
1. Check to see if your Python installation has pip. Enter the following in your terminal:

        pip3 -h
        
     If you see the help text for pip then you have pip installed, otherwise [download and install pip](https://pip.pypa.io/en/latest/installing.html)

2. Install the virtualenv package

        pip3 install virtualenv
        
3. Create the virtual environment

        virtualenv debface_env
        
4. Activate the virtual environment

      Mac OS / Linux
        
        source debface_env/bin/activate

5. Clone the repo from GitHub and then install the various dependencies using pip

      Mac OS / Linux
        
        git clone https://github.com/hrishi508/DebFace.git
        cd DebFace/
        
# Directory Structure
```
.
├── backbones
│   ├── am_softmax.py
│   ├── classifier.py
│   ├── debface.py
│   ├── encoder.py
│   ├── __init__.py
│   └── iresnet.py
├── config.ini
├── dataset_cleaner.py
├── dataset_filter.py
├── dataset_info.py
├── dataset_organizer.py
├── dataset_splitter.py
├── DebFace Computation Graph
│   ├── DebFace_Final
│   ├── DebFace_Final.png
│   ├── DebFace_Final_without_race
│   ├── DebFace_Final_without_race.png
├── full_training_strategy.txt
├── LICENSE
├── model_summary.txt
├── README.md
├── train.py
├── train_without_race.py
└── utils
    └── utils_config.py

3 directories, 23 files

```

# Config File
To make the control of all hyperparameters and paths across this project seamless and simple, I have created a global configuration file - [config.ini](/config.ini). All the scripts in this project access and use the various argument values listed in the config file. If you intend to train the DebFace model on your device, you will have to set the arguments in the config file accordingly. I have included detailed comments for each of the arguments to make it simple to use.

The config.ini file is integrated with all the other scripts in this project via the [utils_config.py](/utils/utils_config.py) script. I have provided a 'ConfigParams' class that extracts all the information from the config file.

# Datasets
For training the DebFace model, we require a dataset that satisfies the following constraints:
1. Frontal Face images organized subject wise with at least 100 images per subject
2. Gender, Age and Race labels associated with the images, i.e. every image will have such a label - [ID, Gender, Age, Race]

I have spent a lot of time looking for such a dataset but the only dataset that I could find which satisfied all the above constraints was the [VGG-Face2 Mivia Age Estimation (VMAGE) Dataset](https://mivia.unisa.it/datasets/vmage/). The only issue with this dataset was that it was based on the [VGG-Face2 dataset](https://github.com/ox-vgg/vgg_face2) which is currently been removed from the [oxford server](https://www.robots.ox.ac.uk/~vgg/data/vgg_face2/) due to [legal issues](https://github.com/ox-vgg/vgg_face2/issues/54). If you find any pubicly available dataset that satisfies all the above constraints, please do [tell us](#contact-us). 

So, for training this model, I went with a dataset which partially satisfies the contraints. I have used the IMFDB [[2]](#2) dataset. The only drawback of using this dataset was that since it contains images of only __Indian Movie Actors__, it is devoid of __Race labels__. You can find the details of this dataset [here](https://cvit.iiit.ac.in/projects/IMFDB/).

The original dataset has many incomplete labels and various mismatches among the labels and the corresponding images. It also has many extra labels that are irrelevant for training the model (the only ones we require are ID, Gender, Age and Race).

So, I first partly cleaned the dataset manually ([available here](https://drive.google.com/drive/folders/1QHmbNRW5kDv8l7MjCw3iFcaOfRWQ8B-y?usp=sharing)), then I designed a few custom scripts to automate the cleaning, flitering and transformation of the IMFDB dataset so that it can be directly used for training the [DebFaceWithoutRace](#model-definition-scripts) model. I have uploaded the final cleaned version of the dataset [here](https://drive.google.com/drive/folders/13oCfNzDiQTYgUsb9w9VJO5yf5xLzcF2U?usp=sharing).

Now, you can either choose to download the [final dataset](https://drive.google.com/drive/folders/13oCfNzDiQTYgUsb9w9VJO5yf5xLzcF2U?usp=sharing) and directly train the DebFace model on it, or you can use my custom-made scripts on the [manually cleaned version](https://drive.google.com/drive/folders/1QHmbNRW5kDv8l7MjCw3iFcaOfRWQ8B-y?usp=sharing) of to generate your dataset for training.

__NOTE: If you download the dataset, please create a 'datasets/' folder in the root directory of this repository that you had cloned earlier in the [environment setup](#environment-setup) section and move this downloaded dataset folder 'IMFDB_final/' into the 'datasets/' folder for the training scripts to work smoothly__

__WARNING: DO NOT RUN THESE SCRIPTS ON THE ORIGINAL IMFDB DATASET!__ 

__They will end up throwing a lot of errors due to the mislabelled samples and mismatches. Only run in this script on the manually cleaned version that I have provided [here](https://drive.google.com/drive/folders/1QHmbNRW5kDv8l7MjCw3iFcaOfRWQ8B-y?usp=sharing).__

Following are the steps to generate your own data from the manually cleaned IMFDB using the custom-made scripts:
1. After cloning this repository, navigate to the root (i.e. DebFace/) and create a 'datasets/' folder there.
2. Download the manually cleaned IMFDB dataset from [here](https://drive.google.com/drive/folders/1QHmbNRW5kDv8l7MjCw3iFcaOfRWQ8B-y?usp=sharing) and move it into the newly created 'datasets/' folder.
3. Set the arguments in the [config.ini](/config.ini) appropriately
4. Navigate to the root directory of the repository, and run the custom scripts in succession using the commands given below:

        python3 dataset_organizer.py FULL-PATH-TO-THE-CONFIG-FILE
        python3 dataset_cleaner.py FULL-PATH-TO-THE-CONFIG-FILE
        python3 dataset_filter.py FULL-PATH-TO-THE-CONFIG-FILE
__NOTE: Replace the FULL-PATH-TO-THE-CONFIG-FILE with the path of the [config.ini](/config.ini) file enclosed in "". For example, for my device, I set the FULL-PATH-TO-THE-CONFIG-FILE to "/home/hrishi/Repos/DebFace/config.ini".__

Following the above steps will create many directories in the 'datasets/' folder. The final dataset directory is the 'IMFDB_final/' which will be used for training. You can igonre the other intermediate directories i.e., 'IMFDB_simplified/' and 'IMFDB_cleaned/'.

To extract metadata about the newly generated IMFDB_final, I have provided a custom script [dataset_info.py](/dataset_info.py). Running this script will log all the important details extracted from the newly generated dataset to a 'IMFDB_final_info.txt' file in the 'datasets/' folder.

    python3 dataset_info.py FULL-PATH-TO-THE-CONFIG-FILE

# Model Definition Scripts
__NOTE: I have also provided a 'DebFaceWithoutRace' model (in addition to the 'DebFace' model). This is the same model as 'DebFace' but excluding the 'race' classifier and all its connection. This is the model that has been used in this project for reasons that have been already covered in the [datasets](#datasets) section.__
The following scripts have been directly taken from the implementation provided by [InsightFace](https://github.com/deepinsight/insightface) [[3]](#3):
1. [init.py](/backbones/__init__.py)
2. [iresnet.py](/backbones/iresnet.py)

The script [am_softmax.py](/backbones/am_softmax.py) has been directly taken from the implementation provided by [DebFace](https://github.com/gongsixue/DebFace). I have tried using the am_softmax from this script according to the paper but their implementation is buggy since instead returning values in the range (0, 1) (logits), it returns all kinds of positive and negative real values. So, wherever am_softmax was supposed to be used, I have replaced it with ordinary softmax.

The following scripts have been created by me from scratch to provide a seamless implementation of the model given in the [DebFace](#1) paper:
1. [encoder.py](/backbones/encoder.py) - This script defines the encoder class that initializes the ArcFace50 encoder provided by [InsightFace](https://github.com/deepinsight/insightface) and appends a ReLU layer to it.

2. [classifier.py](/backbones/classifier.py) - This script contains the classifier class whose input and outputs can be modfied to create the various age, gender, race and ID classifiers as given in the model architecture of the [DebFace](#1) paper. Since the exctl details of the classifiers was missing in the paper, I have used a single layer neural network in the script.

3. [debface.py](/backbones/debface.py) - This script contains the 'DebFace' class which basically initializes the encoder and all the demographic and ID classifiers and integrates them seamlessly. __NOTE: I have also provided a 'DebFaceWithoutRace' class, this is the same model as above but excluding the 'race' classifier and all its connection. This is the class that has been used in this project for reasons that will be covered in the section below.__

# Training the model
I have provided two custom scripts [train.py](/train.py) and [train_without_race.py](/train_without_race.py) to make the training process seamless and make the iteration of the various hyperparameters easier.

__NOTE: As already mentioned earlier in the [datasets](#datasets) section, this project is using the IMFDB dataset for model training which does not contain race labels. Thus, this project uses the [train_without_race.py](/train_without_race.py) to train the model which makes use of the 'DebFaceWithoutRace' class in the [debface.py](/backbones/debface.py).__

__NOTE: If you want to train the model on a dataset which has all the labels (satisfies all the constraints mentioned in the [datasets](#datasets) section), please use [train.py](/train.py) for the same.__

Following are the steps to train the DebFace model on the [final dataset](https://drive.google.com/drive/folders/13oCfNzDiQTYgUsb9w9VJO5yf5xLzcF2U?usp=sharing) that you either already downloaded or genereated by following the steps in the [datasets](#datasets) section:
1. Navigate to the root directory of the repository, and run the [dataset_splitter.py](/dataset_splitter.py) that I have provided to split the final dataset into Train and Test using the command given below:

        python3 dataset_splitter.py FULL-PATH-TO-THE-CONFIG-FILE

Running the above will generate 'Train/' and 'Test/' directories in your 'datasets/' directory that you had created while following the steps in the [dataset](#datasets) section.

2. Set the arguments in the [config.ini](/config.ini) appropriately
4. You're set to start training the model now! Run the command below to begin the training:

        python3 train_without_race.py FULL-PATH-TO-THE-CONFIG-FILE

__NOTE: Replace the FULL-PATH-TO-THE-CONFIG-FILE with the path of the [config.ini](/config.ini) file enclosed in "". For example, for my device, I set the FULL-PATH-TO-THE-CONFIG-FILE to "/home/hrishi/Repos/DebFace/config.ini".__

<!-- # Running the model

# Demo of model inference
<insert, screenrecording and examples of running> -->
   
# Contributing to the project

## Where do I start?
* **Ask us** by reaching out to any of the contributors through the [Contact Us](#contact-us) section. Someone there could need
  help with something.
* You can also **take the initiative** and fix a bug you found, create an issue for discussion or
  implement a feature that we never though of, but always wanted.

## Ok, I found something. What now?

* **[Tell us](#contact-us)**, if you haven't already. Chances are that we have additional information
  and directions.
* **Read the code** and get familiar with the engine component you want to work with.
* Do not hesitate to **[ask us for help](#contact-us)** if you do not understand something.

## How do I contribute my features/changes?

* You can upload work in progress (WIP) revisions or drafts of your contribution to get feedback or support.
* Tell us (again) when you want us to review your work.    
    
# Contact us
* Hrishikesh Kusneniwar - [hrishi508](https://github.com/hrishi508)
* Dr. Sudipta Banerjee - [sudban3089](https://github.com/sudban3089)

# References
<a id="1">[1]</a> 
Gong, Sixue and Liu, Xiaoming and Jain, A (2020). 
Jointly de-biasing face recognition and demographic attribute estimation 
ECCV, [link to the paper](http://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123740324.pdf).

<a id="2">[2]</a> 
Shankar Setty, Moula Husain, Parisa Beham, Jyothi Gudavalli, Menaka Kandasamy, Radhesyam Vaddi, Vidyagouri Hemadri, J C Karure, Raja Raju, Rajan, Vijay Kumar and C V Jawahar (2013). 
Indian Movie Face Database: A Benchmark for Face Recognition Under Wide Variations 
NCVPRIPG, [link to the paper](http://cdn.iiit.ac.in/cdn/cvit.iiit.ac.in/images/ConferencePapers/2013/Shankar2013Indian.pdf).

<a id="3">[3]</a> 
Deng, Jiankang and Guo, Jia and Niannan, Xue and Zafeiriou, Stefanos (2019). 
ArcFace: Additive Angular Margin Loss for Deep Face Recognition 
CVPR, [link to the paper](https://arxiv.org/abs/1801.07698).




