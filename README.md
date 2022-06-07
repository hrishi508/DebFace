# DebFace
# Description
An implementation of the paper - "Jointly de-biasing face recognition and demographic attribute estimation" by Sixue Gong et al., 2020 [[1]](#1)

# Directory Structure
Use the tree command to recursively generate the directory structure and insert here.
```
.
├── adversarial_training.py
├── backbones
│   ├── am_softmax.py
│   ├── classifier.py
│   ├── debface.py
│   ├── encoder.py
│   ├── __init__.py
│   ├── iresnet.py
├── config1.ini
├── config2.ini
├── LICENSE
├── model_summary.txt
├── README.md
├── train_classifiers.py
└── utils
    └── utils_config.py

4 directories, 22 files
```
The above tree shows the overall structure of repository.

# Environment Setup

## Dependencies
1. Python 3.6.9
2. PyTorch 1.9.0+cpu
3. torchsummary 1.5.1
4. argparse 1.4.0
5. configparser 5.2.0

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

# Training the model
## Datasets

# Running the model

# Demo of model inference
<insert, screenrecording and examples of running>
   
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

