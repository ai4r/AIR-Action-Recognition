# AIR Action Recognition

Action Recognition Modeule(FSA-CNN) using 2D skeleton extracted fromm ETRI-Activity3D dataset.

The accuracy is 91.00% for Testset.



## Setting 
-   Python = 3.6.8     
-   Tensorflow-gpu  or tensorflow = 1.12.0    
-   Keras = 2.2.4

## Source files
    .
    ├── TestBed_OpenPose_v4_COCO_6_9100.h5    # Weight file                  
    ├── Test_Code.py                 # Test code that consists of reading samples, loading models with trained weights and testing 
    ├── Training_Code.py             # Training code using ETRI-Activity3D Dataset
    ├── LICENSE.md
    ├── LICENSE_ko.md
    └──README.md

## Installation


1. clone this git

2. Download ETRI-Activity3D_Mat file from 

https://drive.google.com/drive/folders/1KrLsDfJS9nfTZwBB52TEBVT3nHN2yY8e?usp=sharing

and unzip at the root folder.

(every .mat files should be at just inside of "ETRI-Activity3D_Mat" folder)


3.  install the requirements:

pip install keras==2.2.4

pip install tensorflow-gpu==1.12.0

pip install libpython or conda install libpython

(maybe random, math, numpy and os modules are included in libpython)







## Training

run

"python Training_Code.py"

If you want to set pre-trained weights as initialization, 

Unlock the comment of line 211("network.load_weights(weight_path)").


You can get "Weight_save_temp.h5" as weights of the latest epoch, and

"Weight_save.h5" as weights of the best test accuracy during your training.



## Test

run

"python Test_Code.py"

If you want to initialize using your own weights, 

change line 31("weight_path = 'TestBed_OpenPose_v4_COCO_6_9100.h5' ")

to your weight file.







## LICENSE
This software is a part of AIR, and follows the [AIR License and Service Agreement](LICENSE.md).



## Citation
Jang, J., Kim, D., Park, C., Jang, M., Lee, J., & Kim, J. (2020). ETRI-Activity3D: A Large-Scale RGB-D Dataset for Robots to Recognize Daily Activities of the Elderly. arXiv preprint arXiv:2003.01920.