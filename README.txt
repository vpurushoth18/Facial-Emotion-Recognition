# Facial Emotion Recognition Using Residual Masking Network, in Pytorch

## Used Machine Specification
**Training**
platform       - Google colab pro
GPU            - Tesla P100 (16 GB memory)
CUDA version   - 11.2
CPU            - 2 vCPU 
RAM            - 24 GB memory

**Inference**
platform        - NITT CSE Dr.M.Sridevi madam's Software Development lab
GPU             - NVIDIA GEFORCE RTX 3060
CUDA version    - 11.3
CPU             - i7-9700 CPU 3.00GHz
RAM             - 32 GB memory




### Installation

Create a new anaconda enviornment in order to install the required packages needed to run this model.

```
conda create -n myenv python=3.8

```
### Prerequisites

Install the packages required after activating the created enviornment. 

```
pip install -r requirements.txt

```
**NOTE**
 if issue arises while installing pytorch, it can happen because of the gpu in that machine is not compatible with the cuda version in the requirements.txt. So the pytorch has to seperatly be installed to the enviornment. 

## Getting Started 

These instructions will get this model up and running. We have divided into two parts i) Using the already trained model and use the model ii) Building(training) the model from scratch

### Method 1 : Using the built model

Here you can take an advantage of using the already trained model on FER-2013 dataset,FACES and JAFFE datasets giving overall accuracy of 73.001%.

##### Sub-Method 1

M1> To run rmn model, first open the 'rmn.ipynb' notebook and run the cells. This notebook saves the face detected gray-scale images in the 'face_frames_0_rot' folder,just before passing the face detected images to emotion detection model.

(This notebook will import the RMN class from rmn module and creates an object of that class, this object will automatically loads the trained weights to the architecture and makes use of the functions it provides)

##### Sub-Method 2

M2> Run the 'ssd_infer.py' to run real time facial emotion detection. This pyscript saves the face detected gray-scale images in the 'ssd_infer_faces' folder, which goes in to the emotion detection model.   

### Method 2 : Build the model from scratch 

The dataset is given in csv file format. The dataset contains FER 2013 dataset as the main set and other dataset such as FACES and JAFFE which
are controlled condition set which adds little variation in dataset.
The dataset composed as follows.

+----------------+--------------------+------------------------+----------------+
|  Dataset Name  |Traning datapoints  | Validation datapoints  |Test datapoints |
+----------------+--------------------+------------------------+----------------+
|FER 2013        |      26444         |          3859          |   3589         |
|----------------+--------------------+------------------------+----------------+                             
|FACES           |       2052         |           0            |     0          |
|----------------+--------------------+------------------------+----------------+
|JAFFE           |       213          |           0            |     0          |
+----------------+--------------------+------------------------+----------------+

#### setting up the configuration

1) Go to 'configs' folder and select the 'fer2013_config.json'     # this helps to set up the parameters(architecture as well as training parameters for training)

                                                                   # Architectures choices can be made by going through 'models' folder and even the residual masking
                                                                   # architectures variations can be seen in the 'resmasking.py'(the parameter to be changed is 'arch').

#### Training and Testing in private Test set

2) Run the 'main_fer2013.py' file to train the choosen model from scratch.
   After the training and validating the model, it saves the trained weights in
   the 'checkpoint' as the pickel file. To access and run the trained weights

   i) Load the weights using the following code.(jus
     ```
     emo_model = resmasking_dropout1(in_channels=3, num_classes=7)
     state = torch.load('name_given_for_file')
     emo_model.load_state_dict(state["net"])
     emo_model.eval()

     ```````
   .The best checkpoints will chosen at term of best validation accuracy, located at 'checkpoint'.
   .The TensorBoard training logs are located at 'log', to open it, use tensorboard --logs/

     

 



