# thirdeye
> *An effort to ensure seeing remains believing*
Thirdeye is a comprehensive system for deepfake video detection developed by [Mahesha Kulatunga](http://maheshak.com/) for a MSc Data Analytics dissertation while at The University of Warwick. This code base handles the preprocessing, training and evaluating required for creating neural networks to be used for deepfake video classification. Included are 5 pre-trained 3D CNN architectures that can be used for unkown video classification. The following README outlines the basic functionality of the system.  

## Requirements
Thirdeye requires python3.6 or later. Hardware requirements for thirdeye are more or less in-line with the requirements for [TensorFlow](https://www.tensorflow.org/install). TensorFlow is tested and supported on the following 64-bit systems:

- Ubuntu 16.04 or later
- Windows 7 or later
- macOS 10.12.6 (Sierra) or later (no GPU support)
- Raspbian 9.0 or later

In addition, around _____________ disk space is required, ____________ including the DFD dataset.

### Dependencies
#### Python Version
- 3.6  

#### Software
- [FFmpeg](https://ffmpeg.org/)

#### Python Packages
- opencv_python==4.0.0.21
- moviepy==0.2.3.5
- face_recognition==1.2.3
- matplotlib==3.0.2
- pandas==0.23.4
- scipy==1.2.1
- Keras==2.2.4
- tensorflow==1.12.0
- numpy==1.16.1
- scikit_learn==0.21.3

## Set up
- Ensure all dependencies are satisfied.
- Navigate to the thirdeye directory in the command line.  
- Run the *example.py* file to run an example classification of sample videos included in the data folder.
```
python example.py
```

## How do I train a network?
- Ensure training files are stored in the ``` ./Data/TRAIN/DF_RAW ``` and  ``` ./Data/TRAIN/REAL_RAW ```
- Ensure testing files are stored in the ``` ./Data/TEST/DF_RAW ``` and  ``` ./Data/TEST/REAL_RAW ```
- Import and create an instance of thirdeye.
```
import thirdeye  
t = thirdeye.Thirdeye()
```
- Perform preprocessing on training files.
```
t.perform_preprocessing()
```
- Set the network to any desired; choose from *providence*, *odin* and *horus*.
```
t.set_network(<network>)
```
- Initiate training.
```
t.train()
```
- This can all be handled with a single command at system initialization.
```
t = thirdeye.Thirdeye(name=<network>, pre_p=True, force_t=True)
```
For access to the DFD dataset please contact mahesha.kulatunga@gmail.com.

## How do I classify unknown videos?
- Ensure all unknown videos are stored in ``` ./Data/UNKNOWN/UNKNOWN_RAW ```
- Import thirdeye and pick a network.
```
import thirdeye
t = thirdeye.Thirdeye(network=<network>)
```
- Carry out the classification
```
t.classify()
```
