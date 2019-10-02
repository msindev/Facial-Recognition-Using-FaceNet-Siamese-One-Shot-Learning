# Facial-Recognition-Using-FaceNet-Siamese-One-Shot-Learning
This program is used to implement Facial Recognition using Siamese Network architecture. The implementation of the project is based on the research paper : 

> FaceNet: A Unified Embedding for Face Recognition and Clustering
> [arXiv:1503.03832](https://arxiv.org/abs/1503.03832) by [Florian Schroff](https://arxiv.org/search/cs?searchtype=author&query=Schroff%2C+F), [Dmitry Kalenichenko](https://arxiv.org/search/cs?searchtype=author&query=Kalenichenko%2C+D), [James Philbin](https://arxiv.org/search/cs?searchtype=author&query=Philbin%2C+J)

Facenet implements concept of Triplet Loss function to minimize the distance between anchor and positive images, and increase the distance between anchor and negative images.

### Prerequisites

    h5py==2.8.0
	Keras==2.2.4
	tensorflow==1.13.0rc2
	dlib==19.16.0
	opencv_python==3.4.3.18
	imutils==0.5.1
	numpy==1.15.2
	matplotlib==3.0.0
	scipy==1.1.0

Install the packages using `pip install -r requirements.txt`

### Usage
To use the facial recognition system, you need to have a database of images through which the model will calculate image embeddings and show the output. 
The images which are in the database are stored as .jpg files in the directory `./images`.


To generate your own dataset and add more faces to the system, use the following procedure:

> Sit in front of your webcam.
> Use the Image_Dataset_Generator.py script to save 50 images of your face.
> Use this command: `python Image_Dataset_Generator.py` to generate images which will be saved in images folder.

To use the facial recognition system, run the command on your terminal : 
`python face_recognizer.py` 

### References

 1. The code has been implemented using deeplearning.ai course Convolutional Networks Week 4 Assignment, which has the files `fr_utils.py` and `inception_blocks_v2.py`
 2. The keras implementation of the model is by Victor Sy Wang's implementation and was loaded using his code:  [https://github.com/iwantooxxoox/Keras-OpenFace](https://github.com/iwantooxxoox/Keras-OpenFace).
