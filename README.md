# CapsNet-1D
1D Capsule Network for Gesture Recognition in Keras.

# Introduction
The project presents a Keras implementation of one-dimensional deep capsule network (CapsNet) architecture for continuous Indian Sign Language recognition by means of signals obtained from a custom designed wearable IMU system. The performance of the proposed CapsNet architecture is assessed by altering dynamic routing between capsule layers. The proposed CapsNet yields improved accuracy values of 94% for 3 routings and 92.50% for 5 routings in comparison with the convolutional neural network (CNN) that yields an accuracy of 87.99%. Improved learning of the proposed architecture is also validated by spatial activations depicting excited units at the predictive layer. Finally, a novel non-cooperative pick-and-predict competition is designed between CapsNet and CNN. Higher value of Nash equilibrium for CapsNet as compared to CNN indicates the suitability of the proposed approach.

# Usage 
Clone or Download the repository and save the files in your Python directory. Pass your dataset as the input to the 'CapsNet.py' file
and then run the code.

# Dependencies 
1. Python (3.6 or higher)
2. Keras (2.2.0 or higher)
3. TensorFlow (1.10.0 or higher)
3. Numpy

# Results


# Acknowledgment
We would like to recognize the funding support provided by the Science & Engineering Research Board, a statutory body of the Department of Science & Technology (DST), Government of India, SERB file number ECR/2016/000637.

# Reference
https://www.sciencedirect.com/science/article/pii/S0045790619301508

