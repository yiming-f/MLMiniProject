

# **Model Card**:


The input of the problem is an image of a dog. The output is the dog's breed.


The training data consists of 6,166 images of 120 different breeds of dogs. Each image has its own id (the series of numbers and letters before the .jpg) which can be used to identify the breed of the dog through the labels.csv file which has two columns, id and corresponding breed. The source of our training data is [the Stanford Dogs Dataset](http://vision.stanford.edu/aditya86/ImageNetDogs/ ), but we originally found and downloaded the dataset through a [competition](https://www.kaggle.com/c/dog-breed-identification/overview) Kaggle hosted from 2017-2018 using the dataset. Note that the dataset from the competition had 10,222 train images and another 10,000 or so test images, but we had to split their training dataset 60:40 into our train and test sets since their test data didn't have labels.


The intended use of our trained model is to get information about a dog’s breed after inputting a picture of it.


The main thing that could potentially go wrong when using the model is that it outputs the wrong breed. Our training data includes 120 dog breeds which is a lot, but definitely not enough to cover all the dog breeds in existence, especially when you consider mixed breeds. If you input an image of a dog breed not in the dataset, the answer is guaranteed to be wrong. Of course, even if you input an image of a dog breed in the dataset the answer is still not guaranteed to be right. Similarly, if the image inputted has no dog, this would also probably give an incorrect result.


# **Introduction**: 


This project is created to evaluate how well different training methods can identify different dog breeds. 


The two techniques we compared are convolutional neural networks (fine-tuning MobileNet) versus normal neural networks.


Our experimental question is: How does the number of hidden layers affect the accuracy of a fully connected neural network.


More detail on our techniques as well as results are below.


### MobileNet


The first method we plan to use to classify this dataset will be a fine tuning of MobileNet. MobileNet is a convolutional neural network, where convolution kernels are used to collect important details and simplify images. Because MobileNet was constructed to classify a variety of images, we wanted to explore whether it could also be used on a more specific dataset. In order to create this model, we plan to replace the linear classification layer with a 120 neuron layer representing the dog breeds. To get a final prediction, we use torch.argmax with a softmax to get the dog breed with the highest probability. 

### Neural Net
The second method we will use to classify the dataset is a more general classification method that isn't specifically oriented to images: Neural Net


In this technique, each pixel will be passed in as an input. These inputs will then be passed through various linear and activation layers before being passed to a final output layer that will return the probabilities of each dog breed using a softmax function. We get a final prediction using torch.argmax as mentioned above. We plan on experimenting with three different amounts of hidden layers to evaluate their impact on the model's training and accuracy.


This differs from MobileNet as it uses the image's information directly, requiring significantly more parameters and nodes.


### Results
We have found that the CNN ended up with more/less accurate results in comparison to the fully connected neural networks (or maybe it's only better compared to the lower number of networks). This may be due to CNN being specifically built for image processing OR (if less accurate)the number of weights/nodes allowed neural networks to be more specific…maybe


# **Experiment**
As a reminder, our experimental question is: How does the number of hidden layers affect the accuracy of a fully connected neural network?


Our experimental technique was simply making 3 neural networks, with 3, 5, and 7 hidden layers respectively. For each layer, the number of output neurons is half of the number of input neurons. For all of the networks, we repeatedly halved the number of neurons in each layer until we reached the desired number of hidden layers, at which point we made the last hidden layer output 30 neurons, and then in our output layer we take the 30 neurons and change it to output 120 neurons, then get a final breed prediction as detailed in our neural network methods section.


As seen from our results, the fully connected neural networks with more hidden layers had higher accuracy for both the training and validation dataset. This is likely because the increased number of hidden layers allows for better generalization, since all the layers put together are able to learn complex patterns better.





