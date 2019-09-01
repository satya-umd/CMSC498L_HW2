
# Cat Classifier
Both the classifiers have been implemented using pytorch with data processing functions being used
from Assignment 2 template.

A 2-layer architecture was used for the prediction with an additional final layer with the number of
nodes equal to the number of classes for prediction(in our case its 2). In between the layers, ReLu
activation and Sigmoid activation is used respectively. To prevent overfitting, L2 regularizer was
used on the weights of hidden layers along with dropout of 20% in between layers. Cross Entropy
loss was used along with Stochastic gradient descent in both the classifiers.
Hyper parameters of momentum, weight decay was used with gradient descent and Lambda1 pa-
rameter of L2 regularizer was used along with the cross entropy loss.

I played with a lot of different values for Dropout and L1 & L2 regularizers, along with different
losses(like MSELoss), different activation functions(like Softmax and Leaky Relu) and Optimiz-
ers(like Adam), but eventually came back to a simpler implementation of the network. Softmax
was really tough to train and Dropout on a smaller network didn’t yield any significant improve-
ment in accuracy for the Cat Classifier.

For the Cat classifier I ended with 74% accuracy, there were 13 misclassified images in the validation
set, with 9 false negatives and 4 false positives.The major reason for misclassfication might be the
contrast of the Cat’s face and body compared to the background and also the posture of how the cat
is shown in the picture. Most of the training set had images of cats, with a high contrast between
the cat’s body and a different colored, mostly dark background. Also in the training images, the
cat is framed at the centre of the image with it’s head occupying major portion of the image. While
in the misclassified validation images, the cat is not usually centred and is sometimes lying down
or in weird postures. Also, the background in some images doesn’t have a high contrast, which
might be causing the misclassification. Adding more augmented images to our training set as well
as increasing the training data might help improve the accuracy and prevent these false negatives.
Dropout wasn’t very effective for this dataset and my best model eventually ended with a learning
rate = .02 and momentum = 0.9 with weight decay = 0.0005 and Lamda = 0.05 for L2 regularizer.

For the IMDB dataset I ended with about 85% accuracy for both the validation set and the test
set.The main reason for misclassfication might be due to the bag of words model losing the context
of sentences and classifying on the basis of the number of negative and positive words.A better
accuracy can be achieved by training on bigrams and trigrams on the same dataset, which captures
the relative ordering and context of the reviews or using a deeper network. Dropout of around 20%
along with a learning rate = 0.3 and momentum = 0.9 with weight decay = 0.0005 and Lamda =
0.01 for L2 regularizer was used.
