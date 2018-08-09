## Nova Data Science: Getting Hands on with Deep Learning

These are the Jupyter notebooks and PowerPoint from my presentation at the NOVA Data Science Meetup on 7-25-2018. The subject for the evening was how to get started in deep learning, with a focus on convolutional neural networks (CNNs). The presentation gives a brief intro to how neural networks function, how they are trained, etc. The notebooks then give a hands on tutorial, with code that helps get people started with examples of creating a neural network from scratch, using transfer learning, and using Tensorflow's object detection API. All networks were implemented in [Tensorflow](https://www.tensorflow.org) and [Keras](https://keras.io/).

### Building a neural network from scratch
[MNIST.ipynb](https://github.com/laserstonewall/nova_data_science/blob/master/mnist/MNIST.ipynb): This Jupyter notebook shows how to build a simple neural network from scratch, trying out several different potential architectures. The initial architecture is similar to what is presented in the Keras blog's [tutorial](https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html) on building a CNN with limited data. Different modesl are tried to demonstrate how different parameters can be changed in the network, and the effect on training the networks can be observed quickly. The final model matches Keras's [MNIST model architecture](https://github.com/keras-team/keras/blob/master/examples/mnist_cnn.py).

### Using transfer learning
[Transfer_model_MNIST.ipynb](https://github.com/laserstonewall/nova_data_science/blob/master/transfer/Transfer_model_MNIST.ipynb): This notebook shows how to perform object classification in images using transfer learning with the [VGG-16 network](http://www.robots.ox.ac.uk/~vgg/research/very_deep/) pre-trained in [Keras](https://keras.io/applications/). I left the errors produced as we moved through the workflow in the notebook as a guide to solving some of the types of problems you will encounter often as you develop your own neural network applications.

### Tensorflow Object Detection API
[Tensorflow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection): The Tensorflow Object Detection API allows you to do some really neat things, using transfer learning on pre-trained models like Faster RCNN to perform object detection on your own images of interest. However, be prepared to put a few hours into the setup effort. In addition to downloading and labeling all your own images (described below), getting everything to run can be a bit tricky, and require a bit of time spent on Stack Overflow. 

For this part of the tutorial, I was inspired by several blog posts I read to create a model that could track objects in video, specifically to track fighter jets in that classic of American cinema, [Top Gun](https://www.imdb.com/title/tt0092099/). The first two blog posts that got me started are ([here](https://medium.com/coinmonks/part-1-2-step-by-step-guide-to-data-preparation-for-transfer-learning-using-tensorflows-object-ac45a6035b7a) and [here](https://towardsdatascience.com/how-to-train-your-own-object-detector-with-tensorflows-object-detector-api-bec72ecfe1d9)). I followed them closely, except I had never run anything on Google Cloud before, and I couldn't quite get my job to deploy properly. This is more my ignorance than any failure of Google Cloud I'm pretty certain. In the end I followed [this](https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10) tutorial very closely, modifying a few things because I was running on a Linux machine (see below for setting up your own Paperspace GPU machine), and got everything to finally work.

If you choose to do what I did and use the ML-in-a-box Paperspace machine, you won't have to worry about installing Tensorflow yourself. However, you WILL need to clone the [Tensorflow Models repo](https://github.com/tensorflow/models), which contains all the tools you will need for object detection. In order to follow exactly what he did in the tutorial, you will also need to step to this [specific commit](https://github.com/tensorflow/models/tree/079d67d9a0b3407e8d074a200780f3835413ef99) in Git. The way the training and test are called is restructured in later versions, and if you want to use them you will need to call the correct function to train (I have not done this). This means you will need to run 

```git checkout 079d67d9a0b3407e8d074a200780f3835413ef99```

to make sure you have the correct version of the Tensorflow models repository in order to follow along with him. I also recommend reading through the official documentation for [installing](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md) the object detection module, which will show the specific syntax for Linux, which is a bit simpler than on Windows. You will not need to run all the invididual installations if you are using the ML-in-a-box Paperspace instance. The official installation instructions do seem to leave out the `setup.py` step, however, which seems to be necessary to get everything working properly, and which is described in the main tutorial we are following.

You will train from the command line as described in the tutorial, and then you can modify the two provided Jupyter notebooks in this repo in order to perform the test detection on your images/video of interest for your new class.

### Building Your Own Dataset

For the Tensorflow Object Detection API example, you will need to build out your own custom dataset. The tutorial has a few Python files to convert your images once they are labeled, but first you will need to get them labeled in the appropriate format. For this, I first did a Google image search for "fighter jet" and then used the Fatkun Batch Download Image plugin for Google Chrome to automatically pull all the images.

Once the images were pulled down, I used [LabelImg](https://github.com/tzutalin/labelImg) to create bounding boxes around them.

After creating the dataset, you will need to modify a configuration file to have the correct number of classes, and to point to the correct locations for the training images in your project. An [example file](https://github.com/laserstonewall/nova_data_science/blob/master/object_detection/faster_rcnn_inception_v2_coco.config) is available in this repo. You'll need to [download](https://github.com/tensorflow/models/tree/master/research/object_detection/samples/configs) the appropriate sample configuration for your particular neural network. Additionally, you'll need to make a file that specifies your new classes, indexed starting at 1. The example for the fighter jet is available [here](https://github.com/laserstonewall/nova_data_science/blob/master/object_detection/fighter_label.pbtxt).

## Additional Information to Get Started

### Where to train on a GPU

If you are going to be working on neural network models much, you'll want to get access to a GPU. Training on the CPU of your personal computer is possible, but is prohibitively slow if you want to actually experiment with different architectures or train any larger networks. Luckily, there are now several options availabe. In these systems, you pay to use a GPU in a cloud server at an hourly rate. 

Some of the most popular options are [AWS](https://docs.aws.amazon.com/dlami/latest/devguide/gpu.html), Google Cloud, [Crestle](https://www.crestle.com/), and [Paperspace](https://www.paperspace.com/). For the tutorial, I ended up choosing to use Paperspace. Paperspace allows you to set up multiple different machine types, and has a few prebuilt for machine learning that are really convenient.

After you sign up for Paperspace, you will need to request access to their pre-built GPU machines. This usually takes a day or two. I went for the ML-in-a-Box, which comes pre-built with 90% of the packages you will need for this tutorial. With the slower GPU option, this ends up costing $0.51 / hour, which is much more reasonable than dropping $1000+ for your own GPU when you are starting out.

### Recommended further resources

[fast.ai](http://www.fast.ai/): This is one of the best spots to get hands on learning with GPUs. They focus on getting you started with actually running models from the get-go, and then slowly fill in knowledge through a series of projects. This course is totally free, and they have their own pre-built machine type on Paperspace (don't use that machine for this tutorial, however, as it is centered around PyTorch and wont have all the packages you need).

[Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/): I think this is one of the best free resources for learning about neural networks on the web. My background is in physics, which also happens to be the author's background, so I may be biased. I would love to hear others feedback on it.

[Coursera Deep Learning Specialization](https://www.coursera.org/specializations/deep-learning): I haven't actually done these courses, but from what I have heard, they are another great way to learn about neural networks, focusing on the theory behind how they work more.
