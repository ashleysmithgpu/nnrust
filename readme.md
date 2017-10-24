
Quick and dirty neural network implementation for the [MNIST db](http://yann.lecun.com/exdb/mnist)

I'm learning both Rust and Machine Learning, so you probably don't want to use this code :)

## Meanings

* **Neuron**: Holds a value (it's acitvation) and is attached to 0 or more other neurons. Each connection has a weight.
* **Activation**: calculation of how active this neuron is with respect to it's weights. Activation types:
	* Sigmoid acitvation: Mathematical representation of a biological neuron
	* ReLU (Recitfied Linear Unit) acitvation: Faster learning than above, but less close to the biological neuron
	* Tanh acitvation
	* "soft" max acitvation: Usually used on the output nodes to select the highest activated node. This activation changes multiple neuron activations in to one discreet value.
* **Weight**: Each connection between a neuron and another neuron has a weight associcated with it. In training this weight changes. In testing it does not.
* **Bias**: A single value attached to each layer of the network.
* **Training**: Also called back propogation. Runs the network forwards, then adapts the weights according to the errors.
* **Testing**: Testing the network should be performed on different data than training the network.
* [**MLP**](https://en.wikipedia.org/wiki/Multilayer_perceptron) (Multilayer perceptron): At least 3 layer network (I.e. not the one layer network below). Almost the simplest network available
* [**Perceptron**](https://en.wikipedia.org/wiki/Perceptron): A digital representation of a neuron
* **Fully connected**: A layer in a network where each neuron is connected to each neuron in the previous layer by a weight.
* **Epoch**: One full run over the training data.

## One layer networks

Learning rate 0.05:
Tests passed: 8501, tests failed: 1499 (85.009995%) with 60 epochs training (85.007164% training)

* Layers:
	* Input (784)
	* Output (10 neurons, softmax)

## Two layer networks

Learning rate 0.05:
Tests passed: 9299, tests failed: 701 (92.99%) with 60 epochs training (93.315% training)

* Layers:
	* Input (784)
	* Hidden (16 neurons, sigmoid)
	* Output (10 neurons, softmax)

Learning rate 0.05:
Tests passed: 9376, tests failed: 624 (93.76%) with 29 epochs training (93.61% training)

* Layers:
	* Input (784)
	* Hidden (16 neurons, relu)
	* Output (10 neurons, softmax)

Learning rate 0.05:
Tests passed: 9410, tests failed: 590 (94.1%) with 11 epochs training (93.53833% training)

* Layers:
	* Input (784)
	* Hidden (128 neurons, relu)
	* Output (10 neurons, softmax)

## Three layer networks

Learning rate 0.001 seed 0:
Tests passed: 9632, tests failed: 368 (96.32%) with 19 epochs training (96.55333% training)
Tests passed: 9809, tests failed: 191 (98.09%) with 79 epochs training (99.225006% training)

* Layers:
	* Input (784)
	* Hidden (128 neurons, relu)
	* Hidden (16 neurons, relu)
	* Output (10 neurons, softmax)

References:

* https://mmlind.github.io/Simple_1-Layer_Neural_Network_for_MNIST_Handwriting_Recognition/
* https://mmlind.github.io/Simple_3-Layer_Neural_Network_for_MNIST_Handwriting_Recognition/
* https://mxnet.incubator.apache.org/tutorials/python/mnist.html

