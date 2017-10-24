
Quick and dirty neural network implementation for the [MNIST db](http://yann.lecun.com/exdb/mnist)

I'm learning both Rust and Machine Learning, so you probably don't want to use this code :)


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

* Layers:
	* Input (784)
	* Hidden (128 neurons, relu)
	* Hidden (16 neurons, relu)
	* Output (10 neurons, softmax)

References:

* https://mmlind.github.io/Simple_1-Layer_Neural_Network_for_MNIST_Handwriting_Recognition/
* https://mmlind.github.io/Simple_3-Layer_Neural_Network_for_MNIST_Handwriting_Recognition/
* https://mxnet.incubator.apache.org/tutorials/python/mnist.html


