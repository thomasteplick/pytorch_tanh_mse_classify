<h3>PyTorch Neural Network Classification using Linear Layers, Hyperbolic Tangent (tanh(x)) Activation Function, Stochastic Gradient Descent (SGD), and Mean Square Loss</h3>
<p>
This program classifies data from four uniformly-distributed sources.  Each source has ten properties that are uniformly distributed in a sub-interval of (0, 256].
Input samples are randomly taken from the ten regions for each class and are used as the input to the neural network after normalization.  The normalization
divides the input by 256, subtracts 0.5, and multiplies that by two, which produces a floating point number in [-1.0,1.0].
</p>
<img width="190" height="45" alt="image" src="https://github.com/user-attachments/assets/1a4a344b-f6c9-4e7a-a626-f72a86be0d88" />
<p>
The target is encoded to two outputs with class1 = (-1,-1), class2 = (-1,1), class3 = (1,-1), and class 4 = (1,1).  This corresponds to the 
hyperbolic tangent function, tanh(x), which has limits of -1 and 1 and is linear near 0.  It is also has a derivative which is a requirement for backpropagation.
The user can specify 1, 2, or 3 hidden layers with up to 20 features in each hidden layer.  So the input samples consist of the ten property values and the 
target encoded to two outputs in (-1,1).  The user can specifiy the number of epochs and the epoch sample size.  The user can also specify the learning rate and momentum hyperparameters.
After the network is trained, the Mean Square Error Loss per epoch is displayed.  The testing phase follows the training.  A different set of samples is presented to
the neural network.  The number of sample inputs for each class and the number of correct decisions is recorded.  The test results for each class and the totals 
are shown in a table.  The activation function used for each fully-connected layer is the hyberbolic tangent (tanh).  The Mean Square Error Loss function is used to determine
the output error and backpropagation is used to distribute the error to previous-layer weights and biases using partial derivatives and the chain rule of 
Differential Calculus.  PyTorch handles all this transparently to the user with Tensors and Autograd.  The Mean Square Error Loss function takes the squared difference
between the predicted outputs from the output layer and the target values.  It then takes the mean value of this sum as the error to backpropagate.  Again this is 
done under the covers; the Pytorch library handles all the details.
</p>
<p>
The classification results vary depending on the class property distribution.  It is hard to separate classes whose distributions overlap one another.  And the degree to
which each property can vary (its standard deviation) makes it harder to predict.  When the batch size was increased, some of the classes %correct went to zero.
When the batch size was decreased to one, those classes %correct increased.  The large batch sizes smooth the learning curve variation but decreases resolution
and some classes are harder to detect due to their degree of overlap with other classes.  The weights and biases are not corrected during backpropagation for every samples when the batch
size is greater than one.  This means that the weight and bias corrections are done less frequently.  Increasing the number of batches with the batch size set to one does 
increase the %correct up to a certain value and then it levels off with no further increase.
</p>
<p>
This program uses PyTorch (without a GPU), Python 3.13, Matplotlib.Py, and Numpy.
</p>

<pre>
C:\Users\Thoma\eclipse-workspace\PyTorchTanhMSE\classify>linear_tanh_mse.py -h
usage: linear_tanh_mse [-h] {100,200,500,1000,2000,5000,10000} {1,2,10,16,32,64} {0.001,0.002,0.01,0.02} {0.5,0.6,0.7,0.8,0.9} {1,2,3} {10,15,20}

MLP neural network (Tanh, Mean Square Error Loss) for classification using PyTorch

positional arguments:
  {100,200,500,1000,2000,5000,10000}
                        the number of epochs for training and testing
  {1,2,10,16,32,64}     the batch size for training and testing
  {0.001,0.002,0.01,0.02}
                        learning rate hyperparameter for training
  {0.5,0.6,0.7,0.8,0.9}
                        momentum hyperparameter for training
  {1,2,3}               number of hidden layers, depth of MLP
  {10,15,20}            hidden layer width

options:
  -h, --help            show this help message and exit
</pre>

<h4>Class Property Distribution, Uniform, 10 properties, 4 classes</h4>
<img width="635" height="550" alt="image" src="https://github.com/user-attachments/assets/d7b2dc23-5794-4665-b9a3-5ca3e19c747e" />
<pre>
  C:\Users\Thoma\eclipse-workspace\PyTorchTanhMSE\classify>linear_tanh_mse.py 5000 1 .001 .5 3 15
Using cpu device
Model: NeuralNetwork(
  (linear_tanh_stack): Sequential(
    (0): Linear(in_features=10, out_features=15, bias=True)
    (1): Tanh()
    (2): Linear(in_features=15, out_features=15, bias=True)
    (3): Tanh()
    (4): Linear(in_features=15, out_features=15, bias=True)
    (5): Tanh()
    (6): Linear(in_features=15, out_features=2, bias=True)
  )
)
Display the class distributions
Begin training the model: 2026-11-09 10:29:44.539983
**************************************************
**************************************************
**************************************************
...
...
**************************************************
**************************************************
**************************************************

Done training: 2026-11-09 17:12:08.729776
Elapsed time: 6:42:24.189793
Display the Mean Square Error Loss
Saved PyTorch Model State to data\linearTanhMSE.pth
Begin testing the model: 2026-11-09 17:18:41.901699

Done testing.  2026-11-09 17:18:42.467584
Elapsed time: 0:00:00.565885 Test Results:
|--------|--------|--------|--------|
|Class   |Count   |Correct |%Correct|
|--------|--------|--------|--------|
|1       |1188    |1182    |99      |
|--------|--------|--------|--------|
|2       |1276    |1181    |92      |
|--------|--------|--------|--------|
|3       |1307    |942     |72      |
|--------|--------|--------|--------|
|4       |1229    |884     |71      |
|--------|--------|--------|--------|
|Totals  |5000    |4189    |84      |
|--------|--------|--------|--------|

</pre>
<h4>Learning Curve, 5000 batches, batch size = 1, learning rate = .001, momentum = 0.5, 3 hidden layers, 15 features per layer</h4>
<img width="636" height="548" alt="image" src="https://github.com/user-attachments/assets/7b619349-9efa-4ea1-9d2c-e76b7b099efd" />
<h4>Learning Curve, 5000 batches, batch size = 1, learning rate = .001, momentum = 0.5, 2 hidden layers, 20 features per layer</h4>

