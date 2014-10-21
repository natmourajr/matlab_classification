matlab_classification
=====================
# Classification Example with Neural Networks
## Introduction
This tutorial has created to help the students of Signal Processing Laboratory (LPS) of Federal University of Rio de Janeiro (UFRJ). It doesn't have commercial objectives and it cannot be copy for other purpose. [Here](https://github.com/natmourajr/matlab_classification/blob/master/presentation/classificacao.pdf), we have a full classification presentation in PDF (in Portuguese)

## Neural Networks
In machine learning and related fields, artificial neural networks (ANNs) are computational models inspired by an animal's central nervous systems (in particular the brain), and are used to estimate or approximate functions that can depend on a large number of inputs and are generally unknown. Artificial neural networks are generally presented as systems of interconnected "neurons" which can compute values from inputs, and are capable of machine learning as well as pattern recognition thanks to their adaptive nature.


### Normalization
As ANNs have a non-linear activation functions (usually hyperbolic tangent), which has a saturation point, we have to normalize the input vector to fully explore the activation function non-linearity.

 Input Normalization
   The first normalization is esferic one. Esferic Normalization reach zero mean and unitary variance
 Output Normalization
   As Iris Dataset has Target Class in String format, we transform it in float number in [-1,1]

### Training Procsess
 The Training Proccess in MatLab Toolbox can be split in 3 parts (each part with its sets):

   train part (train set): Here, we update ANN weigths.
   test part (test set): Here, we check ANN training progress
   validation (val set): One of Stop Criteria.

## Classification
In machine learning and statistics, classification is the problem of identifying to which of a set of categories (sub-populations) a new observation belongs, on the basis of a training set of data containing observations (or instances) whose category membership is known. 

### Result analysis
In Classification, we have different ways to check (analyse) our results. Here, i will give 5 different examples:

1. Training Proccess Analysis - As we are training a ANN, we should show the training proccess.
![Example of ANN Training Proccess](https://github.com/natmourajr/matlab_classification/blob/master/picts/training_description.png)

2. SP Analysis - SP is a metric which try to balance Pd (Detection Probability) and Pf (False-Alarm Probability). Its formula is SP = sqrt( (sum (Ef_i)/N_{class}) * prod( Ef_i )^(1/N_{class}) )

3. Output Histograms Analysis - Checks if we have confusion zones in response (Qualitative Analysis)
![Example of Output Histograms](https://github.com/natmourajr/matlab_classification/blob/master/picts/histogram2class.png)

4. ROC Analysis - Receive Operating Curve - It's a curve which show every Pd value against every Pf value. We 
should analyse the ROC shape.
![Example of ROC (with SP value)](https://github.com/natmourajr/matlab_classification/blob/master/picts/roc.png)

5. Confusion Matrix Analysis - Show the correct answers and the wrong answers of classifier.
![Example of Confusion Matrix](https://github.com/natmourajr/matlab_classification/blob/master/picts/confusion.png)

## Examples

### run_classifier
Steps:

1. Data Aquisition

2. Normalization (data, targets)

3. Split Training Sets (train, test, validation)

4. Single Training Process

5. Result Analysis (Training Analysis, SP value, Output Histogram Analysis, ROC Analysis, Confusion Matrix Analysis)

### run_classifier_cross_validation
Steps:

1. Data Aquisition

2. Normalization (data, targets)

3. Split Training Sets (train, test, validation)

4. Training Process with Cross validation

5. Result Analysis (Training Analysis, SP value, Output Histogram Analysis, ROC Analysis, Confusion Matrix Analysis)

Cross-Validation: Cross-validation is a model validation technique for assessing how the results of a statistical analysis will generalize to an independent data set. It is mainly used in settings where the goal is prediction or classification, and one wants to estimate how accurately a model will perform in practice.  

### run_classifier_cross_validation_sweep_topology
Steps:

1. Data Aquisition

2. Normalization (data, targets)

3. Split Training Sets (train, test, validation)

4. Training Process with Cross validation and Topology Sweep

5. Result Analysis (Training Analysis, SP value, Output Histogram Analysis, ROC Analysis, Confusion Matrix Analysis)

Topology Sweep: Topology Sweep is a constructive method to discovery which topology is the best for a given problem. We add neuron to a single ANN hidden layer and perform a result analysis.

### run_classifier_multi_class
Steps

1. Data Aquisition

2. Normalization (data, targets)

3. Split Training Sets (train, test, validation)

4. Training Process with Cross validation

5. Result Analysis (Training Analysis, SP value, Confusion Matrix Analysis)

Multi-Class: Now we don't a Class-against-Non-Class problem. For 3 class problem, we should perform a different analysis, only check Training Analysis, SP value and Confusion Matrix (Output Histograms and ROC doesn't make sense for more than 2 class)

### run_classifier_multi_class_cross_validation
To do...

### run_classifier_multi_class_cross_validation_sweep_topology
To do...


