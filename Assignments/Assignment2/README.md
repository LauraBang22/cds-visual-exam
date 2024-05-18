# Assignment 2 - Classification benchmarks with Logistic Regression and Neural Networks

### Assignment Discription
In this assignment I have solved the following tasks for the given data:
- Load the Cifar10 dataset
- Preprocess the data (e.g. greyscale, normalize, reshape)
- Train a classifier on the data
    - A logistic regression classifier *and* a neural network classifier
- Save a classification report
- Save a plot of the loss curve during training

### Repository Structure
In this repository you'll find three subfolders.
- In ```out``` you'll find the results the code have produced.
- In ```src``` you'll find the scripts of code written to solve the tasks given in the assignment.

I have also created a requirements.txt and a setup.sh file for you to run, for the setting up a virtual environment to run the code in. And I  have created .sh scripts to run the code from.

### Data
The data used in this assignment, is a dataset called **Cifar10**.
The data can be found [here](https://www.cs.toronto.edu/~kriz/cifar.html). The data is loaded in the script.

### Reproducebility 
I have created a ```setup.sh``` file that can be run from the terminal using the code: 
```
bash setup.sh
``` 
When running it you create a virtual environment where you run the accompanying ```requirements.txt```. 

I have for this assignment created two different ```run.sh``` files that can be run from the terminal using the code:
```
bash run_logistic_regression.sh
bash run_neural_network.sh
```
Each file opens the virtual environment again, then runs one of the scripts that I have written for this assignment, and finishes off by deactivating the virtual environment. 

### Results
For both scripts I have created a classification report, and for the neural network classifier I have also plotted a loss curve.

#### Logistic regression
When training the logistic regression classifier, I get an accuracy of 32%. That is not particularly good. I could maybe get  better, if I tried training it with some different parameters in the classifier.

#### Neural network
For this classifier I have run a gridsearch to try and find the parameters that would generate the best results in the classification report. The parameters I have tested are these:
```
param_grid = {
        'activation': ['logistic', 'relu'],
        'hidden_layer_sizes': [(50,), (75,), (100,)],
        'max_iter': [1000, 2000, 3000]
        }
```

When I first ran the classifier I dit it with these parameters:
```
classifierMLP = MLPClassifier(activation = "logistic",
                           hidden_layer_sizes = (70,),
                           max_iter=1000,
                           random_state = 42).fit(X_train_final, y_train)
```
That gave me an accuracy of 31%. However, when changing the parameters to what the gridsearch gave as the best result, which was:
```
classifierMLP = MLPClassifier(activation = "relu",
                           hidden_layer_sizes = (100,),
                           max_iter=1000,
                           random_state = 42,
                           verbose=True).fit(X_train_final, y_train)
```
It gave an accuracy of 40%, which still is not great, but at least it  is better than 31%.


