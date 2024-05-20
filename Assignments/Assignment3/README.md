# Assignment 3 - Document classification using pretrained image embeddings

### Assignment Discription
In this assignment I have solved the following tasks:
- Loads the Tobacco3482 data and generates labels for each image
- Train a classifier to predict document type based on visual features
- Present a classification report and learning curves for the trained classifier
- Your repository should also include a short description of what the classification report and learning curve show.

### Repository Structure
In this repository you'll find two subfolders.
- In ```out``` you'll find the results the code have produced.
- In ```src``` you'll find the scripts of code written to solve the tasks given in the assignment.

You'll need to create a folder called ```in```, to upload the data to, for the code to work.

I have also created a requirements.txt and a setup.sh file for you to run, for the setting up a virtual environment to run the code in. And I  have created .sh scripts to run the code from.

### Data
The data I have used in this assignment, is the dataset called Tobacco3482, which can be found [here](https://www.kaggle.com/datasets/patrickaudriaz/tobacco3482jpg?resource=download). You'll need to download it and unpack the  **.zip** file. When you unpack it you have a folder called **archive**, in that folder there is a folder called **tobacco3482-jpg**. With in that folder is another folder called **tobacco3482-jpg**. That is the folder you need to move it to the ```in``` folder you have created in this repository. 

The model I have used to train the data is the model called VGG16, which you can find more information about [here](https://www.geeksforgeeks.org/vgg-16-cnn-model/).

### Reproducebility 
Once you have downloaded the data and put it in the correct folder, you are now ready to run the code.

I have created a ```setup.sh``` file that can be run from the terminal using the code: 
```
bash setup.sh
``` 
When running it you create a virtual environment where you run the accompanying ```requirements.txt```. It also installs the model that is needed in the code.

I have also created a ```run.sh``` file that can be run from the terminal using the code:
```
bash run.sh
```
It opens the virtual environment again, then runs the script I have written to solve the task given, and finishes off by deactivating the virtual environment.

### Results
When running the code, the output is a classification report and a loss curve plot, which can be found in the **out** folder.

In this case I have run the code with 25 epochs. When looking at the classification report that gives me an accuracy of 69%, which I think is very good, considering the data is all different types of text and they are only in black and white.

However, when looking at the loss curve for the code is shows obvious signs of overfitting, which means that the model works really well on the data it has been trained on, but probably won't do very well on unseen data.

To avoid this, there might be other approaches that could give a better loss curve. I might have gotten a better result if I have done some data augmentation, whilst still using the VGG16 model. Or maybe I should have used a different model, that might have worked better on this particular dataset or even trained my own classifier on the data.