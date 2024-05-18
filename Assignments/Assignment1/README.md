# Assignment 1 - Building a simple image search algorithm

### Assignment Discription
In this assignment I have solved the following tasks for the given data:

Define a particular image that you want to work with
- For that image
  - Extract the colour histogram using ```OpenCV```
- Extract colour histograms for all of the **other* images in the data
- Compare the histogram of our chosen image to all of the other histograms 
  - For this, use the ```cv2.compareHist()``` function with the ```cv2.HISTCMP_CHISQR``` metric
- Find the five images which are most simlar to the target image
  - Save a CSV file to the folder called ```out```, showing the five most similar images and the distance metric:

|Filename|Distance
|---|---|
|target|0.0|
|filename1|---|
|filename2|---|

I have also tried to use the model VGG16 to see if that would give me a better result.

### Repository Structure
In this repository you'll find three subfolders.
- In ```in``` you'll upload the data that is being used in the code.
- In ```out``` you'll find the results the code have produced.
- In ```src``` you'll find the scripts of code written to solve the tasks given in the assignment.

I have also created a requirements.txt and a setup.sh file for you to run, for the setting up a virtual environment to run the code in. And I  have created .sh scripts to run the code from.

### Data
The data used in this assignment, is a dataset called **17 Category Flower Dataset**.
The data can be found [here](https://www.robots.ox.ac.uk/~vgg/data/flowers/17/). You'll need to download the folder called **17flowers.zip**  and unpack it in the ```in``` folder in this repository, and rename it **flowers**. 

For an extra part of the assignment I have tried using the model VGG16 instead. More info about the model can be found [here](https://www.geeksforgeeks.org/vgg-16-cnn-model/).

### Reproducebility 
I have created a ```setup.sh``` file that can be run from the terminal using the code: 
```
bash setup.sh
``` 
When running it you create a virtual environment where you run the accompanying ```requirements.txt```. 

I have for this assignment created two different ```run.sh``` files that can be run from the terminal using the code:
```
bash run.sh
bash run_vgg16.sh
```
Each file opens the virtual environment again, then runs one of the scripts that I have written for this assignment, and finishes off by deactivating the virtual environment. 

### Results
For both scripts I have created a dataframe with the top 6 most similar photos to the reference photo, where the one it finds the most similar is the actual reference phote, which it has also compared itself to. I have alså created a plot that shows the reference photo, and the two most similar photos to it. 

#### ```cv2.compareHist()```
For the first way of doing it, the results aren't exactly impressive. My reference photo is a white flower, and according to the the histograms, the most similar photo is one containing a purple flower.

#### ```VGG16```
When running the VGG16 model on the data instead. I then get som more convincing results. The most similar photo is one containing the same type of flower as the one on my reference photo. However, I'm a little surprised that the second most similar photo isn't the most similar, since that is the same type of flower and it is photographed from a similar angle as the reference photo. 
