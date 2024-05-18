# Assignment 4 - Detecting faces in historical newspapers

### Assignment Discription
In this assignment I have solved the following tasks:
- For each of the three newspapers
    - Go through each page and find how many faces are present
    - Group these results together by *decade* and then save the following:
        - A CSV showing the total number of faces per decade and the percentage of pages for that decade which have faces on them
        - A plot which shows the latter information - i.e. percentage of pages with faces per decade over all of the decades avaiable for that newspaper
- Repeat for the other newspapers

### Data
The data I have used in this assignment, is the dataset called Tobacco3482, which can be found [here](https://www.kaggle.com/datasets/patrickaudriaz/tobacco3482jpg?resource=download). You'll need to download it and unpack it the ```in``` folder in this repository. 

The model I have used to train the data is the model called VGG16, which you can find more information about [here](https://www.geeksforgeeks.org/vgg-16-cnn-model/).

### Reproducebility 
I have created a ```setup.sh``` file that can be run from the terminal using the code: 
```
bash setup.sh
``` 
When running it you create a virtual environment where you run the accompanying ```requirements.txt```. 

I have for this assignment created two different ```run.sh``` files that can be run from the terminal using the code:
```
bash run_data.sh
bash run_process.sh
```
Each file opens the virtual environment again, then runs one of the scripts that I have written for this assignment, and finishes off by deactivating the virtual environment. 

### Results
I have for this assignment created two ```.py``` scripts. One is called ```data.py```, in which I load the dataset and run the model on it. I have then createde a dataframe contining three columns "Newspaper","Year", "Faces". Which contains all the information needed to solve the rest of the tasks. and have saved that in the **out**. This script can be very timeconsuming to run, which is why I decided to save the new dataframe for later use.

The other script is called ```process.py```. In that script I load the processed data, so I don't have to run the classifier on the data every time I open the code. I can the create a CSV file for each newspaper, that shows per decade, the number of faces, number of pages, number of pages with faces and the percentage of pages containing faces. I have also created a plot for  each newspaper that shows percent of pages containing faces over time. Those files can be found in the **out** folder.

