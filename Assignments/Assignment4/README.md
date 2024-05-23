# Assignment 4 - Detecting faces in historical newspapers

## Assignment Discription
In this assignment I have solved the following tasks:
- For each of the three newspapers
    - Go through each page and find how many faces are present
    - Group these results together by *decade* and then save the following:
        - A CSV showing the total number of faces per decade and the percentage of pages for that decade which have faces on them
        - A plot which shows the latter information - i.e. percentage of pages with faces per decade over all of the decades avaiable for that newspaper
- Repeat for the other newspapers

## Repository Structure
In this repository you'll find three subfolders.
- In ```in``` you'll upload the data, that the code will run on.
- In ```out``` you'll find the results the code have produced.
- In ```src``` you'll find the scripts of code written to solve the tasks given in the assignment.

I have also created a ```requirements.txt``` and a ```setup.sh``` file for you to run, for the setting up a virtual environment to run the code in. And I  have created ```run.sh``` scripts to run the code from.

## Data
The data I have used  in this assignment is a corpus of historic Swiss newspapers: the Journal de Genève (JDG, 1826-1994); the Gazette de Lausanne (GDL, 1804-1991); and the Impartial (IMP, 1881-2017). You can read more about the data and download it [here](https://zenodo.org/records/3706863). You'll need to download the folder called **images.zip**. When you unpack it you have a folder called **images**, in that folder there is another folder called **images**. That is the folder you need to move it to the ```in``` folder you have created in this repository. 

For this assignmnet I have used a pretrained convolutional neural network (CNN) model, which you can find more information about [here](https://medium.com/@danushidk507/facenet-pytorch-pretrained-pytorch-face-detection-mtcnn-and-facial-recognition-b20af8771144).

## Reproducebility 
For this code to work, you need to be placed in the **Assignment4** folder in your terminal.

Once you have downloaded the data and put it in the correct folder, you are now ready to run the code.

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

## Results
I have for this assignment created two ```.py``` scripts. 

One is called ```data.py```, in which I load the dataset and run the model on it. I have then createde a dataframe contining three columns "Newspaper","Year", "Faces". Which contains all the information needed to solve the rest of the tasks, and have saved that in the **out**. This script can be very timeconsuming to run, which is why I decided to save the new dataframe in the **out** folder, and then use that in my second script.

The second script is called ```process.py```. In that script I load the processed data, so I don't have to run the classifier on the data every time I open the code. I can the create a CSV file for each newspaper, that shows per decade, the number of faces, number of pages, number of pages with faces and the percentage of pages containing faces. I have also created a plot for  each newspaper that shows percent of pages containing faces over time. Those files can be found in the **out** folder.

From the plots, you can tell that the percentage of pages containing faces increases over time, except for GDL that has a really tall spike at the beginning as well. The on with the most pages containing faces is IMP that ends up with 78% of the pages containing faces in the decade of 2010. For JDG it peks at 35 percent in the 1980's, but we also only have data up until the 1990's. If we had as current data for JGD as we have for IMP, it might very well have had a similar peak.

