import pandas as pd
import os
import matplotlib.pyplot as plt

def load_data():
    '''
    This function loads the CSV file, created by the data.py script and returns it.
    '''
    filename = os.path.join("out", "data1.csv")
    data = pd.read_csv(filename)
    data = data.drop('Unnamed: 0', axis=1)
    return data

def data_analysis(data):
    '''
    This function analyzes the data, and creates both a csv file and a plot for every newspaper
    showing the percent of pages containg faces.
    I saves the result in the out folder.
    '''
    newspaper_list = ['GDL', 'IMP', 'JDG']

    for newspaper in newspaper_list:
        data_slice = data[data['Newspaper']==newspaper]

        data_slice.loc[:, 'Decade'] = (data_slice['Year'] // 10) * 10

        decade_groups = data_slice.groupby('Decade')

        num_of_faces = decade_groups['Faces'].sum().rename('Num_of_faces')

        num_of_pages = decade_groups['Faces'].size().rename('Num_of_pages')

        data_filtered = data_slice[data_slice['Faces'] > 0]

        result = data_filtered.groupby('Decade').size()

        data_result = pd.DataFrame(result, columns=['Pages_with_faces'])

        combined_data = pd.concat([num_of_faces, num_of_pages, data_result], axis=1)

        combined_data['Pct_of_pages'] = (combined_data['Pages_with_faces'] / combined_data['Num_of_pages']) * 100

        data_combined = combined_data.reset_index().fillna(0)
        
        outpath = os.path.join("out", newspaper+".csv")
        data_combined.to_csv(outpath)

        data_combined.plot(x='Decade', y='Pct_of_pages')
        plt.title("Line plot of percent of pages containing faces over time (" + newspaper+")")
        plt.xlabel("Decade")
        plt.ylabel("Percent of pages containing faces")
        plt.savefig("out/"+ newspaper+".png")

def main():
    data = load_data()
    data_analysis(data)

if __name__ == "__main__":
    main()