import itertools
import os
import csv

class KaggleDataHandler():
    def __init__(self):
        pass

    # Creating the different combinations based on the training vs testing split ratio
    def create_split(self, target_test_folds):
        folds = [i for i in range(10)]
    
        target_test_fold_names = []
        for fold_number in target_test_folds:
            target_test_fold_names.append("fold"+str(fold_number+1))

        target_training_fold_names = []
        for fold_number in folds:
            if (fold_number in target_test_folds):
                continue
            target_training_fold_names.append("fold"+str(fold_number+1))

        #print("Test folds:", target_test_fold_names)
        #print("Training folds:", target_training_fold_names)
        return (target_test_fold_names, target_training_fold_names)

    # Creating datastructure that contains the filepath to the different .wav based on the combinations found
    def create_set(self, number_of_test_folds):
        combinations= []
        if number_of_test_folds > 1:
            combinations = list(itertools.combinations(range(10), number_of_test_folds))
        else:
            combinations = [i for i in range(10)]

        dataset_folds = []
        for combination in combinations:
            dataset_folds.append(self.create_split(combination))
        
        datasets_filepath_organized = []
        y_datasets_filepath_organized = []
        for dataset in dataset_folds:
            new_dataset = []
            test_folds = {}
            training_folds = {}

            y_new_dataset = []
            y_test_folds = {}
            y_training_folds = {}

            for i, test_fold in enumerate(dataset[0]):
                folder_path = f'dataset/{test_fold}'
                files = os.listdir(folder_path)
                file_paths = []
                for file in files:
                    file_path = folder_path+'/'+file
                    file_paths.append(file_path)
                    print("Wait1.....")
                    y_value = 1 
                    #self.get_class_id(file)
                test_folds[test_fold] = file_paths
                y_test_folds[test_fold] = y_value
            
            for i, training_fold in enumerate(dataset[1]):
                folder_path = f'dataset/{training_fold}'
                files = os.listdir(folder_path)
                file_paths = []
                for file in files:
                    file_path = folder_path+'/'+file
                    file_paths.append(file_path)
                    print("Wait2.....")
                    y_value = 1
                    #self.get_class_id(file)
                training_folds[training_fold] = file_paths 
                y_training_folds[training_fold] = y_value
                
            new_dataset.append((test_folds, training_folds))
            datasets_filepath_organized.append(new_dataset)
            y_new_dataset.append((y_test_folds, y_training_folds))
            y_datasets_filepath_organized.append(y_new_dataset)

            
        return datasets_filepath_organized, y_datasets_filepath_organized
    

    def get_class_id(self, slice_file_name):
        csv_path = "dataset/UrbanSound8K.csv"
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["slice_file_name"] == slice_file_name:
                    return int(row["classID"])
        return None

# EXAMPLE USE CASE
KDHandler = KaggleDataHandler()
number_of_test_folds = 2
datasets_filepath_organized, y_datasets_filepath_organized = KDHandler.create_set(number_of_test_folds)


for i, dataset in enumerate(datasets_filepath_organized):
    print(f'#{i}: ')
    for i, segment in enumerate(dataset):
        print("- Test folds")
        for k, fold in enumerate(segment[0]):
            print(f'############ {fold} ##########')
            for filepath in segment[0][fold]:
                filename = filepath.split("/")[-1]
                print(f'{filename}')
        print("- Training folds")
        for l, fold in enumerate(segment[1]):
            print(f'########### {fold} ###########')
            for filepath in segment[1][fold]:
                filename = filepath.split("/")[-1]
                print(f'{filepath}')
    print('\n')

"""
for i, dataset in enumerate(y_datasets_filepath_organized):
    print(f'#{i}: ')
    for i, segment in enumerate(dataset):
        print("- Test folds")
        for k, fold in enumerate(segment[0]):
            print(f'############ {fold} ##########')
            for classID in segment[0][fold]:
                print(classID)
        print("- Training folds")
        for l, fold in enumerate(segment[1]):
            print(f'########### {fold} ###########')
            for classID in segment[1][fold]:
                print(classID)
    print('\n')
"""

    
   


    

        
            

        


