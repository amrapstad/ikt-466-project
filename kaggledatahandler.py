import itertools
import os
import csv

class KaggleDataHandler():
    def __init__(self):
        if not os.path.exists('processed_data'):
            os.makedirs('processed_data')
            self.create_file_to_label_csv()
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
    
    def create_file_to_label_csv(self):
        folds_name = []
        folds = [i for i in range(10)]
        for fold_number in folds:
            folds_name.append("fold"+str(fold_number+1))

        for i, fold_name in enumerate(folds_name):
            folder_path = f'dataset/{fold_name}'
            files = sorted(os.listdir(folder_path), key=lambda f: int(f.split("-")[0]))
            file_paths = []
            with open(f'processed_data/{fold_name}.csv', 'w', newline='') as outfile:
                writer = csv.writer(outfile)
                writer.writerow(['file_name', 'classID'])
                for file in files:
                    y_value = self.get_class_id(file)
                    writer.writerow([file, y_value])

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
                y_values = []
                for file in files:
                    file_path = folder_path+'/'+file
                    file_paths.append(file_path)
                    print("Wait1.....")
                    y_value = self.get_class_id_pre_data(test_fold,file)
                    y_values.append(y_value)
                test_folds[test_fold] = file_paths
                y_test_folds[test_fold] = y_values
            
            for i, training_fold in enumerate(dataset[1]):
                folder_path = f'dataset/{training_fold}'
                files = os.listdir(folder_path)
                file_paths = []
                y_values = []
                for file in files:
                    file_path = folder_path+'/'+file
                    file_paths.append(file_path)
                    print("Wait2.....")
                    y_value = self.get_class_id_pre_data(training_fold,file)
                    y_values.append(y_value)
                training_folds[training_fold] = file_paths 
                y_training_folds[training_fold] = y_values
                
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

    def get_class_id_pre_data(self, fold_name,slice_file_name):
        csv_path = f'processed_data/{fold_name}.csv'
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["file_name"] == slice_file_name:
                    return int(row["classID"])
        return None

# EXAMPLE USE CASE
#KDHandler = KaggleDataHandler()
#number_of_test_folds = 2
#datasets_filepath_organized, y_datasets_filepath_organized = KDHandler.create_set(number_of_test_folds)
#KDHandler.create_file_to_label_csv()



"""
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
                print(f'{filename}')
    print('\n')
"""

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


    

        
            

        