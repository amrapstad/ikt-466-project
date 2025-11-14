import itertools
import os

class KaggleDataHandler():
    def __init__(self):
        pass

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
        for dataset in dataset_folds:
            new_dataset = []
            test_folds = {}
            training_folds = {}

            for i, test_fold in enumerate(dataset[0]):
                folder_path = f'dataset/{test_fold}'
                files = os.listdir(folder_path)
                file_paths = []
                for file in files:
                    file_path = folder_path+'/'+file
                    file_paths.append(file_path)
                test_folds[test_fold] = file_paths 
            
            for i, training_fold in enumerate(dataset[1]):
                folder_path = f'dataset/{training_fold}'
                files = os.listdir(folder_path)
                file_paths = []
                for file in files:
                    file_path = folder_path+'/'+file
                    file_paths.append(file_path)
                training_folds[training_fold] = file_paths 
                
            new_dataset.append((test_folds, training_folds))
            datasets_filepath_organized.append(new_dataset)
            
        return datasets_filepath_organized

# EXAMPLE USE CASE
KDHandler = KaggleDataHandler()
number_of_test_folds = 2
datasets_filepath_organized = KDHandler.create_set(number_of_test_folds)
for i, dataset in enumerate(datasets_filepath_organized):
    print(f'#{i}: ')
    for fold in dataset:
        print("- Test folds")
        print(fold[0].keys())
        print("- Training folds")
        print(fold[1].keys())
    print('\n')
    
   


    

        
            

        


