from polars import fold
from audioutils import AudioUtil
import ffmpeg
from kaggledatahandler import KaggleDataHandler
from audiopreprocessing import SoundDS
import csv 
import pandas as pd

AudioUtilHandler = AudioUtil()

sig, sr = AudioUtilHandler.open("dataset/fold8/133473-8-0-3.wav")



kag_handler = KaggleDataHandler()
audiopreprocessor = SoundDS()

number_of_test_folds = 2
datasets_filepath_organized, y_datasets_filepath_organized = kag_handler.create_set(number_of_test_folds)




for i, dataset in enumerate(datasets_filepath_organized):
    print(f'#{i}: ')
    for j, segment in enumerate(dataset):
        print("- Test folds")
        for k, fold in enumerate(segment[0]):
            print(f'############ {fold} ##########')
            for f, filepath in enumerate(segment[0][fold]):
                class_id = y_datasets_filepath_organized[i][j][0][fold][f]
                spectrgram_test = audiopreprocessor.__getitem__(filepath, class_id)
                print(f"Spectrogram for test: {spectrgram_test} and class ID: {class_id}")

                filename = filepath.split("/")[-1]
                print(f'{filename}')
        print("- Training folds")
        for l, fold in enumerate(segment[1]):
            print(f'########### {fold} ###########')
            for f, filepath in enumerate(segment[1][fold]):
                class_id = y_datasets_filepath_organized[i][j][1][fold][f]
                spectrgram_train = audiopreprocessor.__getitem__(filepath, class_id)
                print(f"Spectrogram for train: {spectrgram_train} and class ID: {class_id}")

                filename = filepath.split("/")[-1]

                print(f'{filename}')
    print('\n')

