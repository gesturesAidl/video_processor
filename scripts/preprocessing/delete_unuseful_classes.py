import pandas as pd
import os
import shutil


abs_path_to_csvs = "..."
abs_path_to_dataset_folder = "..."

validate_csv = 'validation.csv'
test_csv = 'test.csv'
train_csv = 'train.csv'
labels_csv = 'labels.csv'
csv_dir = abs_path_to_csvs + '/csvs/'
dataset_path = abs_path_to_dataset_folder + '/20bn-jester-v1/'

test_set_output_dir = 'test_set'

tot_labels = pd.read_csv(csv_dir+labels_csv, header=None).iloc[:, 0].to_list() # List of labels in dataset
train_df = pd.read_csv(csv_dir+train_csv, header=None, names=["folder", "action"], sep=';')
validate_df = pd.read_csv(csv_dir+validate_csv, header=None, names=["folder", "action"], sep=';')
test_csv = pd.read_csv(csv_dir+test_csv, header=None, names=["folder", "action"], sep=';')


keep_labels = [
    'No gesture',
    'Doing other things',
    'Stop Sign',
    'Thumb Up',
    'Sliding Two Fingers Down',
    'Sliding Two Fingers Up',
    'Swiping Right',
    'Swiping Left',
    'Turning Hand Clockwise'
]


# Go through all directories in {@dataset_path}
for x in os.walk(dataset_path):
    list_dirs = x[1]
    break

# For each row in dataset (train) delete all videos from classes not in @{keep_labels}
for index, row in train_df.iterrows():
    if not (row['action'] in keep_labels):
        # Delete folder
        try:
            print("del dir: " + str(row['folder']))
            shutil.rmtree(dataset_path + str(row['folder']), ignore_errors=True)
        except:
            print("Unable deleting folder with name: " + str(ror['folder']) + ". cause: Doesn't exist")

# For each row in dataset (validation) delete all videos from classes not in @{keep_labels}
for index, row in validate_df.iterrows():
    if not (row['action'] in keep_labels):
        # Delete folder
        try:
            print("del dir: " + str(row['folder']))
            shutil.rmtree(dataset_path + str(row['folder']), ignore_errors=True)
        except:
            print("Unable deleting folder with name: " + str(ror['folder']) + ". cause: Doesn't exist")

# Create directory in @{abs_path_to_dataset_folder} with name @{test_set_output_dir} to append the test set
if not os.path.exists(abs_path_to_dataset_folder + test_set_output_dir):
    os.makedirs(abs_path_to_dataset_folder + test_set_output_dir)

# For each row in dataset (test) move each test video to @{test_set_output_dir}
for index, row in validate_df.iterrows():
    try:
        print("mv dir: " + str(row['folder']))
        shutil.move(src=dataset_path + str(row['folder']), dst=abs_path_to_dataset_folder + test_set_output_dir)
    except:
        print("Unable to move folder with name: " + str(ror['folder']) + ". cause: Doesn't exist")

