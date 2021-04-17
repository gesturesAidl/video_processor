import pandas as pd

path_to_csvs_folder = "..."
csv_dir = path_to_csvs_folder + '/csvs/'
clean_csvs_folder = "/clean_csvs"

train = pd.read_csv(csv_dir+'train.csv', header=None, names=['idx', 'Action'], sep=';')
validation = pd.read_csv(csv_dir+'validation.csv', header=None, names=['idx', 'Action'], sep=';')

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
# Create directory @{clean_csvs_folder} if not exists
if not os.path.exists(path_to_csvs_folder + clean_csvs_folder):
    os.makedirs(path_to_csvs_folder + clean_csvs_folder)

# labels.csv
labels = pd.DataFrame(keep_labels, columns=['Actions'])
labels.to_csv(path_to_csvs_folder+clean_csvs_folder+'/'+'labels.csv', index=False)

# train.csv
clean_train = train[train['Action'].isin(keep_labels)]
clean_train.to_csv(path_to_csvs_folder+clean_csvs_folder+'/'+'train.csv', index=False)

# validation.csv
clean_val = validation[validation['Action'].isin(keep_labels)]
clean_val.to_csv(path_to_csvs_folder+clean_csvs_folder+'/'+'validation.csv', index=False)
