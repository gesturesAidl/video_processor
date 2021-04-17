import pandas as pd

csv_dir = '/mnt/disks/disk-1/jester_dataset/dataset/csvs/' 

train = pd.read_csv(csv_dir+'jester-v1-train.csv',header= None,
                    names=['idx','Action'],sep=';')

validation = pd.read_csv(csv_dir+'jester-v1-validation.csv',header= None,
                         names=['idx','Action'],sep=';')


selected_classes = [
    'No gesture',
    'Doing other things',
    'Stop Sign',
    'Thumb Up',
    'Sliding Two Fingers Down',
    'Sliding Two Fingers Up',
    'Swiping Right',
    'Swiping Left',
    'Turning Hand Clockwise']

labels = pd.DataFrame(selected_classes,columns=['Actions'])

labels.to_csv(csv_dir+'labels.csv',index=False)

clean_train = train[train['Action'].isin(selected_classes)]
clean_val = validation[validation['Action'].isin(selected_classes)]

clean_train.to_csv(csv_dir+'train.csv',index=False)
clean_val.to_csv(csv_dir+'validation.csv',index=False)