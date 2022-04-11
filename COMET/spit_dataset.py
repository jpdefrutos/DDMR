from shutil import move, copy2
import os

OR_DIR = '/mnt/EncryptedData1/Users/javier/ext_datasets/COMET_dataset/OSLO_COMET_CT/Formatted_128x128x128'
val_split = '/mnt/EncryptedData1/Users/javier/ext_datasets/COMET_dataset/OSLO_COMET_CT/For_validation.txt'
test_split = '/mnt/EncryptedData1/Users/javier/ext_datasets/COMET_dataset/OSLO_COMET_CT/For_testing.txt'

# Create out dirs
os.makedirs(os.path.join(OR_DIR, 'train'), exist_ok=True)
os.makedirs(os.path.join(OR_DIR, 'validation'), exist_ok=True)
os.makedirs(os.path.join(OR_DIR, 'test'), exist_ok=True)

# Copy all to train and then split into validation and test
list_of_files = [os.path.join(OR_DIR, f) for f in os.listdir(OR_DIR) if f.endswith('.h5')]
list_of_files.sort()
for f in list_of_files:
    copy2(f, os.path.join(OR_DIR, 'train'))

#   Get the indices for the validation and test subsets
with open(val_split, 'r') as f:
    val_idcs = f.readlines()[0]
    val_idcs = [int(e) for e in val_idcs.split(',')]

with open(test_split, 'r') as f:
    test_indcs = f.readlines()[0]
    test_indcs = [int(e) for e in test_indcs.split(',')]

#   move the files from train to validation and test
for i in val_idcs:
    move(os.path.join(OR_DIR, 'train', '{:05d}_CT.h5'.format(i)), os.path.join(OR_DIR, 'validation'))
print('Done moving the validation subset.')

for i in test_indcs:
    move(os.path.join(OR_DIR, 'train', '{:05d}_CT.h5'.format(i)), os.path.join(OR_DIR, 'test'))
print('Done moving the validation subset.')

print('Done splitting the data')
print('Training samples: '+str(len(os.listdir(os.path.join(OR_DIR, 'train')))))
print('Validation samples: '+str(len(val_idcs)))
print('Test samples: '+str(len(test_indcs)))
