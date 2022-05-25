import os
from PIL import Image
from datasets.utils import *


# ------------------------------------------
# BRATS dataset
class TestDataset(object):

    def __init__(self, dir_dataset, item, partition,  input_shape=(1, 224, 224), channel_first=True, norm='max',
                 histogram_matching=True, filter_volumes=False):
        # Init properties
        self.dir_dataset = dir_dataset
        self.dir_datasets = dir_dataset
        self.item = item
        self.partition = partition
        self.input_shape = input_shape
        self.channel_first = channel_first
        self.norm = norm
        self.histogram_matching = histogram_matching
        self.nchannels = self.input_shape[0]
        self.ref_image = {}
        self.filter_volumes = filter_volumes
        if 'BRATS' in dir_dataset:
            for iModality in np.arange(0, len(self.item)):
                x = Image.open(
                    '../data/BRATS_5slices/' + self.item[iModality] + '/train/benign/' + 'BraTS19_CBICA_AWV_1_77.jpg')
                x = np.asarray(x)
                self.ref_image[iModality] = x[40:-40, 40:-40]

        # Select all files in partition
        self.images = []
        for subdirs in ['benign', 'malign']:
            for images in os.listdir(self.dir_dataset + self.item[0] + '/' + self.partition + '/' + subdirs + '/'):
                if 'Thumbs.db' not in images:
                    self.images.append(self.dir_dataset + 'modality' + '/' + self.partition + '/' + subdirs + '/' + images)

        # Get number of patients (volumes), and number of slices
        if 'BRATS' in dir_dataset:
            patients = [image.split('/')[-1][:-7] for image in self.images]
        else:
            patients = [image.split('/')[-1][:-6] for image in self.images]
        self.unique_patients = np.unique(patients)
        slices_per_volume = len(patients) // len(self.unique_patients)

        # Load images and masks
        self.X = np.zeros((len(self.unique_patients), len(self.item), slices_per_volume, self.nchannels, self.input_shape[1], self.input_shape[2]))
        self.M = np.zeros((len(self.unique_patients), slices_per_volume, self.nchannels, self.input_shape[1], self.input_shape[2]))
        self.Y = np.zeros((len(self.unique_patients), slices_per_volume))
        for iPatient in np.arange(0, len(self.unique_patients)):

            slices_patient = list(np.sort([iSlice for iSlice in self.images if self.unique_patients[iPatient] in iSlice]))

            if 'ICH' in dir_dataset:
                indexes = np.array([int(id[-5]) for id in slices_patient])
                idx = np.array(np.argsort(indexes))
                slices_patient = [slices_patient[i] for i in idx]

            for iSlice in np.arange(0, slices_per_volume):

                    for iModality in np.arange(0, len(self.item)):

                        # Load image
                        x = Image.open(slices_patient[iSlice].replace('modality', self.item[iModality]))
                        x = np.asarray(x)

                        # Normalization
                        if 'BRATS' in dir_dataset:
                            x = image_normalization(x, self.input_shape[-1], norm=self.norm, channels=self.nchannels,
                                                    histogram_matching=self.histogram_matching, reference_image=self.ref_image[iModality],
                                                    mask=False, channel_first=True)
                        else:
                            x = image_normalization(x, self.input_shape[-1], norm=self.norm, channels=self.nchannels,
                                                    histogram_matching=False,
                                                    mask=False, channel_first=True)
                        self.X[iPatient, iModality, iSlice, :, :, :] = x

                        # Load mask
                        if 'malign' in slices_patient[iSlice]:
                            mask_id = slices_patient[iSlice].replace('malign', 'ground_truth').replace('modality', self.item[iModality])

                            m = Image.open(mask_id)
                            m = np.asarray(m)

                            # Normalization
                            m = image_normalization(m, self.input_shape[-1], norm=self.norm, channels=1,
                                                    histogram_matching=False,
                                                    reference_image=None,
                                                    mask=True, channel_first=True)
                            self.M[iPatient, iSlice, :, :, :] = m
                            self.Y[iPatient, iSlice] = 1

        if self.filter_volumes:
            idx = np.squeeze(np.argwhere(np.sum(self.M, (1, 2, 3, 4)) / (slices_per_volume*self.input_shape[1]*self.input_shape[2]) > 0.001))
            self.X = self.X[idx, :, :, :, :, :]
            self.M = self.M[idx, :, :, :, :]
            self.Y = self.Y[idx, :]
            self.images = list(np.array(self.images)[idx])
            self.unique_patients = list(np.array(self.unique_patients)[idx])

        if len(self.item) == 1:
            self.X = self.X[:, 0, :, :, :, :]


class MultiModalityDataset(object):

    def __init__(self, dir_datasets, modalities, input_shape=(3, 512, 512),  channel_first=True, norm='max',
                 hist_match=True, weak_supervision=False):

        'Internal states initialization'
        self.dir_datasets = dir_datasets
        self.modalities = modalities
        self.input_shape = input_shape
        self.channel_first = channel_first
        self.norm = norm
        self.nChannels = input_shape[0]
        self.hist_match = hist_match
        self.weak_supervision = weak_supervision
        self.ref_image = {}
        if 'BRATS' in dir_datasets:
            for iModality in np.arange(0, len(modalities)):
                x = Image.open('../data/BRATS_5slices/' + modalities[iModality] + '/train/benign/' + 'BraTS19_CBICA_AWV_1_77.jpg')
                x = np.asarray(x)
                self.ref_image[iModality] = x
        self.train_images = []
        self.test_images = []

        # Paths for training data
        name_normal = '/train/benign/'
        name_anomaly = '/test/malign/'

        # Get train images
        train_images = os.listdir(dir_datasets + modalities[0] + name_normal)
        # Remove other files
        train_images = [train_images[i] for i in range(train_images.__len__()) if train_images[i] != 'Thumbs.db']
        for iImage in train_images:
            self.train_images.append(dir_datasets + 'modality' + name_normal + iImage)

        # Get train images
        test_images = os.listdir(dir_datasets + modalities[0] + name_anomaly)
        # Remove other files
        test_images = [test_images[i] for i in range(test_images.__len__()) if test_images[i] != 'Thumbs.db']
        for iImage in test_images:
            self.test_images.append(dir_datasets + 'modality' + name_anomaly + iImage)

        self.train_indexes = np.arange(0, len(self.train_images))
        self.test_indexes = np.arange(0, len(self.test_images)) + len(self.train_images)
        self.images = self.train_images + self.test_images

        # Pre-allocate images
        self.X = np.zeros((len(self.images), len(self.modalities), input_shape[0], input_shape[1], input_shape[2]), dtype=np.float32)
        self.M = np.zeros((len(self.images), 1, input_shape[1], input_shape[2]), dtype=np.float32)
        self.Y = np.zeros((len(self.images), 2), dtype=np.float32)

        # Load, and normalize images
        print('[INFO]: Loading training images...')
        for i in np.arange(len(self.images)):
            for iModality in np.arange(0, len(self.modalities)):
                print(str(i) + '/' + str(len(self.images)), end='\r')

                # Load image
                x = Image.open(self.images[i].replace('modality', modalities[iModality]))
                x = np.asarray(x)

                # Normalization
                if 'BRATS' in dir_datasets:
                    x = image_normalization(x, self.input_shape[-1], norm=self.norm, channels=self.nChannels,
                                            histogram_matching=self.hist_match, reference_image=self.ref_image[iModality],
                                            mask=False, channel_first=True)
                else:
                    x = image_normalization(x, self.input_shape[-1], norm=self.norm, channels=self.nChannels,
                                            histogram_matching=False,
                                            mask=False, channel_first=True)
                self.X[i, iModality, :, :, :] = x

            if 'benign' in self.images[i]:
                self.Y[i, :] = np.array([1, 0])
            else:
                self.Y[i, :] = np.array([0, 1])

                mask_id = self.images[i].replace('malign', 'ground_truth').replace('modality', modalities[iModality])

                y = Image.open(mask_id)
                y = np.asarray(y)
                if len(y.shape) == 3:
                    y = y[:, :, 0]
                # Normalization
                y = image_normalization(y, self.input_shape[-1], norm=self.norm, channels=1,
                                        histogram_matching=False,
                                        reference_image=None,
                                        mask=True, channel_first=True)
                self.M[i, :, :, :] = y

        print('[INFO]: Images loaded')

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.train_indexes)

    def __getitem__(self, index):
        'Generates one sample of data'

        x = self.X[index, :, :, :, :]
        y = self.Y[index, :]

        if len(self.modalities) == 1:
            x = x[0, :, :, :]

        return x, y

# ------------------------------------------
# MVTEC dataset


class MVTECDataset(object):

    def __init__(self, dir_datasets, modalities, input_shape=(3, 512, 512),  channel_first=True, norm='max',
                 weak_supervision=False, partition='train'):

        'Internal states initialization'
        self.dir_datasets = dir_datasets
        self.modalities = modalities
        self.input_shape = input_shape
        self.channel_first = channel_first
        self.norm = norm
        self.nChannels = input_shape[0]
        self.weak_supervision = weak_supervision
        self.images = []

        # Get images
        categories = os.listdir(dir_datasets + modalities[0] + '/' + partition + '/')
        for i_category in categories:
            for iFile in os.listdir(dir_datasets + modalities[0] + '/' + partition + '/' + i_category + '/'):
                self.images.append(dir_datasets + modalities[0] + '/' + partition + '/' + i_category + '/' + iFile)

        # Remove other files
        self.images = [self.images[i] for i in range(self.images.__len__()) if 'Thumbs.db' not in self.images[i]]

        # Pre-allocate images
        self.X = np.zeros((len(self.images), input_shape[0], input_shape[1], input_shape[2]), dtype=np.float32)
        self.M = np.zeros((len(self.images), 1, input_shape[1], input_shape[2]), dtype=np.float32)
        self.Y = np.zeros((len(self.images), 1), dtype=np.float32)

        # Load, and normalize images
        print('[INFO]: Loading training images...')
        for i in np.arange(len(self.images)):
            print(str(i) + '/' + str(len(self.images)), end='\r')

            # Load image
            x = Image.open(self.images[i])
            x = np.asarray(x)

            # Normalization
            x = image_normalization(x, self.input_shape[-1], norm=self.norm, channels=self.nChannels,
                                    histogram_matching=False,
                                    mask=False, channel_first=True)
            self.X[i, :, :, :] = x

            if 'good' in self.images[i]:
                self.Y[i, :] = np.array([0])
            else:
                self.Y[i, :] = np.array([1])

                mask_id = self.images[i].replace(partition, 'ground_truth').replace('.png', '_mask.png')

                y = Image.open(mask_id)
                y = np.asarray(y)
                if len(y.shape) == 3:
                    y = y[:, :, 0]
                # Normalization
                y = image_normalization(y, self.input_shape[-1], norm=self.norm, channels=1,
                                        histogram_matching=False,
                                        reference_image=None,
                                        mask=True, channel_first=True)
                self.M[i, :, :, :] = y

        print('[INFO]: Images loaded')
        if partition == 'train':
            self.train_indexes = np.arange(0, len(self.images))

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.images)

    def __getitem__(self, index):
        'Generates one sample of data'

        x = self.X[index, :, :, :]
        y = self.Y[index, :]

        return x, y


# ------------------------------------------
# Data generator

class WSALDataGenerator(object):

    def __init__(self, dataset, partition, batch_size=16, shuffle=False):

        'Internal states initialization'
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.partition = partition

        if self.partition == 'train':
            self.indexes = self.dataset.train_indexes.copy()
        elif self.partition == 'test':
            self.indexes = self.dataset.test_indexes.copy()

        if self.dataset.weak_supervision:
            self.indexes_abnormal = self.dataset.test_indexes.copy()
            self._idx_abnormal = 0
            self.batch_size_anomaly = np.min(np.array((self.batch_size, len(self.dataset.test_indexes))))

        self._idx = 0
        self._reset()

    def __len__(self):

        N = len(self.indexes)
        b = self.batch_size
        return N // b

    def __iter__(self):

        return self

    def __next__(self):

        # If dataset is completed, stop iterator
        if self._idx + self.batch_size >= len(self.indexes):
            self._reset()
            raise StopIteration()

        if self.dataset.weak_supervision:
            if self._idx_abnormal + self.batch_size_anomaly >= len(self.indexes_abnormal):
                self._idx_abnormal = 0

        # Load images and include into the batch
        X, Y = [], []
        for i in range(self._idx, self._idx + self.batch_size):
            x, y = self.dataset.__getitem__(self.indexes[i])
            X.append(x)
            Y.append(y)
        # Update index iterator
        self._idx += self.batch_size

        if self.dataset.weak_supervision:
            Xa, Ya = [], []
            for i in range(self._idx_abnormal, self._idx_abnormal + self.batch_size_anomaly):
                xa, ya = self.dataset.__getitem__(self.indexes_abnormal[i])
                Xa.append(xa)
                Ya.append(ya)
            # Update index iterator
            self._idx_abnormal += self.batch_size_anomaly

            return np.array(X).astype('float32'), np.array(Y).astype('float32'),\
                   np.array(Xa).astype('float32'), np.array(Ya).astype('float32')
        else:
            return np.array(X).astype('float32'), np.array(Y).astype('float32'),\
                   None, None

    def _reset(self):

        if self.shuffle:
            random.shuffle(self.indexes)
        self._idx = 0


