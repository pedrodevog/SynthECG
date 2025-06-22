# Debug/SyntheticData.py
import numpy as np
import matplotlib.pyplot as plt
from utils.utils import timeit
import os


class SyntheticData:
    def __init__(self, num_classes, num_samples, num_channels, num_datapoints):
        self._sample_rate = 100
        self._num_classes = num_classes
        self._num_samples = num_samples
        self._num_channels = num_channels
        self._num_datapoints = num_datapoints

        self._classes = []
        self._data = np.zeros((self._num_classes, self._num_samples, self._num_channels, self._num_datapoints))

    def print_data_shape(self):
        print(f'Data shape (K x S x C x L):\n'
              f'K: {self._data.shape[0]:<10} (number of classes)\n'
              f'S: {self._data.shape[1]:<10} (number of samples)\n'
              f'C: {self._data.shape[2]:<10} (number of channels)\n'
              f'L: {self._data.shape[3]:<10} (length of each sample)\n')

    def print_class_distributions(self):
        raise NotImplementedError("Subclasses should implement this method.")

    def generate_data(self):
        raise NotImplementedError("Subclasses should implement this method.")

    # @timeit
    def get_data_split(self, train_size=0.8, val_size=0.1, test_size=0.1, random_seed=42, verbose=False):
        # Set random seed for reproducibility
        np.random.seed(random_seed)
        
        # Flatten the data: assume self._data is organized per class,
        # and each class has self._num_samples samples.
        data = self._data.reshape(self._num_classes * self._num_samples,
                                self._num_channels, self._num_datapoints)
        
        # Create labels: each class label repeated for its samples.
        labels = np.repeat(np.arange(self._num_classes), self._num_samples)
        
        # Shuffle data and labels together.
        indices = np.random.permutation(len(labels))
        data = data[indices]
        labels = labels[indices]
        
        # One-hot encode the labels.
        onehot_labels = np.eye(self._num_classes)[labels]
        
        # Determine split sizes.
        total_samples = len(labels)
        num_train = int(total_samples * train_size)
        num_val = int(total_samples * val_size)
        
        # Split the data.
        train_data = data[:num_train]
        train_labels = onehot_labels[:num_train]
        train_class_labels = labels[:num_train]
        
        val_data = data[num_train:num_train + num_val]
        val_labels = onehot_labels[num_train:num_train + num_val]
        val_class_labels = labels[num_train:num_train + num_val]
        
        test_data = data[num_train + num_val:]
        test_labels = onehot_labels[num_train + num_val:]
        test_class_labels = labels[num_train + num_val:]
        
        if verbose:
            # Print the number of samples per class in each split.
            print("Train split sample counts per class:")
            for c in range(self._num_classes):
                count = np.sum(train_class_labels == c)
                print(f"  Class {c}: {count}")
            
            print("\nValidation split sample counts per class:")
            for c in range(self._num_classes):
                count = np.sum(val_class_labels == c)
                print(f"  Class {c}: {count}")
            
            print("\nTest split sample counts per class:")
            for c in range(self._num_classes):
                count = np.sum(test_class_labels == c)
                print(f"  Class {c}: {count}")
        
        return (train_data, train_labels), (val_data, val_labels), (test_data, test_labels)


    @timeit
    def plot_data(self, name='plot'):
        if self._num_classes == 1 and self._num_channels == 1:
            fig, axs = plt.subplots(1, 1, figsize=(15, 2))
            axs = np.array([[axs]])  # Make 2D array with single element
        elif self._num_classes == 1:
            fig, axs = plt.subplots(self._num_channels, 1, figsize=(15, 2 * self._num_channels), sharex=True, sharey=True)
            axs = axs.reshape(-1, 1)  # Reshape to (n_channels, 1)
        elif self._num_channels == 1:
            fig, axs = plt.subplots(1, self._num_classes, figsize=(15, 2), sharex=True, sharey=True)
            axs = axs.reshape(1, -1)  # Reshape to (1, n_classes)
        else:
            fig, axs = plt.subplots(self._num_channels, self._num_classes, figsize=(15, 2 * self._num_channels), sharex=True, sharey=True)

        time_array = np.linspace(0, self._num_datapoints / self._sample_rate, self._num_datapoints)
        for i in range(self._num_classes):
            for j in range(self._num_channels):
                mean = np.mean(self._data[i, :, j, :], axis=0)
                std = np.std(self._data[i, :, j, :], axis=0)
                axs[j, i].plot(time_array, mean)
                axs[j, i].fill_between(time_array, mean - std, mean + std, alpha=0.3)
                if j == 0:
                    axs[j, i].set_title(f'Class {i}')
                if i == 0:
                    axs[j, i].set_ylabel(f'Channel {j} (mV)')

        axs[-1, 0].set_xlabel('Time (s)')
        plt.tight_layout()
        plt.savefig(f"plot_{name}.png")  # Save to check if it's being generated
        plt.show()

    def save_data_as_npy(self, directory):
        # Get data split
        (train_data, train_labels), (val_data, val_labels), (test_data, test_labels) = self.get_data_split()

        # Create directory if it doesn't exist
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        # Save data
        np.save(os.path.join(directory, 'train_data'), train_data)
        np.save(os.path.join(directory, 'train_labels'), train_labels)
        np.save(os.path.join(directory, 'val_data'), val_data)
        np.save(os.path.join(directory, 'val_labels'), val_labels)
        np.save(os.path.join(directory, 'test_data'), test_data)
        np.save(os.path.join(directory, 'test_labels'), test_labels)
