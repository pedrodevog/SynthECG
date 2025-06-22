import numpy as np
from debug.SyntheticData import SyntheticData
from copy import deepcopy


class Sinusoid(SyntheticData):
    def __init__(self, num_classes=5, num_samples=500, num_channels=12, num_datapoints=1000):
        super().__init__(num_classes, num_samples, num_channels, num_datapoints)
        self._define_class_distributions()
        self.generate_data()

    def _define_class_distributions(self, A_mean=1, A_std=0, f_mean=1, f_std=0, phi_mean=0, phi_std=0):
        seed = 42
        np.random.seed(seed)
        for _ in range(self._num_classes):
            amplitude = {'mean': A_mean*np.random.rand(), 'std': A_std}
            # frequency = {'mean': np.random.rand() * 2/3 + 1, 'std': 0} # Frequency between 60bpm and 100bpm (1Hz to 5/3Hz)
            frequency = {'mean': f_mean, 'std': f_std}  
            phase = {'mean': 2 * np.pi * np.random.rand(), 'std': phi_std * np.pi}
            self._classes.append({
                'amplitude': amplitude,
                'frequency': frequency,
                'phase': phase
            })

    def copy(self):
        copied = Sinusoid(self._num_classes, self._num_samples, self._num_channels, self._num_datapoints)
        copied._classes = deepcopy(self._classes)
        copied._data = deepcopy(self._data)
        return copied

    def print_class_distributions(self):
        print(f'{"Class":<10} {"Amplitude (mean, std)":<30} {"Frequency (mean, std)":<30} {"Phase (mean, std)":<30}')
        print('-' * 100)
        for i, class_params in enumerate(self._classes):
            print(f'{i:<21} '
                  f'{class_params["amplitude"]["mean"]:.2f}, {class_params["amplitude"]["std"]:.2f}{"":<21}'
                  f'{class_params["frequency"]["mean"]:.2f}, {class_params["frequency"]["std"]:.2f}{"":<17}'
                  f'{class_params["phase"]["mean"] / np.pi :.2f}, {class_params["phase"]["std"]:.2f}')

    # @timeit
    def generate_data(self):
        time = np.linspace(0, self._num_datapoints / self._sample_rate, self._num_datapoints)

        for i, class_params in enumerate(self._classes):
            base_amplitude = np.random.normal(class_params['amplitude']['mean'], class_params['amplitude']['std'],
                                      (self._num_samples, 1))
            amplitude = np.linspace(-1, 1, self._num_channels) * base_amplitude
            frequency = np.random.normal(class_params['frequency']['mean'], class_params['frequency']['std'],
                                         (self._num_samples, self._num_channels))
            phase = np.random.uniform(
                class_params['phase']['mean'] - class_params['phase']['std'],
                class_params['phase']['mean'] + class_params['phase']['std'],
                (self._num_samples, self._num_channels)
            )

            self._data[i] = amplitude[:, :, np.newaxis] * np.sin(
                2 * np.pi * frequency[:, :, np.newaxis] * time + phase[:, :, np.newaxis])
    
    def phase_shift(self, phase_shift):
        """
        Create phase-shifted data by sampling from the same distributions with added phase
        Args:
            phase_shift: Phase shift in radians
        Returns:
            Phase-shifted version of the data with same shape
        """
        time = np.linspace(0, self._num_datapoints / self._sample_rate, self._num_datapoints)
        shifted = self.copy()
        
        # Make sure we create the data array with the proper shape
        # Assuming self._data shape is (num_classes, num_samples, num_channels, num_datapoints)
        shifted_data = np.zeros((len(self._classes), self._num_samples, self._num_channels, self._num_datapoints))

        for i, class_params in enumerate(self._classes):
            # Update the phase mean in the shifted object
            shifted._classes[i]['phase']['mean'] += phase_shift
            
            # Generate amplitude and frequency using the original distributions
            base_amplitude = np.random.normal(class_params['amplitude']['mean'], class_params['amplitude']['std'],
                                    (self._num_samples, 1))
            amplitude = np.linspace(-1, 1, self._num_channels) * base_amplitude
            frequency = np.random.normal(class_params['frequency']['mean'], class_params['frequency']['std'],
                                    (self._num_samples, self._num_channels))
            
            # Use the updated phase mean but with original std
            phase = np.random.normal(shifted._classes[i]['phase']['mean'], class_params['phase']['std'],
                                (self._num_samples, self._num_channels))

            # Generate the sinusoidal data for this class
            shifted_data[i] = amplitude[:, :, np.newaxis] * np.sin(
                2 * np.pi * frequency[:, :, np.newaxis] * time + phase[:, :, np.newaxis])
        
        # Update the data in the shifted object
        shifted._data = shifted_data

        return shifted
    


    def phase_width(self, phase_width):
        """
        Create sinusoid with same distributions except for phase width.
        
        Args:
            phase_width: The width of the uniform distribution for phase.
                        When 0, all phases will be exactly at their mean value.
                        When 2π, phases will range from [-π, π] around their mean.
        """
        time = np.linspace(0, self._num_datapoints / self._sample_rate, self._num_datapoints)
        shifted = self.copy()
        
        # Create a new data array with proper shape
        width_data = np.zeros((len(self._classes), self._num_samples, self._num_channels, self._num_datapoints))
        
        for i, class_params in enumerate(self._classes):
            # Update the phase width parameter in the shifted object
            shifted._classes[i]['phase']['std'] = phase_width
            
            # Create amplitude and frequency as before
            base_amplitude = np.random.normal(class_params['amplitude']['mean'], class_params['amplitude']['std'],
                                    (self._num_samples, 1))
            amplitude = np.linspace(-1, 1, self._num_channels) * base_amplitude
            frequency = np.random.normal(class_params['frequency']['mean'], class_params['frequency']['std'],
                                    (self._num_samples, self._num_channels))
            
            # Use the updated phase std value to generate phases
            # For uniform distribution, we need to scale by 2 to get the full width
            # phase_width of 2π should give uniform phases in range [-π,π] around the mean
            phase = np.random.uniform(
                class_params['phase']['mean'] - phase_width/2,
                class_params['phase']['mean'] + phase_width/2,
                (self._num_samples, self._num_channels)
            )

            # Generate the sinusoidal data for this class
            width_data[i] = amplitude[:, :, np.newaxis] * np.sin(
                2 * np.pi * frequency[:, :, np.newaxis] * time + phase[:, :, np.newaxis])
        
        # Update the data in the shifted object
        shifted._data = width_data

        return shifted

    
    def frequency_shift(self, frequency_shift):
        """
        Create frequency-shifted data by sampling from the same distributions with added frequency
        Args:
            frequency_shift: Frequency shift (amount to add to the mean frequency)
        Returns:
            Frequency-shifted version of the data with same shape
        """
        time = np.linspace(0, self._num_datapoints / self._sample_rate, self._num_datapoints)
        shifted = self.copy()
        
        # Make sure we create the data array with the proper shape
        # Assuming self._data shape is (num_classes, num_samples, num_channels, num_datapoints)
        shifted_data = np.zeros((len(self._classes), self._num_samples, self._num_channels, self._num_datapoints))

        for i, class_params in enumerate(self._classes):
            # Update the frequency mean in the shifted object
            shifted._classes[i]['frequency']['mean'] += frequency_shift
            
            # Generate amplitude using the original distributions
            base_amplitude = np.random.normal(class_params['amplitude']['mean'], class_params['amplitude']['std'],
                                    (self._num_samples, 1))
            amplitude = np.linspace(-1, 1, self._num_channels) * base_amplitude
            
            # Use the shifted frequency mean when generating the frequencies
            frequency = np.random.normal(shifted._classes[i]['frequency']['mean'], class_params['frequency']['std'],
                                    (self._num_samples, self._num_channels))
            
            # Generate phases using original distribution
            phase = np.random.normal(class_params['phase']['mean'], class_params['phase']['std'],
                                (self._num_samples, self._num_channels))

            # Generate the sinusoidal data for this class
            shifted_data[i] = amplitude[:, :, np.newaxis] * np.sin(
                2 * np.pi * frequency[:, :, np.newaxis] * time + phase[:, :, np.newaxis])
        
        # Update the data in the shifted object
        shifted._data = shifted_data

        return shifted
    

    def frequency_width(self, frequency_width):
        """
        Create sinusoid with same distributions except for frequency width.
        
        Args:
            frequency_width: The width of the normal distribution for frequency.
                        When 0, all frequencies will be exactly at their mean value.
                        When 1, frequencies will range from [0, 1] around their mean.
        """
        time = np.linspace(0, self._num_datapoints / self._sample_rate, self._num_datapoints)
        shifted = self.copy()
        
        # Create a new data array with proper shape
        width_data = np.zeros((len(self._classes), self._num_samples, self._num_channels, self._num_datapoints))
        
        for i, class_params in enumerate(self._classes):
            # Update the frequency width parameter in the shifted object
            shifted._classes[i]['frequency']['std'] = frequency_width
            
            # Create amplitude and phase as before
            base_amplitude = np.random.normal(class_params['amplitude']['mean'], class_params['amplitude']['std'],
                                    (self._num_samples, 1))
            amplitude = np.linspace(-1, 1, self._num_channels) * base_amplitude
            phase = np.random.uniform(
                class_params['phase']['mean'] - class_params['phase']['std']/2,
                class_params['phase']['mean'] + class_params['phase']['std']/2,
                (self._num_samples, self._num_channels)
            )
            
            # Use the updated frequency std value to generate frequencies with Gaussian distribution
            # frequency_width controls how much the frequencies vary around their mean
            frequency = np.random.normal(
                class_params['frequency']['mean'], 
                frequency_width,
                (self._num_samples, self._num_channels)
            )

            # Generate the sinusoidal data for this class
            width_data[i] = amplitude[:, :, np.newaxis] * np.sin(
                2 * np.pi * frequency[:, :, np.newaxis] * time + phase[:, :, np.newaxis])
        
        # Update the data in the shifted object
        shifted._data = width_data

        return shifted


if __name__ == '__main__':
    sinusoid = Sinusoid()
    sinusoid.plot_data()
    sinusoid.print_data_shape()
    sinusoid.print_class_distributions()
    sinusoid.save_data_as_npy('data/sinusoid/')
