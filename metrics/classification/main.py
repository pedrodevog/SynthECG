import argparse
from train import train_xresnet1d101

def main():
    parser = argparse.ArgumentParser(description='Train xresnet1d101 model on ECG data')
    parser.add_argument('--data_folder', type=str, required=True, help='Path to the data folder')
    parser.add_argument('--output_folder', type=str, required=True, help='Path to save the model and results')
    parser.add_argument('--task', type=str, default='superdiagnostic', 
                        choices=['diagnostic', 'subdiagnostic', 'superdiagnostic', 'form', 'rhythm', 'all', 'custom'],
                        help='Classification task type')
    parser.add_argument('--sampling_frequency', type=int, default=100, choices=[100, 500], 
                        help='ECG sampling frequency in Hz')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training')
    
    args = parser.parse_args()
    
    train_xresnet1d101(
        data_folder=args.data_folder,
        output_folder=args.output_folder,
        task=args.task,
        sampling_frequency=args.sampling_frequency,
        epochs=args.epochs,
        batch_size=args.batch_size
    )

if __name__ == "__main__":
    main()