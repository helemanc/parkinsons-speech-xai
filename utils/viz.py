import os 
import torchaudio
import librosa 
import numpy as np 
import matplotlib.pyplot as plt
import torch 
def plot_spectrograms_with_mask(plot_dir, original, attr, target=None, prediction=None):
    """
    Plots the original spectrogram, the normalized mask, and the saliency map in a single figure.

    Parameters:
    - original: np.ndarray, the original spectrogram.
    - attr: np.ndarray, the attribute (mask) used for saliency calculation.
    """
    sample_rate = 16000
    # Normalize the mask (attr)
    mask_normalized = (attr - np.min(attr)) / (np.max(attr) - np.min(attr))

    # Create the saliency map by multiplying the original spectrogram with the normalized mask
    saliency_map = original * mask_normalized  # Element-wise multiplication

    # # Create a single figure with 3 subplots
    # fig, axs = plt.subplots(1, 3, figsize=(15, 5))  # Adjust the figsize as needed

    # # Expand dimensions for plotting first element in the batch
    # original = np.expand_dims(original.T, axis=-1)
    # mask_normalized = np.expand_dims(mask_normalized.T, axis=-1)
    # saliency_map = np.expand_dims(saliency_map.T, axis=-1)

    # # Plot the original spectrogram (flipped vertically)
    # cax0 = axs[0].imshow(original, aspect="auto", cmap="plasma", origin="lower")
    # axs[0].set_title("Original Spectrogram")
    # axs[0].set_ylabel("Frequency")
    # axs[0].set_xlabel("Time")
    # plt.colorbar(cax0, ax=axs[0])

    # # Plot the normalized mask (flipped vertically)
    # cax1 = axs[1].imshow(mask_normalized, aspect="auto", cmap="plasma", origin="lower")
    # axs[1].set_title("Normalized Mask")
    # axs[1].set_xlabel("Time")
    # plt.colorbar(cax1, ax=axs[1])

    # # Plot the saliency map (flipped vertically)
    # cax2 = axs[2].imshow(saliency_map, aspect="auto", cmap="plasma", origin="lower")
    # axs[2].set_title("Saliency Map")
    # axs[2].set_xlabel("Time")
    # plt.colorbar(cax2, ax=axs[2])

    # # Set tick marks for the time axis
    # # Choose specific ticks (e.g., every 100 units)
    # time_ticks = np.arange(0, original.shape[1], 100)  # Adjust based on your data
    # axs[0].set_xticks(time_ticks)
    # axs[1].set_xticks(time_ticks)
    # axs[2].set_xticks(time_ticks)

    # # Optionally, set frequency ticks (for y-axis)
    # frequency_ticks = np.arange(0, original.shape[0], 100)  # Adjust based on your data
    # axs[0].set_yticks(frequency_ticks)
    # axs[1].set_yticks(frequency_ticks)
    # axs[2].set_yticks(frequency_ticks)

    # # Adjust layout to prevent overlap
    # plt.tight_layout()

    # # Save the final figure as one image
    # plt.savefig(
    #     "combined_visualization.png", format="png", dpi=300, bbox_inches="tight"
    # )
    # plt.show()
    # Create the first figure with a linear y-axis

    original = np.transpose(original)
    mask_normalized = np.transpose(mask_normalized)
    saliency_map = np.transpose(saliency_map)

    hop_length_samples = 185
    win_length_samples = 371

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    # Plot the original spectrogram with a linear y-axis
    img1 = librosa.display.specshow(original, sr=sample_rate,  n_fft=1024, hop_length=hop_length_samples, win_length=win_length_samples,  x_axis="time", y_axis="linear", ax=axs[0], cmap="plasma")
    axs[0].set_title("Original Spectrogram (Linear Scale)")
    plt.colorbar(img1, ax=axs[0])

    # Plot the normalized mask with a linear y-axis
    img2 = librosa.display.specshow(mask_normalized, sr=sample_rate,  n_fft=1024, hop_length=hop_length_samples, win_length=win_length_samples,  x_axis="time", y_axis="linear", ax=axs[1], cmap="plasma")
    axs[1].set_title("Normalized Mask (Linear Scale)")
    plt.colorbar(img2, ax=axs[1])

    # Plot the saliency map with a linear y-axis
    img3 = librosa.display.specshow(saliency_map, sr=sample_rate,   n_fft=1024, hop_length=hop_length_samples, win_length=win_length_samples,  x_axis="time", y_axis="linear", ax=axs[2], cmap="plasma")
    axs[2].set_title("Saliency Map (Linear Scale)")
    plt.colorbar(img3, ax=axs[2])

    # set fig title
    fig.suptitle(f"Target {target} - Prediction {prediction}")
    plt.tight_layout(pad = 3.0)
    figname = os.path.join(plot_dir, f"combined_visualization_linear.png")
    plt.savefig(figname, format="png", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

    # # Create the second figure with a log y-axis
    fig, axs_log = plt.subplots(1, 3, figsize=(15, 5))

    # Plot the original spectrogram with a log y-axis
    img1 = librosa.display.specshow(original, sr=sample_rate,  n_fft=1024, hop_length=hop_length_samples, win_length=win_length_samples,  x_axis="time", y_axis="log", ax=axs_log[0], cmap="plasma")
    axs_log[0].set_title("Original Spectrogram (Log Scale)")
    plt.colorbar(img1, ax=axs_log[0])

    # Plot the normalized mask with a log y-axis
    img2 = librosa.display.specshow(mask_normalized, sr=sample_rate,   n_fft=1024, hop_length=hop_length_samples, win_length=win_length_samples,  x_axis="time", y_axis="log", ax=axs_log[1], cmap="plasma")
    axs_log[1].set_title("Normalized Mask (Log Scale)")
    plt.colorbar(img2, ax=axs_log[1])

    # Plot the saliency map with a log y-axis
    img3 = librosa.display.specshow(saliency_map, sr=sample_rate,  n_fft=1024,  hop_length=hop_length_samples, win_length=win_length_samples,  x_axis="time", y_axis="log", ax=axs_log[2], cmap="plasma")
    axs_log[2].set_title("Saliency Map (Log Scale)")
    plt.colorbar(img3, ax=axs_log[2])

    plt.tight_layout(pad=3.0)
    fig.suptitle(f"Target {target} - Prediction {prediction}")
    figname = os.path.join(plot_dir, f"combined_visualization_log.png")
    plt.savefig(figname, format="png", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

def save_interpretations_for_conditions(fold_dir, original, attr, phase, labels, predictions, tf, sample_rate=16000):

    #TODO: fix for batch_size == 1 [indices to retrieve original, attr, phase]

    # Track if we've already plotted for this batch
    plotted = False
    plot01 = plot10 = plot00 = plot11 = False  # Initialize plot conditions as not satisfied

    for i in range(len(labels)):
        if plotted:
            break

        target = int(labels[i])
        prediction_value = predictions[i][0]

        # Construct the folder path for each plot
        plot_dir = os.path.join(fold_dir, f"plot_batch_{target}_{prediction_value}")

        # Check if the plot already exists; if not, evaluate conditions
        if not os.path.exists(plot_dir):
            if ((target == 0 and prediction_value == 1) or 
                (target == 1 and prediction_value == 0) or 
                (target == 0 and prediction_value == 0) or 
                (target == 1 and prediction_value == 1)):
                
                # Create directory for plot if it doesn't exist
                os.makedirs(plot_dir, exist_ok=True)

                # Plot and save spectrogram with mask
                plot_spectrograms_with_mask(plot_dir, original[i], attr[i], target, prediction_value)

                # Save the original audio
                original_wav = tf.invert_stft_with_phase(
                    torch.tensor(original[i]).unsqueeze(0).to("cuda"), 
                    phase[i].unsqueeze(0).to("cuda")
                )
                torchaudio.save(os.path.join(plot_dir, "original.wav"), original_wav.cpu(), sample_rate=sample_rate)

                # Update flags based on target and prediction value
                if target == 0 and prediction_value == 1:
                    plot01 = True
                elif target == 1 and prediction_value == 0:
                    plot10 = True
                elif target == 0 and prediction_value == 0:
                    plot00 = True
                elif target == 1 and prediction_value == 1:
                    plot11 = True

                # Check if all four conditions are satisfied
                if plot01 and plot10 and plot00 and plot11:
                    plotted = True  # Set plotted to True once all plots are completed
                    break

        # Optional: Exit the loop if we’ve reached the end of the batch without fulfilling all conditions
        if i == len(labels) - 1 and not plotted:
            plotted = True


def save_spectrograms(net_input, output_folder, sample_rate=16000, hop_length_samples=185, win_length_samples=371):
    """
    Save the spectrograms of the given net_input to a temporary folder for visualization.

    Args:
    - net_input (Tensor): The input spectrograms to be saved, assumed to be a batch of spectrograms.
    - output_folder (str): The directory where spectrograms will be saved.
    - sample_rate (int, optional): The sample rate for the spectrograms. Default is 16000.
    - hop_length_samples (int, optional): The hop length in samples. Default is 185.
    - win_length_samples (int, optional): The window length in samples. Default is 371.
    """
    # Create the tmp directory if it doesn't exist
    fold_dir = os.path.join(output_folder, "tmp")
    os.makedirs(fold_dir, exist_ok=True)

    # Loop through each net_input (spectrogram)
    for i in range(len(net_input)):
        #random_number = np.random.randint(0, 1000)

        # Create a figure to plot the spectrogram
        fig, ax = plt.subplots(figsize=(10, 5))  # Only one plot area

        # Plot the spectrogram with a linear y-axis
        img = librosa.display.specshow(net_input[i].cpu().numpy().T, 
                                       sr=sample_rate, 
                                       n_fft=1024, 
                                       hop_length=hop_length_samples, 
                                       win_length=win_length_samples, 
                                       x_axis="time", 
                                       y_axis="linear", 
                                       ax=ax, cmap="plasma")
        ax.set_title("Original Spectrogram (Linear Scale)")
        plt.colorbar(img, ax=ax)

        # Adjust layout and save the figure
        plt.tight_layout(pad=3.0)
        figname = os.path.join(fold_dir, f"single_spectrogram_linear_{i}.png")
        plt.savefig(figname, format="png", dpi=300, bbox_inches="tight")
        plt.show()
        plt.close()


