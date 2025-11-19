from audioutils import AudioUtil
from torch.utils.data import DataLoader, Dataset, random_split
from torch import arange
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path



AudioUtilHandler = AudioUtil()
print_output = True

class visualize: 
    def __init__(self):
        pass

    def audio_waveform_png_raw(self, audio_file):
        #Open audio file to get signal and sample rate
        sig, sr = AudioUtilHandler.open(audio_file)

        #Convert to stereo channel if not already
        resig, new_sr = AudioUtilHandler.convert_to_new_channel((sig, sr),2)

        if print_output:
            print (f"Original Sample Rate: {sr}")
            print (f"Original Shape: {sig.shape}")

        #x asis as time in seconds
        num_samples = sig.shape[1]
        time = np.arange(0, num_samples) / sr

        #y axis as amplitude
        audiowave = resig.numpy()[0]

        #the plot 
        plt.figure(figsize=(5, 4), dpi=150)
        plt.plot(range(len(audiowave)), audiowave, linewidth=0.2, color="#1f77b4")
        max_amp = float(audiowave.max())
        pad = 0.1 * max_amp if max_amp > 0 else 0.05  # avoid zero-width axis

        ylim = max_amp + pad
        plt.ylim(-ylim, ylim)
        num_samples = len(audiowave)
        pad_x = int(0.02 * num_samples) or 1
        plt.xlim(-pad_x, num_samples + pad_x)
        plt.xlabel("audio samples")
        plt.ylabel("amplitude")
        plt.title("Audio Waveform raw ")
        plt.tight_layout()

        #save plot as png under audiowaves_plots folder    
        p = Path(audio_file)
        plots_dir = Path("audiowaves_plots")
        plots_dir.mkdir(exist_ok=True)

        outfile = plots_dir / f"{p.parent.name}_{p.stem}.png"
        plt.savefig(outfile, dpi=150, bbox_inches="tight")
        plt.show()




audio_visualizer = visualize()
#dogbark
audio_visualizer.audio_waveform_png_raw("dataset/fold1/197073-3-3-0.wav")

#siren 
#audio_visualizer.audio_waveform_png_raw("dataset/fold8/133473-8-0-3.wav")