from audioutils import AudioUtil
from torch.utils.data import DataLoader, Dataset, random_split
from torch import arange
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB
from kaggledatahandler import KaggleDataHandler
from audiopreprocessing import SoundDS

spectrogram = SoundDS()
AudioUtilHandler = AudioUtil()
print_output = True

class visualize: 
    def __init__(self):
        pass



    ######## PLOTTING RAW AUDIO WAVES ###########
    def plot(self, audio_file, x_axis, y_axis):

        plt.figure(figsize=(5, 4), dpi=150)
        plt.plot(range(len(y_axis)), y_axis, linewidth=0.2, color="#1f77b4")
        max_amp = float(y_axis.max())
        pad = 0.1 * max_amp if max_amp > 0 else 0.05  # avoid zero-width axis

        ylim = max_amp + pad
        plt.ylim(-ylim, ylim)
        x_axis = len(y_axis)
        pad_x = int(0.02 * x_axis) or 1
        plt.xlim(-pad_x, x_axis + pad_x)
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

        if print_output:
            print(f"Audio waveform plot saved to: {outfile}")   

        plt.show()



    ####### GETTING SPECTROGRAM FOR AUDIO FILE ###########
    def audio_mel_spectrogram_png(self, audio_file_path, class_id ):
        #Open audio file to get signal and sample rate
        spec_image = spectrogram.__getitem__(audio_file_path, class_id)

        if print_output:
            print (f"audio_mel_spectrogram_png: Spectrogram Type: {type(spec_image)}")
            print (f"audio_mel_spectrogram_png: Spectrogram{spec_image}")
        return spec_image


    ######## GETTING AUDIOWAVES FROM AUDIOFILE (raw) ###########
    def audio_waveform_png_raw(self, audio_file):
        #Open audio file to get signal and sample rate
        sig, sr = AudioUtilHandler.open(audio_file)

        if print_output:
            print (f"audio_waveform_png_raw:  Sample Rate: {sr}")
            print (f"audio_waveform_png_raw: Sig.shape: {sig.shape}")
            print (f"audio_waveform_png_raw: Sig Type: {type(sig)}")
            print (f"audio_waveform_png_raw: Sig{sig}")

        # number of samples
        num_samples = sig.shape[1]

        #the x-axis as duration in seconds
        time = num_samples / sr
        
        if print_output:
            print (f"\naudio_waveform_png_raw: Number of samples: {num_samples}")
            print (f"audio_waveform_png_raw: Duration in seconds: {time}")




        #y axis as amplitude
        audiowave = sig.numpy()[0]

        self.plot(audio_file, time, audiowave)

        return num_samples, audiowave





audio_visualizer = visualize()
#dogbark
#audio_visualizer.audio_waveform_png_raw("dataset/fold1/197073-3-3-0.wav")

#siren 
#audio_visualizer.audio_waveform_png_raw("dataset/fold8/133473-8-0-3.wav")


audio_visualizer.audio_mel_spectrogram_png("dataset/fold1/180937-7-2-0.wav", 7)