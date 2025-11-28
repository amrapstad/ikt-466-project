from audioutils import AudioUtil
from torch.utils.data import DataLoader, Dataset, random_split
from torch import arange
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB
from kaggledatahandler import KaggleDataHandler
from audiopreprocessing import SoundDS
from scipy import signal
from scipy.io import wavfile
import torch
from kaggledatahandler import KaggleDataHandler

spectrogram_process = SoundDS()
AudioUtilHandler = AudioUtil()
data_handler = KaggleDataHandler()

print_output = True


class visualize: 
    def __init__(self):
        pass
    '''
    ######## PLOTTING NORMALIZED AUDIO WAVES WITH X AXIS AS AUDIO SAMPLES ###########
    def plot(self, audio_file, x_axis, y_axis, type):
        plt.figure(figsize=(7, 4), dpi=150)
        plt.plot(range(len(y_axis)), y_axis, linewidth=0.2, color="#1f77b4")
        max_amp = float(y_axis.max())

        pad = 0.1 * max_amp if max_amp > 0 else 0.05 

        ylim = max_amp + pad
        plt.ylim(-ylim, ylim)
        x_axis = len(y_axis)
        pad_x = int(0.02 * x_axis) or 1
        plt.xlim(-pad_x, x_axis + pad_x)
        plt.xlabel("audio samples")
        plt.ylabel("amplitude")
        plt.title( type + " audio waveform for " + audio_file)
        plt.tight_layout()

        #save plot as png under plots_audiowaves/<fold>/foldX_filename.png    
        p = Path(audio_file)
        fold = p.parent.name
        plots_dir = Path("plots_audiowaves") / fold
        plots_dir.mkdir(parents=True, exist_ok=True)
        outfile = plots_dir / f"{fold}_{p.stem}_{type}.png"
        plt.savefig(outfile, dpi=150, bbox_inches="tight")

        if print_output:
            print(f"Audio waveform plot saved to: {outfile}")
        '''
    
    def plot(self, audio_file, duration_sec, y_axis, kind):
        plt.figure(figsize=(7, 4), dpi=150)
        t = np.linspace(0, duration_sec, num=len(y_axis), endpoint=False)  # seconds
        plt.plot(t, y_axis, linewidth=0.2, color="#1f77b4")

        max_amp = float(y_axis.max())
        pad = 0.1 * max_amp if max_amp > 0 else 0.05
        ylim = max_amp + pad
        plt.ylim(-ylim, ylim)

        pad_x = 0.02 * duration_sec if duration_sec else 0.01
        plt.xlim(-pad_x, duration_sec + pad_x)
        plt.xlabel("time (s)")
        plt.ylabel("amplitude")
        plt.title(f"{kind} audio waveform for {audio_file}")
        plt.tight_layout()

        p = Path(audio_file)
        fold = p.parent.name
        plots_dir = Path("plots_audiowaves") / fold
        plots_dir.mkdir(parents=True, exist_ok=True)
        outfile = plots_dir / f"{fold}_{p.stem}_{kind}.png"
        plt.savefig(outfile, dpi=150, bbox_inches="tight")
        if print_output:
            print(f"Audio waveform plot saved to: {outfile}")
            



    ######## GETTING AUDIOWAVES FROM AUDIOFILE (raw) ###########
    def audio_waveform_png_preprocessed(self, audio_file):
        ###### normalizing audio #######
        audio = AudioUtilHandler.open(audio_file)
        resample_audio = AudioUtil.resample((audio), 44100)
        stereo_audio = AudioUtil.convert_to_new_channel(resample_audio, 2)
        padded_audio = AudioUtil.padding_truncate(stereo_audio, 4000)
        audio = AudioUtil.padding_truncate(padded_audio, 4000)

        sig, sr = audio #signal and sample rate
        channel_type = sig.shape[0] #channel of audio 
        num_samples = sig.shape[1] #number of samples  , x axis
        audio_duration = num_samples / sr #audio duration (s)
        
        if print_output:
            print (f"\audio_waveform_png_preprocessed:  Sample Rate: {sr}")
            print (f"audio_waveform_png_preprocessed: stereo audio stero or mono: {channel_type} channels")
            print (f"audio_waveform_png_preprocessed: audio duration in seconds: {audio_duration} seconds")
            print (f"audio_waveform_png_preprocessed: Number of samples: {num_samples}")
            print (f"audio_waveform_png_preprocessed: Duration in seconds: {audio_duration}")

        #y axis as amplitude
        audiowave = sig.numpy()[0]

        self.plot(audio_file, audio_duration, audiowave, "normalized preprocessed")
        #self.plot_time_axis_as_seconds(audio_file, audio_duration, audiowave, "normalized preprocessed")
        return num_samples, audiowave





    ######## GETTING AUDIOWAVES FROM AUDIOFILE (raw) ###########
    def audio_waveform_png_raw(self, audio_file):
        ###### normalizing audio #######
        sig, sr  = AudioUtilHandler.open(audio_file)

        channel_type = sig.shape[0] #channel of audio 
        num_samples = sig.shape[1] #number of samples  , x axis
        audio_duration = num_samples / sr #audio duration (s)
        
        if print_output:
            print (f"\naudio_waveform_png_raw:  Sample Rate: {sr}")
            print (f"audio_waveform_png_raw: stereo audio stero or mono: {channel_type} channels")
            print (f"audio_waveform_png_raw: audio duration in seconds: {audio_duration} seconds")
            print (f"audio_waveform_png_raw: Number of samples: {num_samples}")
            print (f"audio_waveform_png_raw: Duration in seconds: {audio_duration}")

        #y axis as amplitude
        audiowave = sig.numpy()[0]

        self.plot(audio_file, audio_duration, audiowave, "raw")

        return num_samples, audiowave



        






    def plot_mel_spectrogram_png(self, input, title, audio_path):
        
        plt.figure(figsize=(7,4), dpi=150)
        plt.title(f"{title} Mel Spectrogram for {Path(audio_path).name}")
        plt.imshow(input, origin="lower", aspect="auto", cmap="magma")
        plt.colorbar(label="Intensity (dB)")
        plt.xlabel("Time Frames")
        plt.ylabel("Mel Frequency Bins")
        plt.tight_layout()
        plt.show()

        # Save plot as png under plots_mel_spectrograms/<fold>/foldX_filename.png    
        p = Path(audio_path)
        fold = p.parent.name
        plots_dir = Path("plots_mel_spectrograms") / fold
        plots_dir.mkdir(parents=True, exist_ok=True)
        outfile = plots_dir / f"{fold}_{p.stem}_{title}_spectrogram.png"
        plt.savefig(outfile, dpi=150, bbox_inches="tight")






    #get the augmented spectrgram from the preprocessed audio file and plot it as png
    def audio_mel_spectrogram_png(self, audio_path):

        sliced_file_name = Path(audio_path).name
        class_id = data_handler.get_class_id(sliced_file_name)
        if print_output:
            print(f"Class ID for {audio_path} is {class_id}")

        augmented_spectrogram, _ = spectrogram_process.__getitem__(audio_path, class_id)
        aug_spec_np = augmented_spectrogram[0].numpy()

        self.plot_mel_spectrogram_png(aug_spec_np, "preprocessed", audio_path)
        return aug_spec_np



    #get the mel spectrgram from the raw audio file not preprocessed and plot it as png
    def audio_mel_spectrogram_raw_png(self, audio_path):
        mel_spectrogram_transform = MelSpectrogram(sample_rate=44100, n_fft=2048, hop_length=512, n_mels=128)
        amplitude_to_db_transform = AmplitudeToDB()

        sig, sr = AudioUtilHandler.open(audio_path)
        mel_spectrogram = mel_spectrogram_transform(sig)
        mel_spectrogram_db = amplitude_to_db_transform(mel_spectrogram)
        mel_spectrogram_db_np = mel_spectrogram_db.numpy()[0]

        self.plot_mel_spectrogram_png(mel_spectrogram_db_np, "raw", audio_path)
        return mel_spectrogram_db_np


audio_visualizer = visualize()

file = "dataset/fold1/197073-3-3-0.wav"

#dogbark
#audio_visualizer.audio_waveform_png_preprocessed(file)
#audio_visualizer.audio_waveform_png_raw(file)
#audio_visualizer.audio_mel_spectrogram_raw_png(file)
#audio_visualizer.audio_mel_spectrogram_png(file)



#siren 
#audio_visualizer.audio_waveform_png_preprocessed("dataset/fold8/133473-8-0-3.wav")

#audio_visualizer.audio_waveform_png_preprocessed("dataset/fold1/180937-7-2-0.wav")
#audio_visualizer.audio_mel_spectrogram_png("dataset/fold1/180937-7-2-0.wav", 7)