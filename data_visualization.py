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

spectrogram_process = SoundDS()
AudioUtilHandler = AudioUtil()
print_output = True

class visualize: 
    def __init__(self):
        pass

    ######## PLOTTING NORMALIZED AUDIO WAVES ###########
    def plot(self, audio_file, x_axis, y_axis):
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
        plt.title("Normalized audio waveform for " + audio_file)
        plt.tight_layout()

        #save plot as png under plots_audiowaves/<fold>/foldX_filename.png    
        p = Path(audio_file)
        fold = p.parent.name
        plots_dir = Path("plots_audiowaves") / fold
        plots_dir.mkdir(parents=True, exist_ok=True)
        outfile = plots_dir / f"{fold}_{p.stem}.png"
        plt.savefig(outfile, dpi=150, bbox_inches="tight")

        if print_output:
            print(f"Audio waveform plot saved to: {outfile}")

    ######## PLOTTING NORMALIZED SPECTROGRAM ###########
    def plot_splectrogram(self, frequency, time, spectrogram, audio_file_path):
        plt.figure(figsize=(5, 4), dpi=150)
        plt.pcolormesh(time, frequency, np.log(spectrogram))
        plt.ylabel('Mel Frequency')
        plt.xlabel('Time')
        plt.title('Mel Spectrogram')
        plt.colorbar(format='%+2.0f dB')
        plt.tight_layout()

        #save plot as png under plots_spectrogram/<fold>/foldX_filename.png    
        p = Path(audio_file_path)
        fold = p.parent.name
        plots_dir = Path("plots_spectrogram") / fold
        plots_dir.mkdir(parents=True, exist_ok=True)
        outfile = plots_dir / f"{fold}_{p.stem}.png"
        plt.savefig(outfile, dpi=150, bbox_inches="tight")

        if print_output:
            print(f"Audio spectrogram plot saved to: {outfile}") 

    ####### GETTING SPECTROGRAM FOR AUDIO FILE ###########
    def audio_mel_spectrogram_png(self, audio_file_path, class_id ):

        spec_image = spectrogram_process.__getitem__(audio_file_path, class_id)

        sig, sr = AudioUtilHandler.open(audio_file_path)

        if print_output:
            print (f"audio_mel_spectrogram_png: Spectrogram Type: {type(spec_image)}")
            print (f"audio_mel_spectrogram_png: Spectrogram{spec_image}")

        
        num_samples = sig.shape[1] # number of samples
        time = num_samples / sr  #the x-axis as duration in seconds

        #use second channel if present, otherwise fall back to the first to support mono audio
        channel_idx = 1 if sig.shape[0] > 1 else 0
        frequency, time , spectrogram = signal.spectrogram(sig.numpy()[channel_idx], sr)

        self.plot_splectrogram(frequency, time, spectrogram, audio_file_path)
            
        return frequency, time, spectrogram

    ######## GETTING AUDIOWAVES FROM AUDIOFILE (raw) ###########
    def audio_waveform_png_raw(self, audio_file):
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
            print (f"\naudio_waveform_png_raw:  Sample Rate: {sr}")
            print (f"naudio_waveform_png_raw: stereo audio stero or mono: {channel_type} channels")
            print (f"naudio_waveform_png_raw: audio duration in seconds: {audio_duration} seconds")
            print (f"audio_waveform_png_raw: Number of samples: {num_samples}")
            print (f"audio_waveform_png_raw: Duration in seconds: {audio_duration}")

        #y axis as amplitude
        audiowave = sig.numpy()[0]

        self.plot(audio_file, audio_duration, audiowave)

        return num_samples, audiowave




##### USE CASE EXAMPLE ######

audio_visualizer = visualize()
#dogbark
#audio_visualizer.audio_waveform_png_raw("dataset/fold1/197073-3-3-0.wav")

#siren 
#audio_visualizer.audio_waveform_png_raw("dataset/fold8/133473-8-0-3.wav")

#audio_visualizer.audio_waveform_png_raw("dataset/fold1/180937-7-2-0.wav")
#audio_visualizer.audio_mel_spectrogram_png("dataset/fold1/180937-7-2-0.wav", 7)
