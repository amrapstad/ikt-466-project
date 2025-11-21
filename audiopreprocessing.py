from torch.utils.data import DataLoader, Dataset, random_split
from audioutils import AudioUtil
import torchaudio

class SoundDS(Dataset):
    """ Sound Dataset. """
    def __init__(self):
        self.duration = 4000        # Duration of the .wav files in milliseconds
        self.sr = 44100             # Sample rate: 44100 Hz
        self.channel = 2            # Amount of channels (two is stereo)
        self.shift_pct = 0.4        # Shifts

    def __len__(self):
        """ Return the length of the dataset. """
        return len(self.df)

    def __getitem__(self, filepath, class_id):
        """ Get the i'th item in dataset. """
        # Absolute file path of the audio file - concatenate with the audio directory of the relative path
        audio_file = filepath




        audio = AudioUtil.open(audio_file)



        reaudio = AudioUtil.resample(audio, self.sr)
        rechannel = AudioUtil.convert_to_new_channel(reaudio, self.channel)

        duration_audio = AudioUtil.padding_truncate(rechannel, self.duration)
        shift_audio = AudioUtil.time_shift(duration_audio, self.shift_pct)
        spectrogram = AudioUtil.spectrogram(shift_audio, n_mels=64, n_fft=1024, hop_len=None)
        augmented_spectrogram = AudioUtil.spectro_augmentation(spectrogram, max_mask_pct=0.1, n_freq_masks=2, n_time_masks=2)

        return augmented_spectrogram, class_id