import math, random, torch, torchaudio

from torchaudio import transforms

class AudioUtil:
    """ Load an audio file. Return the signal as a tensor and the sample rate. """
    @staticmethod
    def open(audio_file):
        sig, sr = torchaudio.load(audio_file)
        return sig, sr

    @staticmethod
    def convert_to_new_channel(audio, new_channel):
        """ Convert the sound files from mono to stereo, by duplicating the first channel to the second channel. """
        sig, sr = audio

        # If the same, then don't convert
        if sig.shape[0] == new_channel:
            return audio

        # Convert from stereo to mono by selecing ONLY the first channel
        if new_channel == 1:
            resig = sig[:1, :]

        # Convert from mono to stereo by duplicating the first channel
        else:
            resig = torch.cat([sig,sig])

        return resig, sr


    @staticmethod
    def resample(audio, new_sample_rate):
        """ Resamples the sound file to a new sample rate.
        Since it applies to a single channel, we resample one channel at a time. """
        sig, sr = audio

        # No point in resampling if its the same
        if sr == new_sample_rate:
            return audio

        num_channels = sig.shape[0]

        # First, resample the first channel
        resig = torchaudio.transforms.Resample(sr, new_sample_rate)(sig[:1,:])

        # If more channels, resample the second channel, then merge both of them
        if num_channels > 1:
            retwo = torchaudio.transforms.Resample(sr, new_sample_rate)(sig[:1, :])
            resig = torch.cat([resig, retwo])

        return resig, new_sample_rate

    @staticmethod
    def padding_truncate(audio, max_ms):
        """ Pad (or truncate) the signal to a fixed length 'max_ms' in milliseconds. """
        sig, sr = audio
        num_rows, sig_len = sig.shape
        max_len = sr//1000 * max_ms

        # If signal length is bigger than max length, then truncate it to a given length
        if sig_len > max_len:
            sig = sig[:, :max_len]

        # Length of padding to add at the beginning and the end of the signal
        elif sig_len < max_len:
            pad_begin_length = random.randint(0, max_len - sig_len)
            pad_end_length = max_len - sig_len - pad_begin_length

            pad_begin = torch.zeros((num_rows, pad_begin_length))
            pad_end = torch.zeros((num_rows, pad_end_length))

            sig = torch.cat([pad_begin, sig, pad_end], dim=1)

        return sig, sr

    @staticmethod
    def time_shift(audio, shift_limit):
        """ Shifts the signal to the left / right by a random amount. """
        sig, sr = audio
        _, sig_len = sig.shape
        shift_amount = int(random.random() * shift_limit * sig_len)
        return sig.roll(shift_amount), sr


    @staticmethod
    def spectrogram(audio, n_mels=64, n_fft=1024, hop_len=None):
        """ Return spectrogram of audio """
        sig, sr = audio
        top_db = 80

        # spec will have shape [channel, n_mels, time], where channel is mono, stereo etc
        spec = transforms.MelSpectrogram(sr, n_fft=n_fft, hop_length=hop_len, n_mels=n_mels)(sig)

        # Convert to decibels
        spec = transforms.AmplitudeToDB(top_db=top_db)(spec)
        return spec

    @staticmethod
    def spectro_augmentation(spectrogram, max_mask_pct=0.1, n_freq_masks=1, n_time_masks=1):
        """
        Augment the Spectrogram by masking out some sections of it in both the frequency dimension (ie. horizontal bars) and the time dimension (vertical bars) to prevent         overfitting and to help the model generalise better. The masked sections are replaced with the mean value.
        """
        _, n_mels, n_steps = spectrogram.shape
        mask_value = spectrogram.mean()
        augment_spec = spectrogram

        # Horizontal bars of the spectrogram
        freq_mask_param = max_mask_pct * n_mels
        for _ in range(n_freq_masks):
            augment_spec = transforms.FrequencyMasking(freq_mask_param)(augment_spec, mask_value)

        # Vertical bars of the spectrogram
        time_mask_param = max_mask_pct * n_steps
        for _ in range(n_time_masks):
            augment_spec = transforms.TimeMasking(time_mask_param)(augment_spec, mask_value)

        return augment_spec