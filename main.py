from audioutils import AudioUtil
import ffmpeg

AudioUtilHandler = AudioUtil()

sig, sr = AudioUtilHandler.open("dataset/fold8/133473-8-0-3.wav")
