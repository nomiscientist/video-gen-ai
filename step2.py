import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F

#import IPython

from tortoise.api import TextToSpeech
from tortoise.utils.audio import load_audio, load_voice, load_voices

from LIHQ.procedures.tortoise_scripts import tortoise_run, tortoise_combo_run

tts = TextToSpeech()


# Generating a random voice to speak the sentence below. When you generate one you like, save it using the cell below

text = "Look how easy it is to create an artificial speaker."
voice = 'freeman'
preset = "high_quality" #Options: {"ultra_fast", "fast", "standard", "high_quality"}.

gen = tortoise_run(tts, text, voice, preset)

torchaudio.save('LIHQ/input/audio/Folder1/2.wav', gen.squeeze(0).cpu(), 24000)

# Adding half a second of silence before audio, because FOMM works better when first frame is a closed mouth.
# Also helps with abrupt start/ stop if you want to add silence after as well.

from pydub import AudioSegment

silence_duration = 500  # Set duration in milliseconds. 1000 is 1 second
save_path = 'LIHQ/input/audio/Folder1/1.wav'

silent_segment = AudioSegment.silent(duration=silence_duration)  
silent_segment.export(save_path, format="wav")

from LIHQ.runLIHQ import run

run(face='/workspace/video-gen-ai/LIHQ/input/face/examples/morgan.png')