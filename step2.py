
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
voice = 'random'
preset = "high_quality" #Options: {"ultra_fast", "fast", "standard", "high_quality"}.

gen = tortoise_run(tts, text, voice, preset)

torchaudio.save('LIHQ/input/audio/Folder1/2.wav', gen.squeeze(0).cpu(), 24000)
