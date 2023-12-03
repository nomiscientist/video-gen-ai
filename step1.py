import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F

#import IPython

from tortoise.api import TextToSpeech
from tortoise.utils.audio import load_audio, load_voice, load_voices

from LIHQ.procedures.tortoise_scripts import tortoise_run, tortoise_combo_run

tts = TextToSpeech()
