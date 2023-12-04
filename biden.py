
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
text = "Hello, I'm Joe Biden, and I support this message. In these challenging times, unity is our strength.  Together, we can tackle the pressing issues facing our nation – from health to economy. As your leader, I'm committed to building a more just and prosperous future. Thank you, and may God bless the United States of America."
voice = 'biden'
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


from LIHQ.procedures.face_align.face_crop import crop_face

crop_face(filename = 'LIHQ/input/face/biden.jpg',
          outfile = 'LIHQ/input/face/bidenCrop.jpg')

from LIHQ.runLIHQ import run

run(face='LIHQ/input/face/bidenCrop.jpg', frame_int=2)


#Important! If you want to paste the speaker in the same location as the original image, look at printouts of image cropping cell above

speaker_vid = 'LIHQ/output/finalVidsOut/Folder1.mp4' # Your speaker video created by LIHQ
background = 'LIHQ/input/face/biden.jpg' # Can be image or video (Final video output will be smallest fps between background vid and speaker vid.)
bg_resize = False # Set to False or your desired background size in format: (width, height)
spkr_resize = (221, 221) # Set to False or your desired speaker size in format: (width, height)
offset = [47, 757] # In the form [y, x]. Offset of speaker image onto background, from top left
rotation = 9.37 #Counterclockwise, in degrees
output_path = "biden_static.mp4" # EVERYTHING IN POSTPROCESSING FOLDER GETS DELETED AT THE START OF A NEW RUN!
                                    # So save outside of postprocessing folder.


# Just running a preview of first frame of final video. Adjust parameters above as needed and rerun
%cd '/content/LIHQ'

from IPython.display import Image
from pathlib import Path

from procedures.matting_scripts import matte_preview
from procedures.matting_scripts import matte_combine

#Delete Video Matting files from previous run (if any)
for path in Path("./output/postprocessing/").glob("**/*"): #Check path with cwd
    if path.is_file():
        path.unlink()

#Generating image data
matte_preview(speaker_vid, background, bg_resize, spkr_resize, offset)

#Creating mask with MODNet
%cd '/content/LIHQ/MODNet'
!python -m demo.image_matting.colab.inference \
        --input-path /content/LIHQ/output/postprocessing/input \
        --output-path /content/LIHQ/output/postprocessing/masks \
        --ckpt-path ./pretrained/modnet_photographic_portrait_matting.ckpt
%cd '..'

#Combining speaker, mask, background
matte_combine(offset, rotation)

Image('./output/postprocessing/output/preview.png')
