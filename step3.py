# Adding half a second of silence before audio, because FOMM works better when first frame is a closed mouth.
# Also helps with abrupt start/ stop if you want to add silence after as well.
from pydub import AudioSegment

silence_duration = 500  # Set duration in milliseconds. 1000 is 1 second
save_path = 'LIHQ/input/audio/Folder1/1.wav'

silent_segment = AudioSegment.silent(duration=silence_duration)  
silent_segment.export(save_path, format="wav")
