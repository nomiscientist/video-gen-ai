import os
import subprocess
import sys

def wav2lip_run(adir):
  print("path:" + os.getcwd())
  vid_path = f'{os.getcwd()}/LIHQ/output/FOMM/Round1/{adir}.mp4'
  aud_path = f'{os.getcwd()}/LIHQ/input/audio/{adir}/{adir}.wav'
  out_path = f'{os.getcwd()}/LIHQ/output/wav2Lip/{adir}.mp4'
  os.chdir('LIHQ/Wav2Lip')
  command = f'python /workspace/video-gen-ai/LIHQ/Wav2Lip/inference.py --checkpoint_path /workspace/video-gen-ai/LIHQ/Wav2Lip/checkpoints/wav2lip.pth --face {vid_path} --audio {aud_path} --outfile {out_path}  --pads 0 20 0 0'
  try:
    subprocess.call(command, shell=True)
  except subprocess.CalledProcessError:
    print('!!!!!!! Error with Wav2Lip Paths !!!!!!')
    sys.exit()
  os.chdir('..')
