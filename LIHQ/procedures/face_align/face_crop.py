from pathlib import Path
from io import BytesIO
from matplotlib import pyplot as plt
from PIL import Image
import torchvision
from IPython import display
import numpy as np
from LIHQ.procedures.face_align.shape_predictor import align_face
from LIHQ.procedures.face_align.drive import open_url
import dlib
from mpl_toolkits.axes_grid1 import ImageGrid
from LIHQ.procedures.face_align.bicubic import BicubicDownSample

######################################
# Stolen (and modified) from PULSE face depixelization
# https://github.com/adamian98/pulse
# Who stole it from someone else

def crop_face(filename, outfile):
    #downloading model weights
    f=open_url("https://drive.google.com/uc?id=1huhv8PYpNNKbGCLOaYUjOgR1pY5pmbJx", cache_dir="cache", return_path=True)
    predictor = dlib.shape_predictor(f)

    toPIL = torchvision.transforms.ToPILImage()
    toTensor = torchvision.transforms.ToTensor()

    images = []
    faces = align_face(filename,predictor)
    face = faces[0]
    face = toPIL(toTensor(face).unsqueeze(0).cuda().cpu().detach().clamp(0,1)[0])
    images.append(face)
    face.save(outfile)

    if(len(images)==0): raise Exception("No faces found. Try again with a different image.")