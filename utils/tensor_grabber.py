# This code preprocesses the visual components of Ego4D-AVD into trackwise pytorch tensors for TalkNet+DeiT.
# The code is designed to be run in parallel across multiple HPC sessions simultaneously for efficiency.


# Run the following commands across multiple HPC sessions simultaneously:
# train/val: 
#   python tensor_grabber.py --annotPath {path to ego4d/csv & bbox} --split {train/val} --dataPath {path to video_imgs} --savePath {path to save tensors}
# testing ASD on Ego4D reconfigured validation fold:
#   python tensor_grabber.py --annotPath {path to infer/csv & bbox} --split val --dataPath {path to video_imgs} --savePath {path to save tensors}
# Run across single session to fill any missing tracks:
# train/val:
#   python tensor_grabber.py --annotPath {path to ego4d/csv & bbox} --split {train/val} --dataPath {path to video_imgs} --savePath {path to save tensors} --fillPass
# testing ASD on Ego4D reconfigured validation fold:
#   python tensor_grabber.py --annotPath {path to infer/csv & bbox} --split val --dataPath {path to video_imgs} --savePath {path to save tensors} --fillPass

from PIL import Image
import torch

from transformers import ViTFeatureExtractor

import json
import os 

import scipy.signal as signal

import cv2

from scipy.interpolate import interp1d
import numpy as np
import time

from tqdm import tqdm

import argparse 

def check(track):
    inter_track = []
    framenum = []
    bboxes = []
    for frame in track:
        #print(frame)
        x = frame['x']
        y = frame['y']
        w = frame['width']
        h = frame['height']
        if (w <= 0 or h <= 0 or 
            frame['frame']==0):
            continue
        framenum.append(frame['frame'])
        x = max(x, 0)
        y = max(y, 0)
        bbox = [x, y, x + w, y + h]
        bboxes.append(bbox)
    
    if len(framenum) == 0:
        return inter_track

    framenum = np.array(framenum)
    bboxes = np.array(bboxes)

    gt_frames = framenum[-1] - framenum[0] + 1

    frame_i = np.arange(framenum[0], framenum[-1]+1)

    if gt_frames > framenum.shape[0]:
        bboxes_i = []
        for ij in range(0,4):
            interpfn  = interp1d(framenum, bboxes[:,ij])
            bboxes_i.append(interpfn(frame_i))
        bboxes_i  = np.stack(bboxes_i, axis=1)
    else:
        frame_i = framenum
        bboxes_i = bboxes

    #assemble new tracklet
    template = track[0]
    for i, (frame, bbox) in enumerate(zip(frame_i, bboxes_i)):
        record = template.copy()
        record['frame'] = frame
        record['x'] = bbox[0]
        record['y'] = bbox[1]
        record['width'] = bbox[2] - bbox[0]
        record['height'] = bbox[3] - bbox[1]
        inter_track.append(record)
    return inter_track

class SaveAsTensor():
    def __init__(self, dataPath, annotPath, split, savePath, fill_pass = False):
        self.dataPath = dataPath
        self.miniBatch = []      
        if args.forInfer == True:
            self.savePath = os.path.join(savePath, 'forInfer', split)
        else:
            self.savePath = os.path.join(savePath, split)
        
        self.feature_extractor = ViTFeatureExtractor.from_pretrained('facebook/deit-tiny-patch16-224')
        self.track_direc = os.path.join(annotPath, 'bbox')
        self.fill_pass = fill_pass
        # make save directory
        os.makedirs(self.savePath, exist_ok=True)
        mixLst = open(f'{annotPath}/csv/active_speaker_{split}.csv').read().splitlines()
        

        sortedMixLst = sorted(mixLst, key=lambda data: (int(data.split('\t')[1]), int(data.split('\t')[-1])), reverse=True)         
        start = 0      
        batchSize = 1
        while True:
          length = int(sortedMixLst[start].split('\t')[1])
          end = min(len(sortedMixLst), start + max(int(batchSize / length), 1))
          self.miniBatch.append(sortedMixLst[start:end])
          if end == len(sortedMixLst):
              break
          start = end
          
    def grab_tensors(self):
        batchList = self.miniBatch
        for i, line in enumerate(tqdm(batchList)):
            data = line[0].split('\t')
            print('current trackid:' , data[0])
            if self.fill_pass: # if fill pass then check for track folders without all [faces.pt, images.pt, p_ss.pt] 
                self.fill_missing(data)
            else:
                self.fill(data) # else fill current track folder unless flag directory is present in folder

    def fill(self, data):
        direc = os.path.join(self.savePath, data[0])
        os.makedirs(f'{direc}', exist_ok=True)
        # check if flag in listdir(direc)
        if "flag" not in os.listdir(direc):
            # create a flag file and save it as direc/flag
            os.makedirs(f'{direc}/flag', exist_ok=True)
            print('processing track: ', data[0])
            if args.forInfer == True:
                trackid = data[0]
                with open(f'{self.track_direc}/{trackid}.json', 'r') as f:
                    bbox = json.load(f)
                frames = bbox
                self.process_data(frames, trackid)   
            else: 
                trackid = data[0]
                with open(f'{self.track_direc}/{trackid}.json', 'r') as f:
                        bbox = json.load(f)
                        bbox = { b["frame"]:b for b in bbox }
                track = [bbox[i] for i in range(int(data[-1]), int(data[-1])+int(data[1])) if i in bbox]
                frames = check(track)
                self.process_data(frames, trackid)
            # remove flag folder
        else:
            print('track already processed: ', data[0], 'therefore skipping...')

    def fill_missing(self, data):
        direc = os.path.join(self.savePath, data[0])
        if not all(file in os.listdir(direc) for file in ["faces.pt", "images.pt", "p_ss.pt"]):
            print('filling missing files in: ', data[0])
            if args.forInfer == True:
                trackid = data[0]
                with open(f'{self.track_direc}/{trackid}.json', 'r') as f:
                    bbox = json.load(f)
                frames = bbox
                self.process_data(frames, trackid)
            else:
                trackid = data[0]
                with open(f'{self.track_direc}/{trackid}.json', 'r') as f:
                        bbox = json.load(f)
                        bbox = { b["frame"]:b for b in bbox }
                track = [bbox[i] for i in range(int(data[-1]), int(data[-1])+int(data[1])) if i in bbox]
                frames = check(track)
                self.process_data(frames, trackid)
        else:
            print('track already processed: ', data[0], 'therefore skipping...')

    def process_data(self, frames, trackid): ############# orig
        t0 = time.time()
        videoName = trackid[:36]
        imgFolderPath = os.path.join(self.dataPath, videoName)

        H = 112
        p_ss = []
        dets = {'x':[], 'y':[], 's':[]}
        for frame in frames:
            frameid = frame['frame']
            if args.forInfer == True:
                x1 = frame['x1']
                y1 = frame['y1']
                x2 = frame['x2']
                y2 = frame['y2']
            else:
                x1 = frame['x']
                y1 = frame['y']
                x2 = frame['x'] + frame['width']
                y2 = frame['y'] + frame['height']
            p_ss.append([x1, y1, x2, y2]) # should add smoothing here when not using ground truth localisation
            dets['s'].append(max((y2-y1), (x2-x1))/2)
            dets['y'].append((y2+y1)/2) # crop center x
            dets['x'].append((x2+x1)/2) # crop center y
        kernel_size = min((len(dets['s'])-len(dets['s'])%2+1), 13)
        dets['s'] = signal.medfilt(dets['s'], kernel_size=kernel_size)  # Smooth detections
        dets['x'] = np.array(dets['x'])
        dets['x'][1:] = dets['x'][:-1]*0.8 + dets['x'][1:]*0.2
        dets['y'] = np.array(dets['y'])
        dets['y'][1:] = dets['y'][:-1]*0.8 + dets['y'][1:]*0.2

        faces = []
        images = []
        for i, frame in enumerate(frames):
            frameid = frame['frame']
            if frameid == 0:
                frameid = 1
            img = cv2.imread(f'{imgFolderPath}/img_{int(frameid):05d}.jpg')
            images.append(cv2.resize(img, (224,224)))
            cs  = 0.4
            bs  = dets['s'][i]
            bsi = int(bs * (1 + 2 * cs))  # Pad videos by this amount
            img = np.pad(img, ((bsi,bsi), (bsi,bsi), (0, 0)), 'constant', constant_values=(110, 110))
            my  = dets['y'][i] + bsi  # BBox center Y
            mx  = dets['x'][i] + bsi  # BBox center X
            face = img[int(my-bs):int(my+bs*(1+2*cs)),int(mx-bs*(1+cs)):int(mx+bs*(1+cs))]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            face = cv2.resize(face, (2*H,2*H))
            face = face[int(112-(112/2)):int(112+(112/2)), int(112-(112/2)):int(112+(112/2))]
            faces.append(face)
        
        faces = np.array(faces)
        faces = torch.from_numpy(faces)
        images = np.array(images).astype(np.uint8)
        images = self.feature_extractor(images, return_tensors="pt")['pixel_values']
        p_ss = np.array(p_ss)
        p_ss = torch.from_numpy(p_ss)

        # assert first dimension of faces, images and p_ss are equal
        assert faces.shape[0] == images.shape[0] == p_ss.shape[0], f'first dimension of faces, images and p_ss are not equal: {faces.shape[0]}, {images.shape[0]}, {p_ss.shape[0]} for trackid: {trackid}'

        direc = os.path.join(self.savePath, trackid)
        torch.save(faces.to(torch.float16), f'{direc}/faces.pt')
        torch.save(images.to(torch.float16), f'{direc}/images.pt')
        torch.save(p_ss.to(torch.float16), f'{direc}/p_ss.pt')
        print(f'time taken to save track as tensors: {time.time()-t0}')
        print(f'saved {trackid} successfully!') 




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotPath', type = str, 
        default = "/users/acp21jrc/audio-visual/active-speaker-detection/active_speaker/TalkNet_ASD/Ego4d_TalkNet_ASD/data/ego4d",
        help = "parent directory to where csv i.e active_speaker_{split}.csv and streams for .jsons are stored")
    parser.add_argument('--dataPath', type = str, 
        default = "/mnt/parscratch/users/acp21jrc/ego4d_data/v2/data/video_imgs",
        help = "path to where the video images are stored")
    parser.add_argument('--savePath', type = str,
        default = '/mnt/parscratch/users/acp21jrc/ego4d_data/v2/data/tensors/',
        help = "path to where the tensors will be saved")
    parser.add_argument('--split', type = str, default = 'train')
    parser.add_argument('--fillPass', action='store_true', help='Fill embeddings missed during multiprocessing')
    parser.add_argument('--forInfer', action='store_true', help='Inference or train/val?')
    args = parser.parse_args()

    tensorgrabber = SaveAsTensor(args.dataPath, args.annotPath, args.split, args.savePath, args.fillPass)
    tensorgrabber.grab_tensors()
