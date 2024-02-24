import os, torch, numpy, cv2, random, python_speech_features, json
from scipy.io import wavfile
from scipy.interpolate import interp1d
import numpy as np
from scipy import signal

import torchvision.transforms as transforms

import time

def normalize(samples, desired_rms = 0.1, eps = 1e-4):
    samples = samples / 32768
    rms = np.maximum(eps, np.sqrt(np.mean(samples**2)))
    samples = samples * (desired_rms / rms)
    return samples


def generate_audio_set(dataPath, batchList):
    audioSet = {}
    for line in batchList:
        data = line.split('\t')
        videoName = data[0][:36]
        dataName = data[0]
        fps = float(data[2])
        start = int(data[-1])/fps
        end = (int(data[-1]) + int(data[1]))/fps
        sr, audio = wavfile.read(os.path.join(dataPath, videoName + '.wav'))
        audio = audio[int(start*sr): int(end*sr)]
        if len(audio) == 0:
            audio = np.zeros((int(end*sr)-int(start*sr)))
        audioSet[dataName] = normalize(audio)
    return audioSet


def overlap(dataName, audio, audioSet):   
    noiseName =  random.sample(set(list(audioSet.keys())) - {dataName}, 1)[0]
    noiseAudio = audioSet[noiseName]    
    snr = [random.uniform(-5, 5)]
    if len(noiseAudio) < len(audio):
        shortage = len(audio) - len(noiseAudio)
        noiseAudio = numpy.pad(noiseAudio, (0, shortage), 'wrap')
    else:
        noiseAudio = noiseAudio[:len(audio)]
    noiseDB = 10 * numpy.log10(numpy.mean(abs(noiseAudio ** 2)) + 1e-4)
    cleanDB = 10 * numpy.log10(numpy.mean(abs(audio ** 2)) + 1e-4)
    noiseAudio = numpy.sqrt(10 ** ((cleanDB - noiseDB - snr) / 10)) * noiseAudio
    audio = audio + noiseAudio    
    return audio.astype(numpy.int16)


def load_audio(data, dataPath, numFrames, audioAug, audioSet = None):
    dataName = data[0]
    fps = float(data[2])    
    audio = audioSet[dataName]
    if audioAug == True:
        augType = random.randint(0,1)
        if augType == 1:
            audio = overlap(dataName, audio, audioSet)
        else:
            audio = audio
    # fps is not always 25, in order to align the visual, we modify the window and step in MFCC extraction process based on fps
    audio = python_speech_features.mfcc(audio, 16000, numcep = 13, winlen = 0.025 * 25 / fps, winstep = 0.010 * 25 / fps)
    maxAudio = int(numFrames * 4)
    if audio.shape[0] < maxAudio:
        shortage    = maxAudio - audio.shape[0]
        audio     = numpy.pad(audio, ((0, shortage), (0,0)), 'wrap')
    audio = audio[:int(numFrames * 4),:]  
    return audio

def load_visual(data, dataPath, numFrames, visualAug):
    trackid = data[0]
    faceFolderPath = os.path.join(dataPath, trackid)
    tensor = torch.load(f'{faceFolderPath}/faces.pt').to(torch.float32)
    H = 112
    if visualAug == True:
        # Define some image transformations
        flip = transforms.RandomHorizontalFlip(p=1.0) # flip with probability 1.0
        crop = transforms.RandomResizedCrop(size=H, scale=(0.7, 1.0)) # crop and resize with random scale
        rotate = transforms.RandomRotation(degrees=(-15, 15)) # rotate with random angle
        augType = random.choice(['orig', 'flip', 'crop', 'rotate']) 
    else:
        augType = 'orig'

    if augType == 'orig':
        pass # do nothing
    elif augType == 'flip':
        tensor = flip(tensor)
    elif augType == 'crop':
        tensor = crop(tensor)
    elif augType == 'rotate':
        tensor = rotate(tensor)
    return tensor.numpy()

def load_context(data, dataPath, numFrames, visualAug):
    trackid = data[0]
    imageFolderPath = os.path.join(dataPath, trackid)
    tensor = torch.load(f'{imageFolderPath}/images.pt').to(torch.float32)
    return tensor.numpy()

def load_bboxes(data, dataPath):
    trackid = data[0]
    bboxesFolderPath = os.path.join(dataPath, trackid)
    bboxes = torch.load(f'{bboxesFolderPath}/p_ss.pt').to(torch.float32)
    return bboxes.numpy()

 


# def load_visual(data, dataPath, numFrames, visualAug): 
#     trackid = data[0]
#     videoName = data[0][:36]
#     faceFolderPath = os.path.join(dataPath, videoName)
#     with open(f'data/ego4d/bbox/{trackid}.json', 'r') as f:
#         bbox = json.load(f)
#         bbox = { b["frame"]:b for b in bbox }
#     track = [bbox[i] for i in range(int(data[-1]), int(data[-1])+int(data[1])) if i in bbox]
#     frames = check(track)
    
#     faces = []
#     H = 112
#     if visualAug == True:
#         new = int(H*random.uniform(0.7, 1))
#         x, y = numpy.random.randint(0, H - new), numpy.random.randint(0, H - new)
#         M = cv2.getRotationMatrix2D((H/2,H/2), random.uniform(-15, 15), 1)
#         augType = random.choice(['orig', 'flip', 'crop', 'rotate']) 
#     else:
#         augType = 'orig'
    
#     dets = {'x':[], 'y':[], 's':[]}
#     for frame in frames:
#         frameid = frame['frame']
#         x1 = frame['x']
#         y1 = frame['y']
#         x2 = frame['x'] + frame['width']
#         y2 = frame['y'] + frame['height']
#         dets['s'].append(max((y2-y1), (x2-x1))/2) 
#         dets['y'].append((y2+y1)/2) # crop center x 
#         dets['x'].append((x2+x1)/2) # crop center y
#     kernel_size = min((len(dets['s'])-len(dets['s'])%2+1), 13)
#     dets['s'] = signal.medfilt(dets['s'], kernel_size=kernel_size)  # Smooth detections 
#     dets['x'] = np.array(dets['x'])
#     dets['x'][1:] = dets['x'][:-1]*0.8 + dets['x'][1:]*0.2
#     dets['y'] = np.array(dets['y'])
#     dets['y'][1:] = dets['y'][:-1]*0.8 + dets['y'][1:]*0.2

#     for i, frame in enumerate(frames): 
#         frameid = frame['frame'] = frame['frame']
#         img = cv2.imread(f'{faceFolderPath}/img_{int(frameid):05d}.jpg')
#         cs  = 0.4
#         bs  = dets['s'][i]
#         bsi = int(bs * (1 + 2 * cs))  # Pad videos by this amount 
#         img = numpy.pad(img, ((bsi,bsi), (bsi,bsi), (0, 0)), 'constant', constant_values=(110, 110))
#         my  = dets['y'][i] + bsi  # BBox center Y
#         mx  = dets['x'][i] + bsi  # BBox center X
#         face = img[int(my-bs):int(my+bs*(1+2*cs)),int(mx-bs*(1+cs)):int(mx+bs*(1+cs))]
#         face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
#         face = cv2.resize(face, (2*H,2*H))
#         face = face[int(112-(112/2)):int(112+(112/2)), int(112-(112/2)):int(112+(112/2))]
#         if augType == 'orig':
#             faces.append(face)
#         elif augType == 'flip':
#             faces.append(cv2.flip(face, 1))
#         elif augType == 'crop':
#             faces.append(cv2.resize(face[y:y+new, x:x+new] , (H,H))) 
#         elif augType == 'rotate':
#             faces.append(cv2.warpAffine(face, M, (H,H)))
#     faces = numpy.array(faces[:numFrames])
#     return faces


def check(track):
    inter_track = []
    framenum = []
    bboxes = []
    for frame in track:
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


def load_label(data, numFrames):
    res = []
    labels = data[3].replace('[', '').replace(']', '')
    labels = labels.split(',')
    for label in labels:
        res.append(int(label))
    res = numpy.array(res[:numFrames])
    return res


class train_loader(object):
    def __init__(self, annotPath, audioPath, dataPath, batchSize, **kwargs):
        self.audioPath  = audioPath
        self.dataPath = os.path.join(dataPath, 'train')
        trialFileName = os.path.join(annotPath, 'csv', 'active_speaker_train.csv')
        self.miniBatch = []      
        mixLst = open(trialFileName).read().splitlines()
        sortedMixLst = sorted(mixLst, key=lambda data: (int(data.split('\t')[1]), int(data.split('\t')[-1])), reverse=True)         
        start = 0        
        while True:
          length = int(sortedMixLst[start].split('\t')[1])
          end = min(len(sortedMixLst), start + max(int(batchSize / length), 1))
          self.miniBatch.append(sortedMixLst[start:end])
          if end == len(sortedMixLst):
              break
          start = end

    def __getitem__(self, index):
        batchList    = self.miniBatch[index]
        numFrames   = int(batchList[-1].split('\t')[1])
        audioFeatures, visualFeatures, contextFeatures, bboxes, labels = [], [], [], [], []
        audioSet = generate_audio_set(self.audioPath, batchList) # load the audios in this batch to do augmentation
        for line in batchList:
            data = line.split('\t')       
            audioFeatures.append(load_audio(data, self.audioPath, numFrames, audioAug = True, audioSet = audioSet))  
            visualFeatures.append(load_visual(data, self.dataPath,numFrames, visualAug = True))
            #print('type: ' ,type(load_context(data, self.dataPath, numFrames, visualAug = True)))
            #print('shape: ', load_context(data, self.dataPath, numFrames, visualAug = True).shape)
            contextFeatures.append(load_context(data, self.dataPath, numFrames, visualAug = True))
            bboxes.append(load_bboxes(data, self.dataPath))
            labels.append(load_label(data, numFrames))
        minFrames = min([t.shape[0] for t in visualFeatures])
        audioFeatures = [a[:4*minFrames, :] for a in audioFeatures]
        visualFeatures = [v[:minFrames] for v in visualFeatures]
        contextFeatures = [c[:minFrames] for c in contextFeatures]
        bboxes = [b[:minFrames] for b in bboxes]
        labels = [l[:minFrames] for l in labels]
        audio = torch.FloatTensor(numpy.array(audioFeatures))
        faces = torch.FloatTensor(numpy.array(visualFeatures))
        contextFeatures = torch.FloatTensor(numpy.array(contextFeatures))
        bboxes = torch.FloatTensor(numpy.array(bboxes))
        labels = torch.LongTensor(numpy.array(labels))
        return audio, faces, contextFeatures, bboxes, labels
        
    def __len__(self):
        return len(self.miniBatch)


class val_loader(object):
    def __init__(self, annotPath, audioPath, dataPath, **kwargs):
        self.audioPath  = audioPath
        self.dataPath = os.path.join(dataPath, 'val')
        trialFileName = os.path.join(annotPath, 'csv', 'active_speaker_val.csv')
        self.miniBatch = open(trialFileName).read().splitlines()

    def __getitem__(self, index):
        line       = [self.miniBatch[index]]
        numFrames  = int(line[0].split('\t')[1])
        audioSet   = generate_audio_set(self.audioPath, line)        
        data = line[0].split('\t')
        audioFeatures = [load_audio(data, self.audioPath, numFrames, audioAug = False, audioSet = audioSet)]
        visualFeatures = [load_visual(data, self.dataPath, numFrames, visualAug = False)]
        contextFeatures = [load_context(data, self.dataPath, numFrames, visualAug = False)]
        bboxes = [load_bboxes(data, self.dataPath)]
        labels = [load_label(data, numFrames)]         
        minFrames = min([t.shape[0] for t in visualFeatures])
        audioFeatures = [a[:4*minFrames, :] for a in audioFeatures]
        visualFeatures = [v[:minFrames] for v in visualFeatures]
        contextFeatures = [c[:minFrames] for c in contextFeatures]
        bboxes = [b[:minFrames] for b in bboxes]
        labels = [l[:minFrames] for l in labels]
        audio = torch.FloatTensor(numpy.array(audioFeatures))
        faces = torch.FloatTensor(numpy.array(visualFeatures))
        contextFeatures = torch.FloatTensor(numpy.array(contextFeatures))
        bboxes = torch.FloatTensor(numpy.array(bboxes))
        labels = torch.LongTensor(numpy.array(labels))
        return audio, faces, contextFeatures, bboxes, labels
    
    def __len__(self):
        return len(self.miniBatch)
    
# class test_loader(object): ############ original (no bbox or context images)
#     def __init__(self, trialFileName, audioPath, visualPath, **kwargs):
#         self.audioPath  = audioPath
#         self.visualPath = visualPath
#         self.miniBatch = open(trialFileName).read().splitlines()

#     def __getitem__(self, index):
#         line       = [self.miniBatch[index]]
#         numFrames  = int(line[0].split('\t')[1])
#         audioSet   = generate_audio_set(self.audioPath, line)        
#         data = line[0].split('\t')
#         audioFeatures = [load_audio(data, self.audioPath, numFrames, audioAug = False, audioSet = audioSet)]
#         visualFeatures = [load_visual_predict(data, self.visualPath,numFrames, visualAug = False)]
#         minFrames = min([t.shape[0] for t in visualFeatures])
#         audioFeatures = [a[:4*minFrames, :] for a in audioFeatures]
#         visualFeatures = [v[:minFrames] for v in visualFeatures]
#         audio = torch.FloatTensor(numpy.array(audioFeatures))
#         faces = torch.FloatTensor(numpy.array(visualFeatures))
#         return audio, faces, data[0]

#     def __len__(self):
#         return len(self.miniBatch)


# def load_visual_predict(data, dataPath, numFrames, visualAug): 
#     trackid = data[0]
#     videoName = data[0][:36]
#     faceFolderPath = os.path.join(dataPath, videoName)
#     with open(f'/users/acp21jrc/audio-visual/active-speaker-detection/active_speaker/TalkNet_ASD/Ego4d_TalkNet_ASD/data/infer/bbox/{trackid}.json', 'r') as f:
#         bbox = json.load(f)
#     frames = bbox
#     faces = []
#     H = 112
#     if visualAug == True:
#         new = int(H*random.uniform(0.7, 1))
#         x, y = numpy.random.randint(0, H - new), numpy.random.randint(0, H - new)
#         M = cv2.getRotationMatrix2D((H/2,H/2), random.uniform(-15, 15), 1)
#         augType = random.choice(['orig', 'flip', 'crop', 'rotate']) 
#     else:
#         augType = 'orig'
#     dets = {'x':[], 'y':[], 's':[]}
#     for frame in frames:
#         frameid = frame['frame']
#         x1 = frame['x1']
#         y1 = frame['y1']
#         x2 = frame['x2']
#         y2 = frame['y2']
#         dets['s'].append(max((y2-y1), (x2-x1))/2) 
#         dets['y'].append((y2+y1)/2) # crop center x 
#         dets['x'].append((x2+x1)/2) # crop center y
#     kernel_size = min((len(dets['s'])-len(dets['s'])%2+1), 13)
#     dets['s'] = signal.medfilt(dets['s'], kernel_size=kernel_size)  # Smooth detections 
#     dets['x'] = np.array(dets['x'])
#     dets['x'][1:] = dets['x'][:-1]*0.8 + dets['x'][1:]*0.2
#     dets['y'] = np.array(dets['y'])
#     dets['y'][1:] = dets['y'][:-1]*0.8 + dets['y'][1:]*0.2

#     for i, frame in enumerate(frames): 
#         frameid = frame['frame'] = frame['frame']
#         img = cv2.imread(f'{faceFolderPath}/img_{int(frameid):05d}.jpg')
#         cs  = 0.4
#         bs  = dets['s'][i]
#         bsi = int(bs * (1 + 2 * cs))  # Pad videos by this amount 
#         img = numpy.pad(img, ((bsi,bsi), (bsi,bsi), (0, 0)), 'constant', constant_values=(110, 110))
#         my  = dets['y'][i] + bsi  # BBox center Y
#         mx  = dets['x'][i] + bsi  # BBox center X
#         face = img[int(my-bs):int(my+bs*(1+2*cs)),int(mx-bs*(1+cs)):int(mx+bs*(1+cs))]
#         face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
#         face = cv2.resize(face, (2*H,2*H))
#         face = face[int(112-(112/2)):int(112+(112/2)), int(112-(112/2)):int(112+(112/2))]

#         if augType == 'orig':
#             faces.append(face)
#         elif augType == 'flip':
#             faces.append(cv2.flip(face, 1))
#         elif augType == 'crop':
#             faces.append(cv2.resize(face[y:y+new, x:x+new] , (H,H))) 
#         elif augType == 'rotate':
#             faces.append(cv2.warpAffine(face, M, (H,H)))

#     faces = numpy.array(faces[:numFrames])
#     return faces


# class test_loader(object):   ###### for preprocessed tensor dataloading
#     def __init__(self, trialFileName, audioPath, visualPath, contextPath, bboxPath, **kwargs):
#         self.audioPath  = audioPath
#         self.visualPath = visualPath
#         self.contextPath = contextPath
#         self.bboxPath = bboxPath
#         self.miniBatch = open(trialFileName).read().splitlines()

#     def __getitem__(self, index):
#         line       = [self.miniBatch[index]]
#         numFrames  = int(line[0].split('\t')[1])
#         audioSet   = generate_audio_set(self.audioPath, line)        
#         data = line[0].split('\t')
#         flag = f'output/results/train_val/{data[0]}'
#         # if flag exists then return
#         # if os.path.isdir(flag):
#         #     return False, False, False, False, False, False
        
#         # os.mkdir(flag)

#         print('here is the data: ', data)
#         # if file exists, file = trackdir + trackid + '.json'
#         file = os.path.join('output/results/train_val', data[0] + '.json')
#         if os.path.isfile(file):
#             return False, False, False, False, False

#         #if data[0] != '3b79017c-4d42-40fc-a1bb-4a20bc8ebca7:track_13:0':
#         #    print('skipping')
#         #    return False, False, False, False, data[0]
#         audioFeatures = [load_audio(data, self.audioPath, numFrames, audioAug = False, audioSet = audioSet)]
#         visualFeatures = [load_visual(data, self.visualPath,numFrames, visualAug = False)]
#         contextFeatures = [load_context(data, self.contextPath, numFrames, visualAug = False)]
#         bboxes = [load_bboxes(data, self.bboxPath)]
#         minFrames = min([t.shape[0] for t in visualFeatures])
#         audioFeatures = [a[:4*minFrames, :] for a in audioFeatures]
#         visualFeatures = [v[:minFrames] for v in visualFeatures]
#         contextFeatures = [c[:minFrames] for c in contextFeatures]
#         bboxes = [b[:minFrames] for b in bboxes]
#         audio = torch.FloatTensor(numpy.array(audioFeatures))
#         faces = torch.FloatTensor(numpy.array(visualFeatures))
#         contextFeatures = torch.FloatTensor(numpy.array(contextFeatures))
#         bboxes = torch.FloatTensor(numpy.array(bboxes))
#         return audio, faces, contextFeatures, bboxes, data[0]

#     def __len__(self):
#         return len(self.miniBatch)


class test_loader(object):
    def __init__(self, annotPath, audioPath, dataPath, evalDataType, **kwargs):
        self.audioPath  = audioPath
        self.dataPath = os.path.join(dataPath, evalDataType)
        trialFileName = os.path.join(annotPath, 'csv', f'active_speaker_{evalDataType}.csv')
        self.miniBatch = open(trialFileName).read().splitlines()

    def __getitem__(self, index):
        t0 = time.time()
        line       = [self.miniBatch[index]]
        numFrames  = int(line[0].split('\t')[1])
        audioSet   = generate_audio_set(self.audioPath, line)        
        data = line[0].split('\t')
        audioFeatures = [load_audio(data, self.audioPath, numFrames, audioAug = False, audioSet = audioSet)]
        #visualFeatures, contextFeatures, bboxes = load_visual_predict(data, self.visualPath,numFrames, visualAug = False)
        visualFeatures = [load_visual(data, self.dataPath,numFrames, visualAug = False)]
        contextFeatures = [load_context(data, self.dataPath, numFrames, visualAug = False)]
        bboxes = [load_bboxes(data, self.dataPath)]
        #visualFeatures, contextFeatures, bboxes = [visualFeatures], [contextFeatures], [bboxes]
        minFrames = min([t.shape[0] for t in visualFeatures])
        audioFeatures = [a[:4*minFrames, :] for a in audioFeatures]
        visualFeatures = [v[:minFrames] for v in visualFeatures]
        contextFeatures = [c[:minFrames] for c in contextFeatures]
        bboxes = [b[:minFrames] for b in bboxes]
        audio = torch.FloatTensor(numpy.array(audioFeatures))
        faces = torch.FloatTensor(numpy.array(visualFeatures))
        context = torch.FloatTensor(numpy.array(contextFeatures))
        bboxes = torch.FloatTensor(numpy.array(bboxes))
        print('time for full data load iteration: ', time.time()-t0)
        return audio, faces, context, bboxes, data[0]

    def __len__(self):
        return len(self.miniBatch)


# def load_visual_predict(data, dataPath, numFrames, visualAug): 
#     trackid = data[0]
#     videoName = data[0][:36]
#     faceFolderPath = os.path.join(dataPath, videoName)
#     bboxFile = os.path.join(dataPath, 'infer', 'bbox', trackid + '.json')
#     with open(bboxFile, 'r') as f:
#         bbox = json.load(f)
#     frames = bbox
#     faces = []
#     images = []
#     bboxes = []
#     H = 112
#     if visualAug == True:
#         new = int(H*random.uniform(0.7, 1))
#         x, y = numpy.random.randint(0, H - new), numpy.random.randint(0, H - new)
#         M = cv2.getRotationMatrix2D((H/2,H/2), random.uniform(-15, 15), 1)
#         augType = random.choice(['orig', 'flip', 'crop', 'rotate']) 
#     else:
#         augType = 'orig'
#     dets = {'x':[], 'y':[], 's':[]}
#     for frame in frames:
#         frameid = frame['frame']
#         x1 = frame['x1']
#         y1 = frame['y1']
#         x2 = frame['x2']
#         y2 = frame['y2']
#         bboxes.append([x1, y1, x2, y2])
#         dets['s'].append(max((y2-y1), (x2-x1))/2) 
#         dets['y'].append((y2+y1)/2) # crop center x 
#         dets['x'].append((x2+x1)/2) # crop center y
#     kernel_size = min((len(dets['s'])-len(dets['s'])%2+1), 13)
#     dets['s'] = signal.medfilt(dets['s'], kernel_size=kernel_size)  # Smooth detections 
#     dets['x'] = np.array(dets['x'])
#     dets['x'][1:] = dets['x'][:-1]*0.8 + dets['x'][1:]*0.2
#     dets['y'] = np.array(dets['y'])
#     dets['y'][1:] = dets['y'][:-1]*0.8 + dets['y'][1:]*0.2

#     for i, frame in enumerate(frames): 
#         frameid = frame['frame'] = frame['frame']
#         img = cv2.imread(f'{faceFolderPath}/img_{int(frameid):05d}.jpg')
#         context = cv2.resize(img, (224, 224))
#         images.append(np.array(context.transpose(2,0,1)))
#         cs  = 0.4
#         bs  = dets['s'][i]
#         bsi = int(bs * (1 + 2 * cs))  # Pad videos by this amount 
#         img = numpy.pad(img, ((bsi,bsi), (bsi,bsi), (0, 0)), 'constant', constant_values=(110, 110))
#         my  = dets['y'][i] + bsi  # BBox center Y
#         mx  = dets['x'][i] + bsi  # BBox center X
#         face = img[int(my-bs):int(my+bs*(1+2*cs)),int(mx-bs*(1+cs)):int(mx+bs*(1+cs))]
#         face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
#         face = cv2.resize(face, (2*H,2*H))
#         face = face[int(112-(112/2)):int(112+(112/2)), int(112-(112/2)):int(112+(112/2))]

#         if augType == 'orig':
#             faces.append(face)
#         elif augType == 'flip':
#             faces.append(cv2.flip(face, 1))
#         elif augType == 'crop':
#             faces.append(cv2.resize(face[y:y+new, x:x+new] , (H,H))) 
#         elif augType == 'rotate':
#             faces.append(cv2.warpAffine(face, M, (H,H)))

#     faces = numpy.array(faces[:numFrames])
#     images = numpy.array(images[:numFrames])
#     bboxes = numpy.array(bboxes[:numFrames])
#     return faces, images, bboxes
