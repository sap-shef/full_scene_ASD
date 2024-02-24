# Modified original generate function found in Ego4d_TalkNet_ASD/utils/preprocessing.py
# to generate trackwise .json files containing bounding boxes of all faces present in each frame
# (including the candidate speaker), additionally modified the trackwise .json file for the candidadate speaker
# to include the pid of the candidate speaker for the track. This was necessary because vid:trackid:step 
# does not include pid information, vid refers to the parent video clip, trackid refers to the 
# contiguous track, and step refers to the iteration of the contiguous track, i.e. if track 0
# contains 850 contiguous frames, then there will be 3 steps intotal (300 frams, 300 frames, 250 frames).

# Assumes standard Ego4d_AVD directory configuration: Ego4d_TalkNet_ASD/data/split; Ego4d_TalkNet_ASD/data/json; Ego4D_TalkNet_ASD/data/track_results

# to run for train/val:
#   python annot_preprocessing.py --basePath {path to Ego4d_TalkNet_ASD} --split {train/val}
# to run for test (Ego4D-AVD reconfigured validation fold):
#   python annot_preprocessing.py --basePath {path to Ego4d_TalkNet_ASD} --split test


from tqdm import tqdm
from collections import defaultdict
import pandas as pd
import json
import argparse
import os

def generate_trainval(basePath, split):
    """
        Parameters:
            basePath (str): The directory containing the list of video clips and the av_{split}.json files
            split (str): The split of the dataset (val or test)
      """

    with open(f'{basePath}/data/json/av_{split}.json', 'r') as f:
        ori_annot = json.load(f)
    annotation = { c_annot['clip_uid']: c_annot for v_annot in ori_annot['videos'] for c_annot in v_annot['clips'] }

    with open(f'{basePath}/data/split/{split}.list', 'r') as f:
        videos = [line.strip() for line in f.readlines()]

    inconsistent_frames = 0
    total_frames = 0
    asd_records = []
    for video in tqdm(videos):
        active_speaker = defaultdict(list)
        tracks = defaultdict(list)
        for person in annotation[video]['persons']:
            if person['person_id'] == 'camera_wearer':
                continue
            segments = person['voice_segments']
            for segments in segments:
                active_speaker[person['person_id']].extend(list(range(segments['start_frame'], segments['end_frame']+1)))

            bboxes = person['tracking_paths']
            for bbox in bboxes:
                if 'visual_anchor' in bbox['track_id']:
                    continue
                tracks[bbox['track_id']] = { 'person': person['person_id'], 'frames': bbox['track'] }

        # build background speaker bboxes for each frame
        background_speaker_bboxes = defaultdict(list) #{frame: [bboxes (int(frame): 'x1', 'y1', 'x2', 'y2', 'pid)]}
        for track in tracks:
            pid = tracks[track]['person']
            frames = tracks[track]['frames']
            for frame in frames:
                label = int(frame['frame'] in active_speaker[pid])
                background_speaker_bboxes[int(frame['frame'])].append({'x1' : str(frame['x']), 'y1': str(frame['y']),
                                                                        'x2': str(float(frame['x']) + float(frame['width'])),
                                                                        'y2': str(float(frame['y']) + float(frame['height'])),
                                                                        'pid': int(pid),
                                                                        'label': label})

        for track in tracks:
            label = []
            record = []
            
            frames = tracks[track]['frames']
            
            # insert pid into each frame dictionary within frames
            for frame in frames:
                frame['pid'] = tracks[track]['person']

            for frame in frames:
                label.append(int(frame['frame'] in active_speaker[tracks[track]['person']]))
                # add label to frame dictionary
                frame['activity'] = label[-1]
                    
            step = list(range(0, (frames[-1]['frame']-frames[0]['frame']+1), 300))
            track_id = video+':'+track
            for i, start in enumerate(step):
                
                if len(frames[start:start+300]) == 0:
                    continue
                if len(frames[start:start+300]) > 1:
                    record.append([track_id+':'+str(i), len(frames[start:start+300]), 30.0, label[start:start+300], frames[start]['frame']])
                with open(f'{basePath}/data/ego4d/bbox/{track_id}:{i}.json', 'w+') as f:
                    json.dump(frames[start:start+300], f)

                track_background_speaker_bboxes = defaultdict(list)
                pid = tracks[track]['person']
                for frame in frames[start:start+300]:
                    frameid = int(frame['frame'])
                    orig = background_speaker_bboxes[int(frameid)]
                    track_background_speaker_bboxes[frameid] = [p for p in background_speaker_bboxes[int(frameid)] if int(p['pid']) != int(pid)]
                    modified = track_background_speaker_bboxes[frameid]
                    if len(orig) != len(modified)+1:
                        print('Caution: candidate speaker was not in bboxes_in_track')
                        inconsistent_frames += 1
                    total_frames += 1    
                
                with open(f'{basePath}/data/ego4d/bboxes_per_track/{track_id}:{i}.json', 'w+') as f:
                    json.dump(track_background_speaker_bboxes, f)
                
            asd_records.extend(record)
    print(f'Inconsistent frames: {inconsistent_frames} out of {total_frames} total frames')
    asd_records = pd.DataFrame(asd_records)
    asd_records.to_csv(f'{basePath}/data/ego4d/csv/active_speaker_{split}.csv', header=None, index=False, sep='\t')

def generate_infer(basePath):
    asd_records = []

    with open(f'{basePath}/data/track_results/v.txt', 'r') as f:
            videos = [line.strip().split('.')[0] for line in f.readlines()]

    res2video = { i:video for i, video in enumerate(videos) }
    video2res = { video:i for i, video in enumerate(videos) }
    with open(f'{basePath}/data/split/val.list', 'r') as f: ## needs changing to test for actual Ego4D inference
        videos = [line.strip() for line in f.readlines()]
    tracklets = [f'{basePath}/data/track_results/{v}.txt' for v in videos]
    for tracklet in tqdm(sorted(tracklets)):
        if not os.path.exists(tracklet):
            print(f'{tracklet} does not exist in directory.')
            continue
        global_tracks = defaultdict(list)
        with open(tracklet, 'r') as f:
            res = f.readlines()
        for record in res:
            frame, pid, x1, y1, x2, y2, activity = record.split()
            global_tracks[pid].append({
                'frame': int(frame), 
                'x1': int(x1), 
                'y1': int(y1), 
                'x2': int(x2), 
                'y2': int(y2),
                'pid': int(pid), 
                'video': tracklet.split('/')[-1][:-4],
                'activity': int(activity)
            })

        # build background speaker bboxes for each frame
        background_speaker_bboxes = defaultdict(list) #{frame: [bboxes (int(frame): 'x1', 'y1', 'x2', 'y2', 'pid)]}
        for pid in global_tracks:
            frames = global_tracks[pid]
            for frame in frames:
                background_speaker_bboxes[int(frame['frame'])].append({'x1': str(frame['x1']), 'y1': str(frame['y1']),
                                                                    'x2': str(frame['x2']), 'y2': str(frame['y2']),
                                                                    'pid': int(pid), 'label': str(frame['activity'])})

        local_tracks = defaultdict(list)
        for pid, frames in global_tracks.items():
            count = -1
            last_frame = -2
            track_length = 0
            frames.sort(key=lambda x:x['frame'])
            for f in frames:
                if (f['frame'] > last_frame + 1) or (track_length > 300):
                    count += 1
                    track_length = 0
                last_frame = f['frame']
                video = f['video']
                trackid = f'{video}:{pid}:{count}'
                #print('trackid:', trackid)
                f.pop('video')
                local_tracks[trackid].append(f)
                track_length += 1
        
        for track_id, frames in local_tracks.items():
            bboxes = {}
            for frame in frames:
                bboxes[str(frame['frame'])] = background_speaker_bboxes[frame['frame']]
            with open(f'{basePath}/data/infer/bboxes_per_track/{track_id}.json', 'w+') as f:
                json.dump(bboxes, f)
        for track_id, frames in local_tracks.items():
            with open(f'{basePath}/data/infer/bbox/{track_id}.json', 'w+') as f:
                json.dump(frames, f)
                
            record = []
            # [trackid (video+trackid),  length of tracklets,  fps,  labels,  frame]
            record.append([track_id, len(frames), 30.0, [0], frames[0]['frame']])
            asd_records.extend(record)


    asd_records = pd.DataFrame(asd_records)
    asd_records.to_csv(f'{basePath}/data/infer/csv/active_speaker_val.csv', header=None, index=False, sep='\t')


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--basePath', type=str, 
                        default=f'/users/acp21jrc/audio-visual/active-speaker-detection/active_speaker/TalkNet_ASD/Ego4d_TalkNet_ASD',
                        help = 'path to Ego4d_TalkNet_ASD (assumes standard Ego4D-AVD directory structure)')
    parser.add_argument('--split', type=str, default='train', help='train or val or test')
    args = parser.parse_args()
    if args.split == 'train' or args.split == 'val':
        os.makedirs(f'{args.basePath}/data/ego4d/csv', exist_ok=True)
        os.makedirs(f'{args.basePath}/data/ego4d/bbox', exist_ok=True)
        os.makedirs(f'{args.basePath}/data/ego4d/bboxes_per_track', exist_ok=True)
        generate_trainval(args.basePath, args.split)
    elif args.split == 'test':
        os.makedirs(f'{args.basePath}/data/infer/csv', exist_ok=True)
        os.makedirs(f'{args.basePath}/data/infer/bbox', exist_ok=True)
        os.makedirs(f'{args.basePath}/data/infer/bboxes_per_track', exist_ok=True)
        generate_infer(args.basePath)
    

if __name__ == "__main__":
    run()
