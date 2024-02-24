#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Usage: python postprocess_eval.py

import glob, json, os
from argparse import ArgumentParser

import numpy as np
from sklearn.metrics import average_precision_score


def generate_results(track_direc, split, save_res):
    '''
    Takes the trackwise json files from the ASD output and aggregates them by video clip and saves as 
    .txt file (of the form: frame pid x1 y1 x2 y2 confidence label) for use by Ego4D evaluation code.
    
    Assumes ASD output is in the format vid_id:trackid:step.json (where step refers to the contiguous tracklet number
    due to the frame cap).

    Assumes track_direc is in format vid_id.txt

        Parameters:
            track_direc (str): The directory containing the tracking results
            split (str): The split of the dataset (val or test)
            save_res (bool): Whether to save the results in the format of the ground truth for usage of Ego4D evaluation code

        Returns:
            hyps (list): List of confidence scores for all tracklets in the given evaluation set
    '''
    
    with open(f'v.txt', 'r') as f:
        vid_ids = f.readlines()
    vid_ids = [v.strip().split('.')[0] for v in vid_ids]

    with open(f'{split}.list', 'r') as f:
        videos = [line.strip() for line in f.readlines()]
    tracklets = [f'{track_direc}track_results/{v}.txt' for v in videos] # accumulated tracklets per video (no ASD)

    i = 0
    hyp_results = {}
    for t in tracklets: # accumulated tracklets for single video (no ASD)
        if not os.path.exists(t):
            print('tracking file {t} does not exist')
            continue
        trackid = t.split('/')[-1][:-4]
        vid_num = vid_ids.index(str(trackid))
        asdres = glob.glob(f'output/results/{trackid}*.json') # all the tracklets for a video (output of the ASD system)
        print('length of asdres', len(asdres))
        pidre = {} # asd output for a single video
        for asd in asdres: # single tracklet output of ASD system
            print(asd)
            with open(asd, 'r') as f:
                lines = json.load(f)
                for line in lines:
                    print(line)
                    identifier = '{}:{}'.format(line['frame'], line['pid'])
                    pidre[identifier] = line # {frame:pid: {frame, x1, y1, x2, y2, pid, score, label}}
        with open(t, 'r') as f: 
            lines = f.readlines() # all tracking results for a video (no ASD)
        
        new_lines = []
        for line in lines: # frame pid x1 y1 x2 y2
            line = ' '.join(line.split()[:-1]) # removes the last element (i.e. in the ground truth tracking label)
            data = line.split()
            identifier = '{}:{}'.format(data[0], data[1])
            if identifier in pidre: # if data is in ASD results (i.e. ASD has a prediction for this tracklet)
                new_lines.append('{} {} {}\n'.format(line, pidre[identifier]['score'], pidre[identifier]['label'])) 
            else:
                i += 1
                print(t, line)
                new_lines.append('{} {} {}\n'.format(line, 0, 0))
        hyp_results[vid_num] = [float(line.split(' ')[-2]) for line in new_lines] # confidence scores
        if save_res:
            with open(f'output/final/{vid_num}.txt', 'w+') as f:
                f.writelines(new_lines) # (frame, pid, x top left, y top left, x bottom right, y bottom right, confidence score, label)
    
    print('total files missing from ASD system output: ', i)
    
    hyps = []
    for key in sorted(hyp_results.keys()):
        hyps.extend(hyp_results[key])
    return hyps

def extract_gt(gtPath):
    gt_files = sorted(glob.glob(gtPath + '/*.txt'))
    gts = []
    for gt_file in gt_files:
        gt = np.loadtxt(gt_file)
        gts.extend(gt[:,-1])
    return gts

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--evalDataType', type=str, default="val", help='Choose the dataset for evaluation, val or test')
    parser.add_argument('--save_res', type=int, default='0', help='Save the results in the format of the ground truth')
    parser.add_argument('--trackPath', type = str, default = '/users/acp21jrc/audio-visual/active-speaker-detection/active_speaker/TalkNet_ASD/Ego4d_TalkNet_ASD/data/', help='Path to the tracking results')
    parser.add_argument('--gtPath', type = str, default='/mnt/parscratch/users/acp21jrc/ego4d_results/asd_results/ground-truth-reformatted', help='Path to the ground truth ASD results')
    args = parser.parse_args()
    args.save_res = bool(args.save_res)
    hyps = generate_results(args.trackPath, args.evalDataType, args.save_res)
    gts = extract_gt(args.gtPath)
    # assert same length
    assert len(hyps) == len(gts), 'length of hyps and gts are not the same: {} vs {}'.format(len(hyps), len(gts))
    print('average gt: ', np.mean(gts), '+-', np.std(gts))
    print('average hyp: ', np.mean(hyps), '+-', np.std(hyps))
    print('average precision score: ', average_precision_score(gts, hyps))