# Usage: python inferTalkNetDeiT.py --nDataLoaderThread 12

import torch, argparse, warnings

from dataLoader import test_loader
from ASD import ASD

def main():
    # The structure of this code is learnt from https://github.com/clovaai/voxceleb_trainer
    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser(description = "TalkNet Training")
    # # Training details
    
    parser.add_argument('--nDataLoaderThread', type=int, default=4,  help='Number of loader threads')
    # Data path
    parser.add_argument('--annotPath',  type=str, default="/users/acp21jrc/audio-visual/active-speaker-detection/active_speaker/TalkNet_ASD/Ego4d_TalkNet_ASD/data/infer", help='Save path of Ego4d dataset')
    parser.add_argument('--audioPath',   type=str, default="/mnt/parscratch/users/acp21jrc/ego4d_data/v2/data/wave", help='Save path of audio features')
    parser.add_argument('--dataPath',  type=str, default="/mnt/parscratch/users/acp21jrc/ego4d_data/v2/data/tensors/")
    # Data selection
    parser.add_argument('--evalDataType', type=str, default="val", help='Choose the dataset for evaluation, val or test')
    parser.add_argument('--checkpoint',  type=str, default="Ego4D_best.model", help='Model checkpoint')
    parser.add_argument('--contextLossContribution', type=float, default=0.4, help='The contribution of context loss')
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('available device:', device)

    args = parser.parse_args()
    
    loader = test_loader(annotPath      = args.annotPath, \
                         audioPath      = args.audioPath, \
                         dataPath       = args.dataPath, 
                         evalDataType   = args.evalDataType)

    testLoader = torch.utils.data.DataLoader(loader, batch_size = 1, shuffle = False, num_workers = args.nDataLoaderThread)
   
    s = ASD(device = device, **vars(args)) #create instance of TalkNet (infrastructure not architecture)
    pretrained_model = args.checkpoint
    s.loadParameters(pretrained_model) #load parameters from checkpoint
    print("Model %s loaded from previous state!"%(pretrained_model)) 
    
    s.predict_network(loader = testLoader, **vars(args)) #inference

if __name__ == '__main__':
    main()
