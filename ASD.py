import torch, math
import torch.nn as nn
import torch.nn.functional as F

from torch.cuda.amp import autocast, GradScaler


import sys, time, numpy, os, tqdm, json

from sklearn.metrics import average_precision_score

from loss import lossAVC, lossA, lossV, lossC
from model.talkNetDeiTModel import talkNetDeiTModel

class ASD(nn.Module):
    def __init__(self, lr = 0.0001, lrDecay = 0.95, device = 'cpu', **kwargs):
        #wandb.init(project="trainTalkNet-sample_loop")
        super(ASD, self).__init__()    
        self.scaler = GradScaler()
        self.device = device    
        self.contextLossWeight = kwargs['contextLossContribution']
        self.model = talkNetDeiTModel().to(device)
        self.lossAVC = lossAVC().to(device)
        self.lossA = lossA().to(device)
        self.lossV = lossV().to(device)
        self.lossC = lossC().to(device)
        self.optim = torch.optim.Adam(self.parameters(), lr = lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optim, step_size = 1, gamma=lrDecay)
        print(time.strftime("%m-%d %H:%M:%S") + " Model para number = %.2f"%(sum(param.numel() for param in self.model.parameters()) / 1024 / 1024))

    def train_network(self, loader, epoch, **kwargs):
        self.train()
        self.scheduler.step(epoch - 1)
        index, top1, loss, lossC = 0, 0, 0, 0
        lr = self.optim.param_groups[0]['lr']  
        cumulative_mem = 0      
        for num, (audioFeature, visualFeature, contextFeatures, bboxes, labels) in enumerate(tqdm.tqdm(loader), start=1):
            self.zero_grad()
            contextFeatures, bboxes = contextFeatures.squeeze(0), bboxes.squeeze(0)
            batchsize, streamlength = contextFeatures.shape[:2]
            # print('************************************************')
            # print('audio features shape', audioFeature.shape, audioFeature.dtype)
            # print('video Feature shape', visualFeature.shape, visualFeature.dtype)
            # print('contexts shape', contextFeatures.shape, contextFeatures.dtype)
            # print('bboxes shape', bboxes.shape, bboxes.dtype)
            # print('labels shape', labels.shape, labels.dtype)
            #print('************************************************')
        
            with autocast(dtype = torch.float16):
                # context modelling operations
                contextFeatures = contextFeatures.reshape((-1, 3, 224, 224))
                bboxes = bboxes.reshape((-1, 4)) # (batchsize*streamlength, 4)
                emb_img = self.model.forward_context_frontend(contextFeatures.to(self.device))
                coords = self.model.forward_bbox_frontend(bboxes.unsqueeze(1).to(self.device))
                contextEmbed = self.model.forward_context_attention(emb_img, coords) # (batchsize*streamlength, 64)
                contextEmbed = contextEmbed.reshape((batchsize, streamlength, -1)) # (batchsize, streamlength, 64)
                
                # audio visual modelling of candidate speaker operations
                audioEmbed = self.model.forward_audio_frontend(audioFeature[0].to(self.device)) # feedForward
                visualEmbed = self.model.forward_visual_frontend(visualFeature[0].to(self.device))
                audioEmbed, visualEmbed = self.model.forward_cross_attention(audioEmbed, visualEmbed)
                
                # modality fusion operations
                outsAVC= self.model.forward_audio_visual_backend(audioEmbed, visualEmbed, contextEmbed)  
                
                # back end operations for losses
                outsA = self.model.forward_audio_backend(audioEmbed)
                outsV = self.model.forward_visual_backend(visualEmbed)
                outsC = self.model.forward_context_backend(contextEmbed)
                labels = labels[0].reshape((-1)).to(self.device) # Loss
                nlossAVC, _, _, prec = self.lossAVC.forward(outsAVC, labels)
                nlossA = self.lossA.forward(outsA, labels)
                nlossV = self.lossV.forward(outsV, labels)
                nlossC = self.lossC.forward(outsC, labels)
                
            nloss = nlossAVC + 0.4 * nlossA + 0.4 * nlossV + self.contextLossWeight * nlossC
            loss += nloss.detach().cpu().numpy()
            lossC += nlossC.detach().cpu().numpy()
            top1 += prec

            # (without autocast)            
            #nloss.backward()
            #self.optim.step()
            
            # (with autocast)
            self.scaler.scale(nloss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
            
            index += len(labels)
            sys.stderr.write(time.strftime("%m-%d %H:%M:%S") + \
            " [%2d] Lr: %5f, Training: %.2f%%, "    %(epoch, lr, 100 * (num / loader.__len__())) + \
            " Loss: %.5f, ACC: %2.2f%% \r"        %(loss/(num), 100 * (top1/index)))
            sys.stderr.flush()  
        sys.stdout.write("\n")      
        return loss/num, lossC, lr

    
    def evaluate_network(self, loader, **kwargs):
        self.eval()
        predScores = []
        gtLabels = []
        top1 = 0
        index = 0
        for audioFeature, visualFeature, contextFeatures, bboxes, labels in tqdm.tqdm(loader):
            with torch.no_grad():       
                contextFeatures, bboxes = contextFeatures.squeeze(0), bboxes.squeeze(0)
                batchsize, streamlength = contextFeatures.shape[:2]
                if audioFeature[0].shape[1] == 0 or contextFeatures.shape[1] != bboxes.shape[1]:
                    continue   
                
                # context modelling operations
                contextFeatures = contextFeatures.reshape((-1, 3, 224, 224))
                bboxes = bboxes.reshape((-1, 4)) # (batchsize*streamlength, 4)
                emb_img = self.model.forward_context_frontend(contextFeatures.to(self.device))
                coords = self.model.forward_bbox_frontend(bboxes.unsqueeze(1).to(self.device))
                contextEmbed = self.model.forward_context_attention(emb_img, coords) # (batchsize*streamlength, 64)
                contextEmbed = contextEmbed.reshape((batchsize, streamlength, -1)) # (batchsize, streamlength, 64)

                # audio visual modelling of candidate speaker operations
                audioEmbed  = self.model.forward_audio_frontend(audioFeature[0].to(self.device))
                visualEmbed = self.model.forward_visual_frontend(visualFeature[0].to(self.device))
                audioEmbed, visualEmbed = self.model.forward_cross_attention(audioEmbed, visualEmbed)
                
                # modality fusion operations
                outsAVC = self.model.forward_audio_visual_backend(audioEmbed, visualEmbed, contextEmbed)  
                
                labels = labels[0].reshape((-1)).to(self.device)             
                _, predScore, _, prec = self.lossAVC.forward(outsAVC, labels)
                top1 += prec 
                index += len(labels)
                predScore = predScore[:,1].detach().cpu().numpy()
                predScores.extend(predScore)
                gtLabels.extend(labels.detach().cpu().numpy())
    
    	# bAP
        bAP = average_precision_score(gtLabels, predScores)
       
        return 100 * (top1/index), bAP

    def saveParameters(self, path):
        torch.save(self.state_dict(), path)

    def loadParameters(self, path):
        selfState = self.state_dict()
        loadedState = torch.load(path, map_location=torch.device(self.device))
        for name, param in loadedState.items():
            origName = name
            if name not in selfState:
                name = name.replace("module.", "")
                if name not in selfState:
                    print("%s is not in the model."%origName)
                    continue
            if selfState[name].size() != loadedState[origName].size():
                sys.stderr.write("Wrong parameter length: %s, model: %s, loaded: %s"%(origName, selfState[name].size(), loadedState[origName].size()))
                continue
            selfState[name].copy_(param)


    def predict_network(self, loader, **kwargs): #loader is torch loader
        self.eval() #evaluation mode
        os.makedirs('output/results', exist_ok=True) 
        for audioFeature, visualFeature, contextFeature, bboxes, trackid in tqdm.tqdm(loader): #use __getitem__ method in loader class as iterable
            t0 = time.time()
            if trackid == False:
                print('already exists therefore skipped')
                continue
            durationSet = {1,1,1,2,2,2,3,3,4,5,6} # Use this line can get more reliable result
            
            necessary_value = contextFeature.shape[2]
            if bboxes.shape[2] != necessary_value:
                to_pad_by = necessary_value - bboxes.shape[2]
                target = bboxes[:, :, -to_pad_by:, :] 
                bboxes = torch.cat((bboxes, target), dim=2) # concatenate along the third dimension
            
            videoFeature = numpy.array(visualFeature[0, 0, ...])
            audioFeature = audioFeature[0, 0, ...]
            contextFeature = contextFeature[0, 0, ...]
            bboxes = bboxes[0, 0, ...] 

            length = min((audioFeature.shape[0] - audioFeature.shape[0] % 4) / (100*30/25), videoFeature.shape[0])
            allScore = [] # Evaluation use TalkNet
            for duration in durationSet:
                batchSize = int(math.ceil(length / duration))
                scores = []
                with torch.no_grad():
                    for i in range(batchSize):
                        inputA = torch.FloatTensor(audioFeature[int(i * duration * (100*30/25)): int((i+1) * duration * (100*30/25)), :]).unsqueeze(0).to(self.device)
                        inputV = torch.FloatTensor(videoFeature[i * duration * 30: (i+1) * duration * 30,:,:]).unsqueeze(0).to(self.device)
                        inputC = torch.FloatTensor(contextFeature[i * duration * 30: (i+1) * duration * 30,:,:,:]).to(self.device)
                        inputB = torch.FloatTensor(bboxes[i * duration * 30: (i+1) * duration * 30,:]).to(self.device)
                        batchsize, streamlength = inputV.shape[0], inputV.shape[1]
            
                        # context modelling operations
                        emb_img = self.model.forward_context_frontend(inputC.to(self.device))
                        coords = self.model.forward_bbox_frontend(inputB.unsqueeze(1).to(self.device))
                        contextEmbed = self.model.forward_context_attention(emb_img, coords) # (batchsize*streamlength, 64)
                        contextEmbed = contextEmbed.reshape((batchsize, streamlength, -1)) # (batchsize, streamlength, 64)
 
                        # audio visual modelling operations
                        embedA = self.model.forward_audio_frontend(inputA)
                        embedV = self.model.forward_visual_frontend(inputV)
                        embedA, embedV = self.model.forward_cross_attention(embedA, embedV) 
                        
                        # # modality fusion operations	
                        out = self.model.forward_audio_visual_backend(embedA, embedV, contextEmbed)
                        score = self.lossAVC.forward(out, labels = None)
                        scores.extend(score)
                allScore.append(scores)
            allScore = numpy.round((numpy.mean(numpy.array(allScore), axis = 0)), 1).astype(float)
            with open(f'{kwargs["annotPath"]}/bbox/{trackid[0]}.json', 'r') as f:
                bbox = json.load(f)
            for i, frame in enumerate(bbox):
                frame['score'] = str(allScore[i])
                frame['label'] = int(allScore[i].item()>-3.0)
            with open(f'output/results/{trackid[0]}.json', 'w+') as f:
                json.dump(bbox, f)
            print('inference time taken: ', time.time()-t0)
 