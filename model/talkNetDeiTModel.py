import torch
import torch.nn as nn

from model.audioEncoder      import audioEncoder
from model.visualEncoder     import visualFrontend, visualTCN, visualConv1D
from model.attentionLayer    import attentionLayer

from transformers import ViTModel

class talkNetDeiTModel(nn.Module):
    def __init__(self):
        super(talkNetDeiTModel, self).__init__()
        # Context Modelling Module
        self.vit_model = ViTModel.from_pretrained('facebook/deit-tiny-patch16-224')

        # MLP to project bbox to appropriate dimension
        self.fc1 = nn.Linear(4, 128)
        self.act = nn.Sigmoid()
        self.fc2 = nn.Linear(128, 192)
        self.drop = nn.Dropout(0.1)

        # cross attention to condition on position of scrutinised speaker
        self.crossAttention = attentionLayer(d_model = 192, nhead = 1)
        self.fc3 = nn.Linear(192, 64)
        

        # Visual Temporal Encoder
        self.visualFrontend  = visualFrontend() # Visual Frontend 
        # self.visualFrontend.load_state_dict(torch.load('visual_frontend.pt', map_location="cuda"))
        # for param in self.visualFrontend.parameters():
        #     param.requires_grad = False       
        self.visualTCN       = visualTCN()      # Visual Temporal Network TCN
        self.visualConv1D    = visualConv1D()   # Visual Temporal Network Conv1d

        # Audio Temporal Encoder 
        self.audioEncoder  = audioEncoder(layers = [3, 4, 6, 3],  num_filters = [16, 32, 64, 128])
        
        # Audio-visual Cross Attention
        self.crossA2V = attentionLayer(d_model = 128, nhead = 8)
        self.crossV2A = attentionLayer(d_model = 128, nhead = 8)

        # Audio-visual Self Attention
        self.selfAV = attentionLayer(d_model = 320, nhead = 8)

    def forward_context_frontend(self, images):
        output = self.vit_model(images)
        return output.last_hidden_state[:, 1:, :]
        
    def forward_bbox_frontend(self, bbox):
        bbox = self.fc1(bbox)
        bbox = self.act(bbox)
        bbox = self.fc2(bbox)
        bbox = self.drop(bbox)
        return bbox
    
    def forward_context_attention(self, emb_img, coords):
        x, attn_weights = self.crossAttention(src = emb_img, tar = coords)
        x = torch.mean(x, dim = 1)
        x = self.fc3(x)
        return x

    def forward_visual_frontend(self, x):
        B, T, W, H = x.shape  
        x = x.view(B*T, 1, 1, W, H)
        x = (x / 255 - 0.4161) / 0.1688
        x = self.visualFrontend(x)
        x = x.view(B, T, 512)        
        x = x.transpose(1,2)     
        x = self.visualTCN(x)
        x = self.visualConv1D(x)
        x = x.transpose(1,2)
        return x

    def forward_audio_frontend(self, x):    
        x = x.unsqueeze(1).transpose(2, 3)        
        x = self.audioEncoder(x)
        return x

    def forward_cross_attention(self, x1, x2):
        x1_c, A2V_weights = self.crossA2V(src = x1, tar = x2)
        x2_c, V2A_weights = self.crossV2A(src = x2, tar = x1)        
        return x1_c, x2_c

    def forward_audio_visual_backend(self, x1, x2, x3): 
        x = torch.cat((x1,x2, x3), 2) 
        x, attn_weights = self.selfAV(src = x, tar = x)    
        x = torch.reshape(x, (-1, 320))
        return x    

    def forward_audio_backend(self,x):
        x = torch.reshape(x, (-1, 128))
        return x

    def forward_visual_backend(self,x):
        x = torch.reshape(x, (-1, 128))
        return x
    
    def forward_context_backend(self,x):
        x = torch.reshape(x, (-1, 64))
        return x

