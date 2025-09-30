from torch.utils.data import Dataset
import torch
import numpy as np

__all__ = ['MMDataset']

class MMDataset(Dataset):
        
    def __init__(self, label_ids, text_feats, video_feats, audio_feats):
        
        self.label_ids = torch.tensor(label_ids)
        self.text_feats = torch.tensor(text_feats)
        self.video_feats = torch.tensor(np.array(video_feats))
        #print(audio_feats.type)
        #print(len(audio_feats))
        #print(audio_feats[0].type())
        print(audio_feats)
        print(len(audio_feats))
        i=0
        for ii in audio_feats:
            print(i)
            i=i+1
            assert(ii.dtype==np.float64)
            print(ii.shape)
            print(ii)
            assert(ii.shape[0]==480)  
            assert(ii.shape[1]==768)  

        print(audio_feats[0].shape)
        print(audio_feats[0].dtype)
        print(len(audio_feats))
        #print(audio_feats[len(audio_feats)-1])
        #exit()
        self.audio_feats = torch.tensor(np.array(audio_feats))
        #exit()
        #self.audio_feats = torch.tensor(audio_feats)

        self.size = len(self.text_feats)

    def __len__(self):
        return self.size

    def __getitem__(self, index):

        sample = {
            'label_ids': self.label_ids[index], 
            'text_feats': self.text_feats[index],
            'video_feats': self.video_feats[index],
            'audio_feats': self.audio_feats[index]
        } 
        return sample