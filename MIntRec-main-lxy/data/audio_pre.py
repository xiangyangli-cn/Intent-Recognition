import logging
import os
import numpy as np
import pickle
import torch

__all__ = ['AudioDataset']

class AudioDataset:

    def __init__(self, args, base_attrs):
        
        self.logger = logging.getLogger(args.logger_name)
        audio_feats_path = os.path.join(base_attrs['data_path'], args.audio_data_path, args.audio_feats_path)
        
        if not os.path.exists(audio_feats_path):
            raise Exception('Error: The directory of audio features is empty.')
        
        self.feats = self.__load_feats(audio_feats_path, base_attrs)

        self.feats = self.__padding_feats(args, base_attrs)
        #exit()

    def __load_feats(self, audio_feats_path, base_attrs):

        self.logger.info('Load Audio Features Begin...')
        with open(audio_feats_path, 'rb') as f:
            print(audio_feats_path)
            #exit()
            audio_feats = pickle.load(f)
        #print(audio_feats)
        #audio_feats = audio_feats.astype(float)  #add lxy
        for ii  in audio_feats.keys():
            print(ii)
            print(audio_feats[ii].shape)
            print(audio_feats[ii].dtype)
            assert(audio_feats[ii].dtype==torch.float32)

            #exit()
        #exit()
        print(audio_feats['S04_E06_365'])
        #exit()
        
        train_feats = [audio_feats[x] for x in base_attrs['train_data_index']]
        dev_feats = [audio_feats[x] for x in base_attrs['dev_data_index']]
        test_feats = [audio_feats[x] for x in base_attrs['test_data_index']]  

        print(train_feats[0].shape) 
        print(train_feats[0].dtype) 
        print(len(train_feats))
        print(len(dev_feats))
        print(len(test_feats))
        #exit()

        self.logger.info('Load Audio Features Finished...')
        
        return {
            'train': train_feats,
            'dev': dev_feats,
            'test': test_feats
        }


    def __padding(self, feat, audio_max_length, padding_mode = 'zero', padding_loc = 'end'):
        """
        padding_mode: 'zero' or 'normal'
        padding_loc: 'start' or 'end'
        """
        assert padding_mode in ['zero', 'normal']
        assert padding_loc in ['start', 'end']

        audio_length = feat.shape[0]
        
        #exit()
        if audio_length >= audio_max_length:
            print("TTTTTT")
            print(audio_max_length)
            a=feat[0:audio_max_length, :]
            print(a.shape)
            #exit()
            return a.astype(np.float64)
            #return np.feat[audio_max_length, :]
            

        if padding_mode == 'zero':
            pad = np.zeros([audio_max_length - audio_length, feat.shape[-1]])
        elif padding_mode == 'normal':
            mean, std = feat.mean(), feat.std()
            pad = np.random.normal(mean, std, (audio_max_length - audio_length, feat.shape[1]))
        
        if padding_loc == 'start':
            feat = np.concatenate((pad, feat), axis = 0)
        else:
            feat = np.concatenate((feat, pad), axis = 0)

        return feat

    def __padding_feats(self, args, base_attrs):

        audio_max_length = base_attrs['benchmarks']['max_seq_lengths']['audio']
        print(audio_max_length)
        #exit()

        padding_feats = {}

        for dataset_type in self.feats.keys():
            print(dataset_type)
            print(args.padding_mode)
            print(args.padding_loc)
            #exit()
            feats = self.feats[dataset_type]

            tmp_list = []

            #i=0
            for feat in feats:
                #i=i+1
                feat = np.array(feat)
                #print(feat)
                #print(feat.dtype)
                assert(feat.dtype==np.float32)
                assert(feat.shape[1]==768)
                assert(feat.shape[0]>1)
                #print(i)
                #exit()
                #exit()
                padding_feat = self.__padding(feat, audio_max_length, padding_mode=args.padding_mode, padding_loc=args.padding_loc)
                tmp_list.append(padding_feat)

            padding_feats[dataset_type] = tmp_list

        return padding_feats   