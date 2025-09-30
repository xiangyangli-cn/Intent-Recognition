from moviepy.editor import *
from tqdm import tqdm
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import os
import argparse
import pickle
import argparse
import librosa
import torch

__all__ = ['AudioFeature']

def parse_arguments():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_video_path', type=str, default='MIA-datasets/raw_video', help="The directory of the raw video path.")
    parser.add_argument('--audio_data_path', type=str, default='data_preprocess/audio_data', help="The directory of the audio data path.")
    parser.add_argument('--raw_audio_path', type=str, default='data_preprocess/raw_audio', help="The directory of the raw audio path.")
    parser.add_argument("--audio_feats_path", type=str, default='audio_feats_test.pkl', help="The directory of audio features.")
    parser.add_argument("--lxy_audio_path", type=str, default='/data1/lyq/MIntRec/Audio', help="The directory of audio path")
    parser.add_argument("--lxy_audio_feats_path", type=str, default='lxy.pkl', help="The directory of audio features.")



    args = parser.parse_args()

    return args

class AudioFeature:
    
    def __init__(self, args):
        
        self.processor = Wav2Vec2Processor.from_pretrained("./cache/wav2vec2-base-960h")
        self.model = Wav2Vec2Model.from_pretrained("./cache/wav2vec2-base-960h")
        
        # self.__get_raw_audio(args)
    
        audio_feats = self.__gen_feats_from_audio(args, use_wav2vec2=True)
        self.__save_audio_feats(args, audio_feats)

            
    def __gen_feats_from_audio(self, args, use_wav2vec2=False):
    
        audio_feats = {}
        raw_audio_path = args.lxy_audio_path
        print(raw_audio_path)
        #exit()

        for s_path in tqdm(os.listdir(raw_audio_path),  desc = 'All'):
            print(s_path)
            #continue
            
            s_path_dir = os.path.join(raw_audio_path, s_path)
            print(s_path_dir)
            #continue

            ai=s_path_dir.find('.')
            bi=s_path_dir.rfind('/')
            print(ai)
            print(bi)
            #exit()
            audio_id = s_path_dir[bi+1:ai]
            print(audio_id)
            #continue
                    
            if use_wav2vec2:
                wav2vec2_feats = self.__process_audio(s_path_dir)
                audio_feats[audio_id] = wav2vec2_feats
            else:
                mfcc = self.__process_audio(s_path_dir)
                audio_feats[audio_id] = mfcc
        #exit()
        return audio_feats

    def __process_audio(self, read_file_path):  #obtain features
    
        y, sr = librosa.load(read_file_path, sr = 16000)
        audio_feats = self.processor(y, sampling_rate = sr, return_tensors="pt").input_values
        with torch.no_grad():
            audio_feats = self.model(audio_feats).last_hidden_state.squeeze(0)
        
        return audio_feats

    def __save_audio_feats(self, args, audio_feats): #save fature

        audio_feats_path = os.path.join(args.lxy_audio_feats_path)

        with open(audio_feats_path, 'wb') as f:
            pickle.dump(audio_feats, f)

if __name__ == '__main__':

    args = parse_arguments()
    audio_data = AudioFeature(args)