# --coding:utf-8--
import os

from encoder.utils import convert_audio
import torchaudio
import torch
from decoder.pretrained import WavTokenizer

import time

import logging

device1=torch.device('cuda:1')
# device2=torch.device('cpu')

# input_path = "./WavTokenizer/data/infer/lirbitts_testclean"
out_folder = '/home1/lujingyu/projects/WavTokenizer_jingyu/result/infer'
# os.system("rm -r %s"%(out_folder))
# os.system("mkdir -p %s"%(out_folder))
# ll="libritts_testclean500_large"
# ll="wavtokenizer_smalldata_frame40_3s_nq1_code4096_dim512_kmeans200_attn_testclean_epoch34"

# tmptmp=out_folder+"/"+ll

# os.system("rm -r %s"%(tmptmp))
# os.system("mkdir -p %s"%(tmptmp))

# 自己数据模型加载
config_path = "/data/lujingyu_data/checkpoints/WavTokenizer-medium-music-audio-75token/wavtokenizer_mediumdata_music_audio_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml"
model_path = "/data/lujingyu_data/checkpoints/WavTokenizer-medium-music-audio-75token/wavtokenizer_medium_music_audio_320_24k.ckpt"
wavtokenizer = WavTokenizer.from_pretrained0802(config_path, model_path)
wavtokenizer = wavtokenizer.to(device1)
# wavtokenizer = wavtokenizer.to(device2)

# with open(input_path,'r') as fin:
#     x=fin.readlines()

x = ['/home1/lujingyu/projects/WavTokenizer_jingyu/data/howmanydays.wav']

# 完成一些加速处理

features_all=[]

for i in range(len(x)):

    wav, sr = torchaudio.load(x[i])
    # print("***:",x[i])
    wav = convert_audio(wav, sr, 24000, 1)                             # (1,131040)
    print(wav.shape)
    bandwidth_id = torch.tensor([0])
    wav=wav.to(device1)
    print(i)

    features,discrete_code= wavtokenizer.encode_infer(wav, bandwidth_id=bandwidth_id)
    features_all.append(features)

# wavtokenizer = wavtokenizer.to(device2)

for i in range(len(x)):

    bandwidth_id = torch.tensor([0])

    bandwidth_id = bandwidth_id.to(device1) 

    print(i)
    audio_out = wavtokenizer.decode(features_all[i], bandwidth_id=bandwidth_id)   
    # print(i,time.time()) 
    # breakpoint()                        # (1, 131200)
    audio_path = out_folder + '/' + x[i].split('/')[-1]
    print(wav.shape)
    print(audio_out.shape)
    if not os.path.exists(out_folder + '/' + x[i].split('/')[-1]):
        os.makedirs(out_folder, exist_ok=True)
    # os.makedirs(out_folder + '/' + ll, exist_ok=True)
    torchaudio.save(audio_path, audio_out.cpu(), sample_rate=24000, encoding='PCM_S', bits_per_sample=16)





