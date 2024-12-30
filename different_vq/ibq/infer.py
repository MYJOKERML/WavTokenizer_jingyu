# --coding:utf-8--
import os

from encoder.utils import convert_audio
import torchaudio
import torch
from decoder.pretrained import WavTokenizer
from tqdm import tqdm

import time

import logging

device1=torch.device('cuda:1')
# device2=torch.device('cpu')

input_path = "/data/lujingyu_data/datasets/LibriTTS/test.txt"
out_folder = 'result/infer'
# os.system("rm -r %s"%(out_folder))
# os.system("mkdir -p %s"%(out_folder))
# ll="libritts_testclean500_large"
# ll="wavtokenizer_smalldata_frame40_3s_nq1_code4096_dim512_kmeans200_attn_testclean_epoch34"

# tmptmp=out_folder+"/"+ll

# os.system("rm -r %s"%(tmptmp))
# os.system("mkdir -p %s"%(tmptmp))

# 自己数据模型加载
config_path = "/home1/lujingyu/projects/WavTokenizer_jingyu/different_vq/ibq/configs/config.yaml"
model_path = "/home1/lujingyu/projects/WavTokenizer_jingyu/ckpt/jingyu/ibq_epoch63.ckpt"
wavtokenizer = WavTokenizer.from_pretrained0802(config_path, model_path)
wavtokenizer = wavtokenizer.to(device1)
# wavtokenizer = wavtokenizer.to(device2)

with open(input_path,'r') as fin:
    x=fin.readlines()

libritts_root = '/data/lujingyu_data/datasets/LibriTTS'
x = [os.path.join(libritts_root, i.strip()) for i in x]
# 完成一些加速处理

features_all=[]

print("\033[32mstart encoding...\033[0m")
for i in tqdm(range(len(x))):

    wav, sr = torchaudio.load(x[i])
    # print("***:",x[i])
    wav = convert_audio(wav, sr, 24000, 1)                             # (1,131040)
    bandwidth_id = torch.tensor([0])
    wav=wav.to(device1)

    features,discrete_code= wavtokenizer.encode_infer(wav, bandwidth_id=bandwidth_id)
    features_all.append(features)

# wavtokenizer = wavtokenizer.to(device2)
print("\033[32mstart decoding...\033[0m")
for i in tqdm(range(len(x))):

    bandwidth_id = torch.tensor([0])

    bandwidth_id = bandwidth_id.to(device1) 

    audio_out = wavtokenizer.decode(features_all[i], bandwidth_id=bandwidth_id)   
    # print(i,time.time()) 
    # breakpoint()                        # (1, 131200)
    audio_path = out_folder + '/' + x[i].split('/')[-1]
    if not os.path.exists(out_folder + '/' + x[i].split('/')[-1]):
        os.makedirs(out_folder, exist_ok=True)
    # os.makedirs(out_folder + '/' + ll, exist_ok=True)
    torchaudio.save(audio_path, audio_out.cpu(), sample_rate=24000, encoding='PCM_S', bits_per_sample=16)

print("\033[32mfinish\033[0m")


