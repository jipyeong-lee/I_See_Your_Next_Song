import os
import re
import streamlit as st
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import time
import base64
import json
from pydub import AudioSegment
from pathlib import Path
from spleeter.separator import Separator
import shutil
from PIL import Image
# extract_one 실행하기 위한 import
import json
import os
import time
import tensorboard_logger
import tqdm
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import models
from crnn import CRNN2D_elu2
from dataset import DefaultSet
from utils import AverageMeter
import argparse
import ast
import datetime
import sys
import torchaudio
# dataset.py 
import bisect
import csv
import functools
import random
torchaudio.set_audio_backend('soundfile')
from torch.nn import functional as F
from torch.utils.data import Dataset
from torchaudio import sox_effects, transforms
#TJAE
import math
import torch.nn as nn
from tqdm.notebook import tqdm
import glob
import warnings
from torch.utils.data import Dataset, DataLoader
warnings.filterwarnings(action='ignore')
# 유사도
from numpy import dot
from numpy.linalg import norm
from scipy.stats import pearsonr
import umap

def documentation():
    st.subheader('''
    실행 방법 <Music Recommendation System>
    ''')

    st.write('''
    노래를 10초 이상 부르신 다음 음악 파일을 업로드만 하면 자신의 음색에 어울리는 노래 추천 :)
    ''')
    st.write('#### Task => Run Program 이동!')
    main_img = Image.open('4singer_image.png')
    st.image(main_img)

if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')

#charlie
############# 원래 ##############
class TimeAutoEncoder(nn.Module):
    def __init__(self):
        super(TimeAutoEncoder, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels = 48, out_channels = 512, kernel_size = 3, stride = 1, padding = 0, dilation = 1),
            #nn.BatchNorm1d(512),
            nn.ReLU(),
        )
    
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels = 512, out_channels = 256, kernel_size = 3, stride = 1, padding = 0, dilation = 2),
            #nn.BatchNorm1d(256),
            nn.ReLU(),
        )
            
        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels = 256, out_channels = 64, kernel_size = 3, stride = 1, padding = 0, dilation = 4),
            #nn.BatchNorm1d(128),
            nn.ReLU(),
        )

        self.conv4 = nn.Sequential(
            nn.Conv1d(in_channels = 128, out_channels = 64, kernel_size = 3, stride = 1, padding = 0, dilation = 8),
            #nn.BatchNorm1d(64),
            nn.ReLU(),
        )

        self.conv5 = nn.Sequential(
            nn.Conv1d(in_channels = 64, out_channels = 16, kernel_size = 3, stride = 1, padding = 0, dilation = 16),
            #nn.BatchNorm1d(32),
            nn.ReLU(),
        )

        self.conv6 = nn.Sequential(
            nn.Conv1d(in_channels = 32, out_channels = 16, kernel_size = 3, stride = 1, padding = 0, dilation = 32),
            #nn.BatchNorm1d(16),
            nn.ReLU(),
        )

        self.conv7 = nn.Sequential(
            nn.Conv1d(in_channels = 16, out_channels = 4, kernel_size = 3, stride = 1, padding = 0, dilation = 64),
            #nn.BatchNorm1d(8),
            nn.ReLU(),
        )

        self.encoder_fc = nn.Sequential(
            nn.Linear(8 * 7501, 256),
            #nn.BatchNorm1d(128),
            nn.LeakyReLU(),
        )
        
        self.decoder_fc = nn.Sequential(
            nn.Linear(256, 8 * 7501),
            nn.ReLU(),
        )

        self.t_conv1 = nn.Sequential(
            # nn.ConvTranspose1d(in_channels = 8, out_channels = 16, kernel_size  = 3, stride = 1, dilation=62),
            nn.Conv1d(in_channels = 16, out_channels = 16, kernel_size = 3, stride = 1, padding = 0, dilation = 1),
            #nn.BatchNorm1d(16),
            nn.ReLU(),
        )

        self.t_conv2 = nn.Sequential(
            # nn.ConvTranspose1d(in_channels = 16, out_channels = 32, kernel_size  = 3, stride = 1, dilation = 30),
            nn.Conv1d(in_channels = 32, out_channels = 32, kernel_size = 3, stride = 1, padding = 0, dilation = 2),
            #nn.BatchNorm1d(32),
            nn.ReLU(),
        )

        self.t_conv3 = nn.Sequential(
            # nn.ConvTranspose1d(in_channels = 32, out_channels = 64, kernel_size  = 3, stride = 1, dilation=14),
            nn.Conv1d(in_channels = 64, out_channels = 64, kernel_size = 3, stride = 1, padding = 0, dilation = 4),
            #nn.BatchNorm1d(64),
            nn.ReLU(),
        )

        self.t_conv4 = nn.Sequential(
            # nn.ConvTranspose1d(in_channels = 64, out_channels = 128, kernel_size  = 3, stride = 1, dilation = 6),
            nn.Conv1d(in_channels = 128, out_channels = 128, kernel_size = 3, stride = 1, padding = 0, dilation = 8),
            #nn.BatchNorm1d(128),
            nn.ReLU(),
        )

        self.t_conv5 = nn.Sequential(
            # nn.ConvTranspose1d(in_channels = 128, out_channels = 256, kernel_size  = 3, stride = 1, dilation=2),
            nn.Conv1d(in_channels = 256, out_channels = 256, kernel_size = 3, stride = 1, padding = 0, dilation = 16),
            #nn.BatchNorm1d(256),
            nn.ReLU(),
        )

        self.t_conv6 = nn.Sequential(
            # nn.ConvTranspose1d(in_channels = 256, out_channels = 512, kernel_size  = 3, stride = 1, dilation = 1),
            nn.Conv1d(in_channels = 512, out_channels = 512, kernel_size = 3, stride = 1, padding = 0, dilation = 32),
            #nn.BatchNorm1d(512),
            nn.ReLU(),
        )

        self.t_conv7 = nn.Sequential(
            # nn.ConvTranspose1d(in_channels = 512, out_channels = 48, kernel_size  = 3, stride = 1, dilation= 1),
            nn.Conv1d(in_channels = 1024, out_channels = 48, kernel_size = 3, stride = 1, padding = 0, dilation = 64)
        )

        self.conv1to3 = nn.Sequential(
            nn.Conv1d(in_channels = 512, out_channels = 64, kernel_size = 3, stride = 1, padding = 0, dilation = 4),
            #nn.BatchNorm1d(512),
            nn.ReLU(),
        )# out : 128 * 7501
        
        self.conv2to4 = nn.Sequential(
            nn.Conv1d(in_channels = 256, out_channels = 32, kernel_size = 3, stride = 1, padding = 0, dilation = 8),
            #nn.BatchNorm1d(512),
            nn.ReLU(),
        )# out : 64 * 7501
        
        self.conv3to5 = nn.Sequential(
            nn.Conv1d(in_channels = 128, out_channels = 16, kernel_size = 3, stride = 1, padding = 0, dilation = 16),
            #nn.BatchNorm1d(512),
            nn.ReLU(),
        )# out : 32 * 7501
        
        self.conv4to6 = nn.Sequential(
            nn.Conv1d(in_channels = 64, out_channels = 8, kernel_size = 3, stride = 1, padding = 0, dilation = 32),
            #nn.BatchNorm1d(512),
            nn.ReLU(),
        )# out : 16 * 7501
        
        self.conv5to7 = nn.Sequential(
            nn.Conv1d(in_channels = 32, out_channels = 4, kernel_size = 3, stride = 1, padding = 0, dilation = 64),
            #nn.BatchNorm1d(512),
            nn.ReLU(),
        )# out : 8 * 7501
        
    def forward(self, mel_spec):
        x = F.pad(mel_spec, pad = (2, 0, 0, 0))
        x1 = self.conv1(x) # 48 * 7501 => 512 * 7501 
        print(x1.shape)
        
        x1to2 = F.pad(x1, pad = (4, 0, 0, 0)) #512 * 7501 
        x1to3 = F.pad(x1, pad = (8, 0, 0, 0)) #512 * 7501 
        #print('x1to2',x1to2.shape)
        #print('x1to3',x1to3.shape)

        x2 = self.conv2(x1to2) # 512 * 7501 => 256 * 7501
        print(x2.shape)
        
        x2to3 = F.pad(x2, pad = (8, 0, 0, 0)) # 256 * 7501
        #x2to4 = F.pad(x2, pad = (16, 0, 0, 0)) # 256 * 7501
        #print('x2to3 :',x2to3.shape)
        #print('x2to4 :',x2to3.shape)
        
        x3 = self.conv3(x2to3) # 256 * 7501 => 64 * 7501
        x3_connec = self.conv1to3(x1to3) # In : 512 * 7501 , out: 64 * 7501     128
        x3= torch.cat([x3,x3_connec],1) # 128 * 7501
        print(x3.shape)

        x3to4 = F.pad(x3, pad = (16, 0, 0, 0)) # 128 * 7501
        x3to5 = F.pad(x3, pad = (32, 0, 0, 0)) # 128 * 7501
        #print('x3to4 :',x3to4.shape)
        #print('x3to5 :',x3to5.shape)
        
        x4 = self.conv4(x3to4) # 128 * 7501 => 64 * 7501
        #x4_connec = self.conv2to4(x2to4) # In : 256 * 7501 , out : 32 * 7501 
        #x4 = torch.cat([x4,x4_connec],1) # 64 * 7501
        print(x4.shape)
        
        x4to5 = F.pad(x4, pad = (32, 0, 0, 0)) # 64 * 7501
        #x4to6 = F.pad(x4, pad = (64, 0, 0, 0)) # 64 * 7501
        #print('x4to5 :',x4to5.shape)
        #print('x4to6 :',x4to6.shape)

        x5 = self.conv5(x4to5) # 64 * 7501 = > 16 * 7501 
        x5_connec = self.conv3to5(x3to5) # 128 * 7501 => 16 * 7501
        x5 = torch.cat([x5,x5_connec],1) # 32 * 7501
        print(x5.shape)
        
        x5to6 = F.pad(x5, pad = (64, 0, 0, 0)) # 32 * 7501
        x5to7 = F.pad(x5, pad = (128, 0, 0, 0)) # 32 * 7501
        #print('x5to6 :',x5to6.shape)
        #print('x5to7 :',x5to7.shape)

        x6 = self.conv6(x5to6) # 32 * 7501 => 16 * 7501
        #x6_connec = self.conv4to6(x4to6) # 64 * 7501 => 8 * 7501
        #x6 = torch.cat([x6,x6_connec],1) # 16 * 7501
        print(x6.shape)
        x6to7 = F.pad(x6, pad = (128, 0, 0, 0)) # 32 * 7501
        #print('x6to7 :',x6to7.shape)
        
        x7 = self.conv7(x6to7) # 16 * 7501 => 4 *7501 
        pre_encode = torch.flatten(x7)
        x7_connec = self.conv5to7(x5to7) # 32 * 7501 => 4 * 7501
        x7 = torch.cat([x7,x7_connec],1) # 8 * 7501
        print(x7.shape)
        encode = self.encoder_fc(x7.view(-1, 8 * 7501))

        #encode = self.encoder_fc(x.view(-1, 8 * 1876))

        # print('decode')
        x = self.decoder_fc(encode)
        x = x.view(-1, 8, 7501)
        
        #x = torch.swapaxes(torch.fliplr(torch.swapaxes(x, 1, 2)), 1, 2)
        
        x = torch.cat([x7,x],1) # 4,16,7501
        #x = self.concat7_shape(x) 
        x = F.pad(x, pad = (2, 0, 0, 0)) 
        x = self.t_conv1(x) # 4,8,7501
        #print(x.shape,'new_cat7')

        x = torch.cat([x6,x],1)
        #x = self.concat6_shape(x)
        x = F.pad(x, pad = (4, 0, 0, 0))
        x = self.t_conv2(x)
        
        x = torch.cat([x5,x],1)
        #x = self.concat5_shape(x)
        x = F.pad(x, pad = (8, 0, 0, 0))
        x = self.t_conv3(x)

        x = torch.cat([x4,x],1)
        #x = self.concat4_shape(x)
        x = F.pad(x, pad = (16, 0, 0, 0))
        x = self.t_conv4(x)

        x = torch.cat([x3,x],1)
        #x = self.concat3_shape(x)
        x = F.pad(x, pad = (32, 0, 0, 0))
        x = self.t_conv5(x)
        
        x = torch.cat([x2,x],1)
        #x = self.concat2_shape(x)
        x = F.pad(x, pad = (64, 0, 0, 0))
        x = self.t_conv6(x)
        
        x = torch.cat([x1,x],1)
        #x = self.concat1_shape(x)
        x = F.pad(x, pad = (128, 0, 0, 0))
        x = self.t_conv7(x)
        
        #print(x.shape)
        #x = torch.swapaxes(torch.fliplr(torch.swapaxes(x, 1, 2)), 1, 2)
        return encode, x

# @st.cache(allow_output_mutation=True)
#### 노래 들려주기
def display_wavfile(wavpath):
    audio_bytes = open(wavpath, 'rb').read() # 말그래도 audio파일을 바이트로 받아들이는 것
    st.audio(audio_bytes, format=f'audio/.wav', start_time=0) # display

##### 노래 업로드 
def upload_and_save_wavfiles(directory):
    # 1. 디렉토리가 있는지 확인, 없으면 만든다
    if not os.path.exists(directory):
        os.makedirs(directory)

    uploaded_file = st.file_uploader("upload", type=['wav', 'mp3'], accept_multiple_files=False)
    if uploaded_file is not None:
        if uploaded_file.name.endswith('wav'):
            audio = AudioSegment.from_wav(uploaded_file)
            file_type = 'wav'
        elif uploaded_file.name.endswith('mp3'):
            audio = AudioSegment.from_mp3(uploaded_file)
            file_type = 'mp3'

        # 2. 이제는 디렉토리가 있으니, 파일을 저장
        audio.export(os.path.join(directory, uploaded_file.name), format=file_type)
        st.write("### Original Audio")
        st.write("{}".format(uploaded_file.name.split('.wav')[0]))
        display_wavfile(os.path.join(directory, uploaded_file.name)) 
        name = uploaded_file.name

    return directory, name

##### spleeter를 활용하여 목소리 추출하기
def spleete_audio(directory, name):
    separator = Separator('spleeter:2stems')
    separator.separate_to_file(directory + '/' + name, 'separate/')
    shutil.move("separate/"+name.split('.wav')[0]+"/vocals.wav", "separate/vocals.wav")
    # vocal_file = os.listdir('separate/'+name.split('.wav')[0]+'/')[0]
    # # os.renames('data_vocal/'+singer+'/'+song_name+'/'+vocal_file,'data_vocal/'+singer+'/'+song_name+'.wav')
    shutil.rmtree('separate/'+name.split('.wav')[0]+'/')
    st.write("#### Separate Audio")
    display_wavfile("separate/vocals.wav") 

def make_all(subset):
    data = pd.DataFrame()
    if subset=='all':
      lst = ['separate/vocals.wav']
    elif subset=='all_pitch':
      lst = ['separate/pitch_vocal.wav']
    data['vocal'] = lst
    data['label'] = ''
    data.to_csv('separate/{}.csv'.format(subset),index=False)
    return data

# extract 하는 함수들이니 놀라지 마시오!
def parse_options(subset, extract_name, npnp):
    parser = argparse.ArgumentParser()

    # load data
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=64, help='Number of workers to load data')

    # resume path
    parser.add_argument('--model_path', type=str,default='extract_model/ckpt_epoch_{}.pth'.format(npnp), help='Path to load the saved model')

    # dataset split
    parser.add_argument('--subset', type=str, default='{}'.format(subset), help='Dataset subset name for training')

    # specify folder
    parser.add_argument('--data_path', type=str, default='separate/', help='Path to load data')
    parser.add_argument('--save_path', type=str, default='separate/{}_{}'.format(extract_name, npnp),  help='Path to save results')

    # pitch shift by gender
    parser.add_argument('--gender', type=int, default=2)

    # misc
    parser.add_argument('--seed', type=int, default=42)

    return parser.parse_args()

def inference(dataloader, encoder, softmax, opts):
    encoder.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()

    features = torch.zeros(len(dataloader.dataset), opts.feat_dim)
    features = features.cuda()

    end = time.time()
    for batch_index, (indices, inputs, labels) in enumerate(dataloader):
        data_time.update(time.time() - end)

        batch_size = inputs.size(0)
        inputs = inputs.float().cuda()

        # ===================forward=====================
        with torch.no_grad():
            features[indices] = encoder(inputs)

        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        print('\033[F\033[KVal:', end='\t')
        print('[{0}/{1}]'.format(batch_index + 1, len(dataloader)), end='\t')
        print(f'BT {batch_time.val:.3f} ({batch_time.avg:.3f})', end='\t')
        print(f'DT {data_time.val:.3f} ({data_time.avg:.3f})', flush=True)

    return features.cpu()

def main(opts):
    if not torch.cuda.is_available():
        print('Only support GPU mode')
        sys.exit(1)

    # fix all parameters for reproducibility
    random.seed(opts.seed)
    np.random.seed(opts.seed)
    os.environ['PYTHONHASHSEED'] = str(opts.seed)
    torch.manual_seed(opts.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(opts.seed)
        torch.cuda.manual_seed_all(opts.seed)
    ########### encoder ##############
    checkpoint = torch.load(opts.model_path)
    opts.feat_dim = checkpoint['opts'].feat_dim
    encoder = CRNN2D_elu2(input_size=1 + checkpoint['opts'].n_fft // 2, feat_dim=checkpoint['opts'].feat_dim, dropout=0)
    state_dict = {key.partition('module.')[2]: checkpoint['model'][key] for key in checkpoint['model'].keys()}
    encoder.load_state_dict(state_dict, strict=True)
    for param in encoder.parameters():
        param.requires_grad = False
    # Multi-GPU
    encoder = encoder.cuda()
    encoder = torch.nn.DataParallel(encoder)
    softmax = torch.nn.Softmax(dim=1).cuda()
    ########### data ##############
    dataset = DefaultSet(opts.data_path, opts.subset, checkpoint['opts'].input_len, checkpoint['opts'].n_fft, opts.gender)
    dataloader = DataLoader(dataset, batch_size=opts.batch_size, shuffle=False,
                            num_workers=opts.num_workers, pin_memory=True, drop_last=False)
    ########### inference ##############
    features = inference(dataloader, encoder, softmax, opts)
    np.savez(opts.save_path, files=dataset.files, features=features.numpy())

def extract_voice(subset, extract_name, npnp):
    st.write("{}음색 특징이 뽑히는 중입니다..".format(npnp))
    make_all(subset)
    opts = parse_options(subset, extract_name, npnp)
    main(opts)
    st.write('''
        **음색 추출 완료!!!!!!**
        ''')

def npz_load(path, files):
  pp = np.load('{}/{}.npz'.format(path, files))
  song = list([pp[k] for k in pp][0])
  content = list([pp[k] for k in pp][1])

  ext = pd.DataFrame(columns=['song'],data=song)
  df = pd.DataFrame(columns=['extract_0'],data=content[0])

  for i in range(len(content)-1):
    i += 1
    col_name = 'extract_'+str(i)
    df2 = pd.DataFrmae(columns=[col_name],data=content[1])
    df = pd.concat([df,df2],axis=1)
  st.write(df)
  df.to_csv('{}/{}.csv'.format(path, files))

def load(index):
    files = tuple(pd.read_csv('separate/all.csv')['vocal'])
    audio, sample_rate = torchaudio.load(files[index])
    if sample_rate != 16000:
        transform = transforms.Resample(sample_rate, 16000)
        audio = transform(audio)
        sample_rate = 16000
    assert sample_rate == sample_rate
    return torch.unsqueeze(torch.mean(audio, axis=0), dim=0)  # make it mono

def reshape(audio, length):
    current = audio.shape[1]
    if current < length:
        audio = F.pad(audio, (0, length - current))
    elif current > length:
        idx = random.randint(0, current - length)
        audio = audio[:, idx: idx + length]
    return audio

def pitch_shift_by_male(audio):
    source = reshape(audio, 128000 + 50)
    pitch = 6
    sample_rate = 16000
    effects = [['pitch', str(pitch * 100)], ['rate', str(sample_rate)]]
    target, sample_rate = sox_effects.apply_effects_tensor(source, sample_rate, effects)
    
    assert sample_rate == sample_rate

    return reshape(target, 128000)

def pitch_shift_by_female(audio):
    source = reshape(audio, 128000 + 50)
    pitch = -6
    sample_rate = 16000
    effects = [['pitch', str(pitch * 100)], ['rate', str(sample_rate)]]
    target, sample_rate = sox_effects.apply_effects_tensor(source, sample_rate, effects)
    
    assert sample_rate == sample_rate

    return reshape(target, 128000)

def gender_pitch_shift(input_len=128000, gender=0):
    audio = reshape(load(0),input_len)
    if gender == 0:
        audio = pitch_shift_by_male(audio)
    elif gender == 1:
        audio = pitch_shift_by_female(audio)
    path = 'separate/pitch_vocal.wav'
    torchaudio.save(path, audio,16000)
    st.write('### Pitch Shift')
    st.write('키 조정한 목소리')
    display_wavfile(path)

def make_min2_mel(song_paths='separate/vocals.wav',sample_rate=16000): # song_paths = wav_path 입력
    
    wav, wav_sample_rate = torchaudio.load(song_paths)
    
    if wav_sample_rate != sample_rate:
        transform = torchaudio.transforms.Resample(wav_sample_rate, sample_rate)
        wav = transform(wav)
            
    wav = torch.unsqueeze(torch.mean(wav, axis=0), dim=0) # mono
    wav_to_mel = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=512, hop_length=256, n_mels=48) # to mel

    mel = wav_to_mel(wav)
    mel = mel.cpu().detach().numpy() # 멜 임베딩을 numpy로
    
    return mel

def get_mel_embeding(model, train_loader):
    model.eval()
    mel_embeding_li = []
    with torch.no_grad():
        for batch in train_loader: 
            
            mel = torch.FloatTensor(batch).to(DEVICE)
            
            encode, output = model(mel)
            mel_embeding_li.append(encode.detach().cpu().numpy())

    return mel_embeding_li

def mel_to_embedding(idx):
    mel = make_min2_mel(song_paths = 'separate/vocals.wav')
    mel = [mel.squeeze()[:,:7501]]
    inference_batch_li = DataLoader(mel, batch_size=1, shuffle=False,drop_last=False)

    inference_model = TimeAutoEncoder().to(DEVICE)
    inference_model.load_state_dict(torch.load('TJAE/TimeAutoEncoder_skipconnection_charlie_val.pt', map_location = DEVICE))

    inference_embedding = get_mel_embeding(inference_model,inference_batch_li)
    inference_df = pd.DataFrame(inference_embedding[0])
    inference_df.index = [idx]
    inference_df.to_csv('TJAE/tjae_vocal.csv')
    st.write('**TJAE를 활용하여 추출한 embedding 예시**')
    st.write(inference_df)

# 코사인
def cos_sim(A, B):
    return dot(A, B)/(norm(A)*norm(B))
# 피어슨
def pear_sim(a, b):
    return np.dot((a - np.mean(a)), (b - np.mean(b))) / ((np.linalg.norm(a - np.mean(a))) * (np.linalg.norm(b - np.mean(b))))

def get_similarity():
    # tcae concat
    tcae = pd.read_csv('TJAE/tcae_charlie_embedding.csv')
    tcae_vocal = pd.read_csv('TJAE/tjae_vocal.csv')
    tcae.index = pd.read_csv('TJAE/all.csv')['vocal']
    tcae_vocal = tcae_vocal.drop('Unnamed: 0',axis=1)
    tcae_vocal.index = ['보컬보컬']
    tcae_all = pd.concat([tcae,tcae_vocal])

    # nn np concat
    df = pd.read_csv('separate/extract_neg_neg.csv',index_col=0)
    df2 = pd.read_csv('separate/extract_neg_pos.csv',index_col=0)
    df3 = pd.read_csv('separate/extract_pitch_neg_neg.csv',index_col=0)
    df4 = pd.read_csv('separate/extract_pitch_neg_pos.csv',index_col=0)
    df.columns = ['보컬보컬']
    df2.columns = ['보컬보컬']
    df3.columns = ['보컬보컬']
    df4.columns = ['보컬보컬']
  
    nn_a = pd.read_csv('extract_model/nn.csv')
    np_a = pd.read_csv('extract_model/np.csv')

    nn_all = pd.concat([nn_a,df],axis=1).T
    np_all = pd.concat([np_a,df2],axis=1).T
    nn_pitch_all = pd.concat([nn_a,df3],axis=1).T
    np_pitch_all = pd.concat([np_a,df4],axis=1).T
    reducer = umap.UMAP(n_components=30,random_state=42)
    tcae_embedding = reducer.fit_transform(tcae_all)
    st.write('**정확한 추천을 위해 데이터 정리 중입니다(차원축소)**')
    nn_embedding = reducer.fit_transform(nn_all)
    np_embedding = reducer.fit_transform(np_all)
    st.write('**추천을 하기 위해 열심히 일하는 중입니다..!**')
    nn_pit_embedding = reducer.fit_transform(nn_pitch_all)
    np_pit_embedding = reducer.fit_transform(np_pitch_all)
    tcae_embedding=pd.DataFrame(tcae_embedding,index=tcae_all.index)
    nn_embedding=pd.DataFrame(nn_embedding,index=nn_all.index)
    np_embedding=pd.DataFrame(np_embedding,index=np_all.index)
    nn_pit_embedding=pd.DataFrame(nn_pit_embedding,index=nn_pitch_all.index)
    np_pit_embedding=pd.DataFrame(np_pit_embedding,index=np_pitch_all.index)

    tcae_test_audio = np.squeeze(tcae_embedding.loc[tcae_embedding.index.str.contains('보컬보컬'),:])
    nn_test_audio = np.squeeze(nn_embedding.loc[nn_embedding.index.str.contains('보컬보컬'),:])
    np_test_audio = np.squeeze(np_embedding.loc[np_embedding.index.str.contains('보컬보컬'),:])
    nn_pit_test_audio = np.squeeze(nn_pit_embedding.loc[nn_pit_embedding.index.str.contains('보컬보컬'),:])
    np_pit_test_audio = np.squeeze(np_pit_embedding.loc[np_pit_embedding.index.str.contains('보컬보컬'),:])
    
    def cos_sim_test(sim , embedding, test_audio):
      np_pear_lst = []
      name_lst = []
      for i in embedding.index:
        np_pear_lst.append(sim(test_audio, np.squeeze(embedding.loc[i,:])))
        name_lst.append(i)
          
      nn_pear_df = pd.DataFrame(np_pear_lst, name_lst)
      nn_pear_df.columns = ['cos']
      return nn_pear_df

    cos_tcae = cos_sim_test(cos_sim, tcae_embedding,tcae_test_audio)
    cos_nn = cos_sim_test(cos_sim, nn_embedding,nn_test_audio)
    cos_np = cos_sim_test(cos_sim, np_embedding,np_test_audio)
    cos_nn_pit = cos_sim_test(cos_sim ,nn_pit_embedding,nn_pit_test_audio)
    cos_np_pit = cos_sim_test(cos_sim, np_pit_embedding,np_pit_test_audio)

    pear_tcae = cos_sim_test(pear_sim, tcae_embedding,tcae_test_audio)
    pear_nn = cos_sim_test(pear_sim, nn_embedding,nn_test_audio)
    pear_np = cos_sim_test(pear_sim, np_embedding,np_test_audio)
    pear_nn_pit = cos_sim_test(pear_sim ,nn_pit_embedding,nn_pit_test_audio)
    pear_np_pit = cos_sim_test(pear_sim, np_pit_embedding,np_pit_test_audio)

    cos_tcae['cos']=cos_tcae['cos']*0.1+cos_nn['cos']*0.07+cos_np['cos']*0.08+cos_nn_pit['cos']*0.12+cos_np_pit['cos']*0.13+\
    pear_tcae['cos']*0.1+pear_nn['cos']*0.07+pear_np['cos']*0.08+pear_nn_pit['cos']*0.12+pear_np_pit['cos']*0.13

    st.write(cos_tcae.sort_values(by='cos', ascending = False).head(10))


    # tcae_vocal.columns = extract_1.columns
    # df = pd.concat([extract_1,extract_2,extract_3,extract_4,tcae_vocal])
    # df = pd.DataFrame(df.mean(axis=0)).T
    # df.index = ['보컬']
    # tcae.columns = df.columns
    # tcae_vocal = pd.concat([tcae,df])
    # reducer = umap.UMAP(n_components=10, random_state=100)
    

# get_similarity('nn', 'extract_neg_neg.csv')

def run_program():
    st.write("👇👇👇👇👇")
    st.write('''
        **노래 업로드를 하면 오류가 뜨지 않아요ㅠㅠ**
        ''')
    directory, name = upload_and_save_wavfiles('original')
    st.write('''
        -------
        ''')
    st.write('''
        조금만 기다려주세요....\n
        \n
        ''')
    spleete_audio(directory, name)
    extract_voice('all', 'extract', 'neg_neg')
    npz_load('separate','extract_neg_neg')
 
    extract_voice('all', 'extract', 'neg_pos')
    npz_load('separate','extract_neg_pos')
    st.write('''
        -------
        ''')
    gen = st.radio(
    "남자인가요, 여자인가요?",
    ('남자', '여자'))
    if gen == '남자':
        gender_pitch_shift(gender=0)
    elif gen == '여자':
        gender_pitch_shift(gender=1)

    extract_voice('all_pitch', 'extract_pitch','neg_neg')
    npz_load('separate','extract_pitch_neg_neg')

    extract_voice('all_pitch', 'extract_pitch','neg_pos')
    npz_load('separate','extract_pitch_neg_pos')
    st.write('''
        -------
        ''')
    st.write('### TJAE')
    st.write('보컬에서 뽑을 수 있는 다른 정보까지 추출중입니다..')
    mel_to_embedding('{}/{}'.format(directory, name))
    st.write('**추출 완료!!!!**')
    st.write('''
        -------
        ''')
    get_similarity()
    # get_similarity('nn', 'extract_neg_neg.csv')
    # get_similarity('np', 'extract_neg_pos.csv')
    # get_similarity('nn', 'extract_pitch_neg_neg.csv')
    # get_similarity('np', 'extract_pitch_neg_pos.csv')
    

def home_page():
    st.write('''
    #### Login by User Name\n
    \n
    <==== Enter your name\n
    <==== Next Click "Confirm"
    ''')
    username = st.sidebar.text_input("User Name")

    if st.sidebar.checkbox("Confirm"):
        st.success("Logged In as {}".format(username))

        task = st.selectbox("Task", ["Documentation", "Run Program"])
        if task == "Documentation":
            documentation()

        elif task == "Run Program":
            run_program()
            # if st.sidebar.checkbox("program_start"):
            #     run_program()
        

if __name__ == "__main__":
    st.set_page_config(layout="wide")
    showWarningOnDirectExecution = False
    
    st.markdown("""
        <style>
        .css-1aumxhk {
            padding: 0em 1em;
        }
        </style>
    """, unsafe_allow_html=True)
    st.sidebar.markdown(f"<h5 style='text-align: right;'>컨퍼런스_이노래는어떻조</h5>", unsafe_allow_html=True)
    st.title("Voice - Music Recommendation System")
    st.write('''
    ### 당신의 음색에 어울리는 노래를 원하신다면 지금 실행해보세요!
    ''')
    st.sidebar.title("Sidebar")
    menu = ["Home", "Feedback", "Opinions"]
    choice = st.sidebar.selectbox("Menu", menu)
    if choice == "Home":
        home_page()
    # elif choice == "Feedback":
    #     feedback_after_program()
    # else:
    #     opinion_page()