import numpy as np
import os
import random
import skimage.transform as sk_trans
import WaveGlow_functions as wf
import torch
from tqdm import tqdm

# WaveGlow/STFT parameters
samplingFrequency = 22050
n_melspec = 80
hop_length_UTI = 181 # sampling rate of ultrasound files

# parameters of ultrasound images
framesPerSec = 22050 / 181
n_lines = 63
n_pixels = 412
n_pixels_reduced = 103 

frame_selected_from = 10
ave_frame = 1000
random_seed = 32
train_rate = 2 / 3

# read *.ult files
def read_ult(filename, NumVectors = n_lines, PixPerVector = n_pixels):
    ult_data = np.fromfile(filename, dtype = 'uint8')
    ult_data = np.reshape(ult_data, (-1, NumVectors, PixPerVector))
    return ult_data
    
stft = wf.TacotronSTFT(
    filter_length=1024, hop_length = hop_length_UTI, win_length = 1024, 
    n_mel_channels = n_melspec, sampling_rate = samplingFrequency, 
    mel_fmin = 0, mel_fmax = 8000)

# prepare speaker list
dir_base = '..//..//data//UltraSuite//core-uxtd//core//'
dir_tv = '..//..//data//SpeakerMix//'
speakers = os.listdir(dir_base)

speaker_idx_dic = dict()
for speaker_idx, speaker in enumerate(speakers):
    speaker_idx_dic[speaker] = speaker_idx

ult_file_all = []
wav_file_all = []
speaker_all = []

for speaker in speakers:
    speaker_path = dir_base + speaker + '//'
    all_file_names = os.listdir(speaker_path)
    for file_name in all_file_names:
        if '.ult' in file_name:
            ult_file_all.append(speaker_path + file_name)
        wav_name = file_name.split('.')[0] + '.wav'
            wav_file_all.append(speaker_path + wav_name)
            speaker_all.append(speaker_idx_dic[speaker])

# shuffle ult & wav & speaker
random.seed(random_seed)
random.shuffle(ult_file_all)
random.seed(random_seed)
random.shuffle(wav_file_all)
random.seed(random_seed)
random.shuffle(speaker_all)

ult_files = dict()
wav_files = dict()
speaker_files = dict()

# split train and valid dataset
ult_files['train'] = ult_file_all[: int(train_rate * len(ult_file_all))]
ult_files['valid'] = ult_file_all[int(train_rate * len(ult_file_all)): ]
wav_files['train'] = wav_file_all[: int(train_rate * len(wav_file_all))]
wav_files['valid'] = wav_file_all[int(train_rate * len(wav_file_all)): ]
speaker_files['train'] = speaker_file_all[: int(train_rate * len(speaker_file_all))]
speaker_files['valid'] = speaker_file_all[int(train_rate * len(speaker_file_all)): ]

ult = dict()
melspec = dict()
ultmel_size = dict()

# generate training and validation dataset
for train_valid in ['train','valid']:
    
    num_ult_frame = len(ult_files[train_valid])
    n_ult_frames = int(num_ult_frame * ave_frame / frame_selected_from)
    ult[train_valid] = np.empty((n_ult_frames, n_lines, n_pixels_reduced))
    melspec[train_valid] = np.empty((n_ult_frames, n_melspec + 1))

    ultmel_size[train_valid] = 0
    # load all training/validation data
    for file_index in tqdm(range(len(ult_files[train_valid]))):
        
        ult_file = ult_files[train_valid][file_index]
        wav_file = wav_files[train_valid][file_index]
        
        ult_data = read_ult(ult_file)
        mel_data = wf.get_mel(wav_file, stft)
        mel_data = np.fliplr(np.rot90(mel_data.data.numpy(), axes = (1,0)))

        ultmel_len = np.min((len(ult_data), len(mel_data)))
        ult_data = ult_data[0:ultmel_len]
        mel_data = mel_data[0:ultmel_len]

        # print(wav_file, ult_data.shape, mel_data.shape)
            
        ult_len = 0
            
        for i in range(int(ultmel_len / frame_selected_from)):
            frame_idx = i * frame_selected_from
            ult[train_valid][ultmel_size[train_valid] + i] = \
            sk_trans.resize(ult_data[frame_idx], (n_lines, n_pixels_reduced), 
                            preserve_range = True)
            melspec[train_valid][ultmel_size[train_valid] + i, : n_melspec] = \
            mel_data[frame_idx]
            ult_len += 1            
            if train_valid == 'train': 
                melspec[train_valid][ultmel_size[train_valid] + i, -1] = \ 
                speaker_idx_dic[speaker]

        ultmel_size[train_valid] += ult_len
        
        print('frames for', train_valid, ':', ultmel_size[train_valid])


    ult[train_valid] = ult[train_valid][0 : ultmel_size[train_valid]]
    melspec[train_valid] = melspec[train_valid][0 : ultmel_size[train_valid]]
    
    # scale input to [-1,1]
    ult[train_valid] /= 255
    ult[train_valid] -= 0.5
    ult[train_valid] *= 2
    
    # reshape ult for CNN
    ult[train_valid] = np.reshape(ult[train_valid], (-1, n_lines, n_pixels_reduced))

    np.save(dir_tv + train_valid + '_ult.npy', ult[train_valid])
    np.save(dir_tv + train_valid + '_melspec.npy', melspec[train_valid])
    
    print('numpy saved')
    
    torch_data_ult = torch.from_numpy(ult[train_valid])
    torch_data_mel = torch.from_numpy(melspec[train_valid])
    
    torch.save(torch_data_ult, dir_tv + train_valid + '_ult.pt')
    torch.save(torch_data_mel, dir_tv + train_valid + '_melspec.pt')
    
    print('torch saved')
