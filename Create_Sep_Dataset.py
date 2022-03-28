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
dir_tv = '..//..//data//SpeakerSep//'
speakers = os.listdir(dir_base)

random.seed(random_seed)
random.shuffle(speakers)

speaker_dic = dict()
speaker_dic['train'] = []
speaker_dic['valid'] = []
speaker_train_num = int(len(speakers) * train_rate)
speaker_idx_dic = dict()

for speaker_index in range(len(speakers)):
    if speaker_index < speaker_train_num:
        speaker_dic['train'].append(speakers[index])
        speaker_i_dic[speakers[index]] = speaker_index
    else:
        speaker_dic['valid'].append(speakers[index])

ult = dict()
melspec = dict()
ultmel_size = dict()

# generate training and validation dataset
for train_valid in ['train','valid']:
    
    speaker_all = []
    ult_file_all = []
    
    #get pathes of ultrasound files and corresponding speakers' names
    for speaker in speaker_dic[train_valid]:
        speaker_path = dir_base + speaker + '//'
        all_file_names = os.listdir(speaker_path)
        for file_name in all_file_names:
            if '.ult' in file_name:
                ult_file_all.append(speaker_path + file_name)
                file_name.split('.')[0]
                wav_name = file_name.split('.')[0] + '.wav'
                wav_file_all.append(speaker_path + wav_name)
                speaker_all.append(speaker)
    
    n_ult_frames = int(len(ult_file_all) * ave_frame / frame_selected_from)
    ult[train_valid] = np.empty((n_ult_frames, n_lines, n_pixels_reduced))
    melspec[train_valid] = np.empty((n_ult_frames, n_melspec + 1))
    if train_valid == 'valid':
        melspec[train_valid][:, -1] = -1
    ultmel_size[train_valid] = 0
    
    # read ultrasound frames and mel_spectrograms
    for file_index in tqdm(range(len(ult_file_all))):
        
        ult_file = ult_file_all[file_index]
        wav_file = wav_file_all[file_index]
        speaker = speaker_all[file_index]
        
        ult_data = read_ult(ult_file)
        mel_data = wf.get_mel(wav_file, stft)
        mel_data = np.fliplr(np.rot90(mel_data.data.numpy(), axes = (1,0)))
        
        ultmel_len = np.min((len(ult_data), len(mel_data)))
        ult_data = ult_data[0:ultmel_len]
        mel_data = mel_data[0:ultmel_len]
        
        # print(wav_file.shape, ult_data.shape, mel_data.shape)

        ult_len = 0

        for i in range(int(ultmel_len / frame_selected_from)):
            frame_idx = i * frame_selected_from
            ult[train_valid][ultmel_size[train_valid] + i] = \
            sk_trans.resize(ult_data[frame_idx], (n_lines, n_pixels_reduced), 
                            preserve_range = True)
            melspec[train_valid][ultmel_size[train_valid] + i, : n_melspec] = \ 
            mel_data[frame_idx]
            if train_valid == 'train': 
                melspec[train_valid][ultmel_size[train_valid] + i, -1] = \ 
                speaker_idx_dic[speaker]
            ult_len += 1

        ultmel_size[train_valid] += ult_len

        print('frames for', train_valid, ':', ultmel_size[train_valid])

    ult[train_valid] = ult[train_valid][0: ultmel_size[train_valid]]
    melspec[train_valid] = melspec[train_valid][0: ultmel_size[train_valid]]
    
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
