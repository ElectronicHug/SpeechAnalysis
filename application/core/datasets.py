from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import librosa
from torch import LongTensor, FloatTensor, HalfTensor, Tensor

class BasicAudioDataset(Dataset):
    def __init__(self, phrase_df=None, target=None, idx=None, audio_df=None):
        
        self.phrase_df = phrase_df
        
        if target is not None:
            self.target = np.array(target)
        else:
            self.target = None
        #self.index = np.array(idx)
        self.sample_rate = 8000
        self.audio_df = audio_df
        self.PERIOD = 5
        
    def __len__(self):
        return phrase_df.shape[0]
    
    def prepare_audio_sample_to_n_sec_long(self, input_audio, sampling_rate, use_random=True):
            
        len_input_audio = len(input_audio)
        effective_length_audio = sampling_rate * self.PERIOD

        if len_input_audio < effective_length_audio:
            zeros_audio = np.zeros(effective_length_audio, dtype=input_audio.dtype)
            if use_random:
                start_pasted_audio = np.random.randint(effective_length_audio - len_input_audio)
            else:
                start_pasted_audio = int((effective_length_audio - len_input_audio)/2)
            zeros_audio[start_pasted_audio:start_pasted_audio + len_input_audio] = input_audio
            output_audio = zeros_audio.astype(np.float32)
        elif len_input_audio > effective_length_audio:
            
            if use_random:
                start_cuted_audio = np.random.randint(len_input_audio - effective_length_audio)
            else:
                start_cuted_audio = int((len_input_audio - effective_length_audio)/2)
            output_audio = input_audio[start_cuted_audio:start_cuted_audio + effective_length_audio].astype(np.float32)
        else:
            output_audio = input_audio.astype(np.float32)
            
        return output_audio

    def from_audio_decibel_abs_fourie_audio(self, input_audio, sampling_rate, use_random=True):

        period_len_audio = self.prepare_audio_sample_to_n_sec_long(input_audio, sampling_rate, use_random)

        abs_fourie_audio = np.abs(librosa.stft(period_len_audio));

        decibel_abs_fourie_audio = librosa.power_to_db(abs_fourie_audio**2, ref=np.max)
        
        return decibel_abs_fourie_audio
        
    def __getitem__(self, idx):
        
        phrase_row = self.phrase_df.iloc[idx]
        audio_id = phrase_row['audio_id']
        input_audio = self.audio_df['audio_wave'].iloc[audio_id]
        phrase_sampling_rate = self.audio_df['sampling_rate'].iloc[audio_id]
    
        phrase_input_audio = input_audio[int(phrase_sampling_rate*phrase_row['phrase_start']):int(phrase_sampling_rate*phrase_row['phrase_stop'])]
        
        decibel_abs_fourie_audio = self.from_audio_decibel_abs_fourie_audio(phrase_input_audio, phrase_sampling_rate)
        
        sample = {'features':  FloatTensor(np.expand_dims(decibel_abs_fourie_audio, 0))}
        if self.target is not None:
            sample['targets'] = int(self.target[idx])
            
        return sample
    
    def slice_prepare_test_audio(self, input_audio, sampling_rate):
        
        if sampling_rate != self.sample_rate:
            input_audio = librosa.resample(input_audio, sampling_rate, self.sample_rate)
            sampling_rate = self.sample_rate
            
        prepared_audio_list = []
        number_of_slices = max(int(len(input_audio)/(self.sample_rate * self.PERIOD)+0.7), 1)
        for slice_ind in range(number_of_slices):
            slice_input_audio = input_audio[int(slice_ind*self.sample_rate * self.PERIOD):int((slice_ind+1)*self.sample_rate * self.PERIOD)]
            preprocessed_audio_slice = self.from_audio_decibel_abs_fourie_audio(slice_input_audio, self.sample_rate, use_random=False)
            prepared_audio_list.append(preprocessed_audio_slice)
            
        return prepared_audio_list