
import numpy as np
import os
import random
import pdb
import shutil

from utils import _gen_noisy, _gen_clean, _create_split_h5, search_wav, split_list
from multiprocessing import Pool
from functools import partial
from sklearn.utils import shuffle
from os.path import join


class Synth:

    def __init__(self, clean_list, noise_list, sr_clean, sr_noise):
        self.clean_list = clean_list
        self.noise_list = noise_list
        self.sr_clean = sr_clean
        self.sr_noise = sr_noise

    def gen_noisy(self, snr_list, noisy_dir, data_num=None, ADD_CLEAN=None, cpu_cores=4):
        # This function takes snr to generate noisy data.
        # ADD_CLEAN: add clean-to-clean as training data.
        # data_num: randomly choose clean data number in clean dir.
        # Set sampling rate of clean and noise data.


        if not os.path.exists(noisy_dir):
            os.makedirs(noisy_dir)

        clean_data_raw_list = self.clean_list
        noise_data_list_tmp = self.noise_list
        if data_num:
            tmp = []
            for i in range(data_num):
                tmp.append(random.choice(clean_data_raw_list))
            clean_data_list = tmp
        else:
            clean_data_list = clean_data_raw_list
        clean_length = len(clean_data_list)
        noise_length = len(noise_data_list_tmp)
        noise_train_num = np.int8(
            np.around(np.random.uniform(1, noise_length-1, clean_length)))
        noise_data_list = [noise_data_list_tmp[tmp] for tmp in noise_train_num]
        print('Clean data:', clean_length)
        print('Noise data:', noise_length)
        print('From {} noise file generate {} noisy file'.format(noise_length, clean_length))

        num_list = range(clean_length)
        # _gen_noisy(clean_data_list, noise_data_list, noisy_train_dir, snr, TIMIT_dir_name, num)
        for snr in snr_list:
            pool = Pool(cpu_cores)
            func = partial(_gen_noisy, clean_data_list,
                           noise_data_list, noisy_dir, snr, self.sr_clean, self.sr_noise)
            pool.map(func, num_list)
            pool.close()
            pool.join()

        
        if ADD_CLEAN:
            ## This function will set 'n0' as clean data
            num_list = range(len(clean_data_list))
            pool = Pool(cpu_cores)
            func = partial(_gen_clean, clean_data_list,
                           noisy_dir, '0dB')
            pool.map(func, num_list)
            pool.close()
            pool.join()

class GenMatrix:
    # This function generate noisy data to h5 file for training NN model.

    def __init__(self, save_h5_dir, save_h5_name, noisy_dir):
        self.save_h5_dir = save_h5_dir
        self.save_h5_name = save_h5_name
        self.noisy_dir = noisy_dir

        if not os.path.exists(self.save_h5_dir):
            os.makedirs(self.save_h5_dir)

    def create_h5(self, split_num, iter_num, input_sequence, DEL_TRAIN_WAV):

        cpu_cores = int(split_num / iter_num)
        tmp1 = []
        tmp2 = []
        tmp3 = []
        # noisy_dir = join(noisy_dir, 'train')
        training_data_list = search_wav(self.noisy_dir)
        print('Total training files: ', len(training_data_list))

        
        for file in training_data_list:
            try:
                snr, noise_name, clean_name1, clean_neme2 = file.split('/')[-1].split('_')
                clean_file = join(self.noisy_dir, '_'.join(['0dB', 'n0', clean_name1, clean_neme2]))
                noisy_file = file
            except:
                snr, noise_name, clean_name = file.split('/')[-1].split('_')
                clean_file = join(self.noisy_dir, '_'.join(['0dB', 'n0', clean_name]))
                noisy_file = file



            tmp1.append(clean_file)
            tmp2.append(noisy_file)
        
        training_num = 30000
        t1, t2 = shuffle(np.array(tmp1), np.array(tmp2))
        t1 = t1[:training_num]
        t2 = t2[:training_num]

        clean_split_list = split_list(t1, wanted_parts=split_num)
        noisy_split_list = split_list(t2, wanted_parts=split_num)


        start = 0
        end = cpu_cores
        for num in range(iter_num):
            print(start, end)
            pool = Pool(cpu_cores)
            func = partial(_create_split_h5, clean_split_list, noisy_split_list,
                           self.save_h5_dir, self.save_h5_name, input_sequence)
            pool.map(func, range(start, end))
            pool.close()
            pool.join()
            start = end
            end += cpu_cores
        if DEL_TRAIN_WAV:
            shutil.rmtree(self.noisy_dir)
