
import librosa
import numpy as np
import scipy
import os
import h5py
from glob import iglob
from shutil import copy2
from os.path import join


epsilon = np.finfo(float).eps


def np_REG_batch(data1, data2, batch_size, data_len):
    n_start = 0
    n_end = batch_size
    l = data_len
    while True:
        if n_end >= l:
            yield data1[n_start:]
            yield data2[n_start:]
            n_start = 0
            n_end = batch_size
        else:
            yield data1[n_start:n_end]
            yield data2[n_start:n_end]

        n_start = n_end
        n_end += batch_size


def search_wav(data_path):
    file_list = []
    for filename in iglob('{}/**/*.WAV'.format(data_path), recursive=True):
        file_list.append(str(filename))
    for filename in iglob('{}/**/*.wav'.format(data_path), recursive=True):
        file_list.append(str(filename))
    return file_list

def split_list(alist, wanted_parts=20):
    length = len(alist)
    return [alist[i * length // wanted_parts: (i + 1) * length // wanted_parts]
            for i in range(wanted_parts)]

def wav2spec(wavfile, sr, forward_backward=None, SEQUENCE=None, norm=True, hop_length=256):
    # Note:This function return three different kind of spec for training and
    # testing
    y, sr = librosa.load(wavfile, sr, mono=True)
    NUM_FRAME = 2  # number of backward frame and forward frame
    NUM_FFT = 512

    D = librosa.stft(y,
                     n_fft=NUM_FFT,
                     hop_length=hop_length,
                     win_length=512,
                     window=scipy.signal.hann)
    D = D + epsilon
    Sxx = np.log10(abs(D)**2)
    if norm:
        Sxx_mean = np.mean(Sxx, axis=1).reshape(257, 1)
        Sxx_var = np.var(Sxx, axis=1).reshape(257, 1)
        Sxx_r = (Sxx - Sxx_mean) / Sxx_var
    else:
        Sxx_r = np.array(Sxx)
    idx = 0
    # set data into 3 dim and muti-frame(frame, sample, num_frame)
    if forward_backward:
        Sxx_r = Sxx_r.T
        return_data = np.empty(
            (100000, np.int32(NUM_FRAME * 2) + 1, np.int32(NUM_FFT / 2) + 1))
        frames, dim = Sxx_r.shape

        for num, data in enumerate(Sxx_r):
            idx_start = idx - NUM_FRAME
            idx_end = idx + NUM_FRAME
            if idx_start < 0:
                null = np.zeros((-idx_start, dim))
                tmp = np.concatenate((null, Sxx_r[0:idx_end + 1]), axis=0)
            elif idx_end > frames - 1:
                null = np.zeros((idx_end - frames + 1, dim))
                tmp = np.concatenate((Sxx_r[idx_start:], null), axis=0)
            else:
                tmp = Sxx_r[idx_start:idx_end + 1]

            return_data[idx] = tmp
            idx += 1
        shape = return_data.shape
        if SEQUENCE:
            return return_data[:idx]
        else:
            return return_data.reshape(shape[0], shape[1] * shape[2])[:idx]

    else:
        Sxx_r = np.array(Sxx_r).T
        shape = Sxx_r.shape
        if SEQUENCE:
            return Sxx_r.reshape(shape[0], 1, shape[1])
        else:
            return Sxx_r

def spec2wav(wavfile, sr, output_filename, spec_test, hop_length=None):

    y, sr = librosa.load(wavfile, sr, mono=True)
    D = librosa.stft(y,
                     n_fft=512,
                     hop_length=hop_length,
                     win_length=512,
                     window=scipy.signal.hann)
    D = D + epsilon
    phase = np.exp(1j * np.angle(D))
    Sxx_r_tmp = np.array(spec_test)
    Sxx_r_tmp = np.sqrt(10**Sxx_r_tmp)
    Sxx_r = Sxx_r_tmp.T
    reverse = np.multiply(Sxx_r, phase)

    result = librosa.istft(reverse,
                           hop_length=hop_length,
                           win_length=512,
                           window=scipy.signal.hann)

    y_out = librosa.util.fix_length(result, len(y), mode='edge')
    y_out = y_out/np.max(np.abs(y_out))
    maxv = np.iinfo(np.int16).max
    librosa.output.write_wav(
        output_filename, (y_out * maxv).astype(np.int16), sr)


def copy_file(input_file, output_file):
    copy2(input_file, output_file)


def _gen_noisy(clean_file_list, noise_file_list, save_dir, snr, sr_clean, sr_noise, num=None):
    sr_clean = 16000
    sr_noise = 20000
    SNR = float(snr.split('dB')[0])
    clean_file = clean_file_list[num]
    noise_file = noise_file_list[num]
    clean_name = clean_file.split('/')[-1].split('.')[0]
    noise_name = noise_file.split('/')[-1].split('.')[0]
    y_clean, sr_clean = librosa.load(clean_file, sr_clean, mono=True)
    #### scipy cannot conver TIMIT format ####

    clean_pwr = sum(abs(y_clean)**2) / len(y_clean)
    y_noise, sr_noise = librosa.load(noise_file, sr_noise, mono=True)
    
    tmp_list = []
    if len(y_noise) < len(y_clean):
        tmp = (len(y_clean) // len(y_noise)) + 1
        y_noise = np.array([x for j in [y_noise] * tmp for x in j])
        y_noise = y_noise[:len(y_clean)]
    else:
        y_noise = y_noise[:len(y_clean)]
    y_noise = y_noise - np.mean(y_noise)
    noise_variance = clean_pwr / (10**(SNR / 10))
    noise = np.sqrt(noise_variance) * y_noise / np.std(y_noise)
    y_noisy = y_clean + noise
    maxv = np.iinfo(np.int16).max
    save_name = '{}_{}_{}.wav'.format(snr, noise_name, clean_name)
    librosa.output.write_wav(
        '/'.join([save_dir, save_name]), (y_noisy * maxv).astype(np.int16), sr_clean)

def _gen_clean(clean_file_list, save_dir, snr, num=None):
    sr_clean = 16000
    noise_name = 'n0'
    clean_file = clean_file_list[num]
    y_clean, sr_clean = librosa.load(clean_file, sr_clean, mono=True)


    clean_name = clean_file.split('/')[-1].split('.')[0]
    maxv = np.iinfo(np.int16).max
    save_name = '{}_{}_{}.wav'.format(snr, noise_name, clean_name)
    librosa.output.write_wav(
        '/'.join([save_dir, save_name]), (y_clean * maxv).astype(np.int16), sr_clean)

def _create_split_h5(clean_split_list,
                     noisy_split_list,
                     save_dir,
                     file_name,
                     input_sequence=True,
                     split_num=None):
    irm_tmp = []
    noisy_tmp = []
    clean_tmp = []
    count = 0
    for clean_file, noisy_file in zip(clean_split_list[split_num], noisy_split_list[split_num]):
        data_name = noisy_file.split('/')[-1]
        # you can set noisy data is sequence or not
        noisy_spec = wav2spec(
            noisy_file, sr=16000, forward_backward=True, SEQUENCE=False, norm=True, hop_length=256)
        clean_spec = wav2spec(
            clean_file, sr=16000, forward_backward=False, SEQUENCE=False, norm=False, hop_length=256)

        noisy_tmp.append(noisy_spec)
        clean_tmp.append(clean_spec)
        if count % int(len(clean_split_list[split_num]) / 10) == 0:
            tmp = int(len(clean_split_list[split_num]) / 10)
            print('Part {} {}%'.format(split_num, 10 * count / tmp))
        count += 1
        if clean_spec.shape[0] == noisy_spec.shape[0]:
            continue
        else:
            print('Mismatch', noisy_file, clean_file)
            print('Clean shape:', clean_spec.shape)
            print('Noisy shape: ', noisy_spec.shape)

    noisy_data = np.vstack(noisy_tmp)
    y_clean_data = np.vstack(clean_tmp)
    print('Clean shape:', y_clean_data.shape)
    print('Noisy shape: ', noisy_data.shape)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with h5py.File(join(save_dir, '{}_{}.h5'.format(file_name, split_num)), 'w') as hf:
        hf.create_dataset('noisy_data', data=noisy_data)
        hf.create_dataset('clean_data', data=y_clean_data)

    del noisy_data
    del y_clean_data