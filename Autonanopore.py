import numpy as np
import pyabf
import os
from scipy.stats import gumbel_r
from scipy.stats import kstest


def Autonanopore_train_control(path_control, data_scale, mode, sampling_rate, section_length):#modeはblockageかrising
    ex = os.path.splitext(path_control)
    data_control = read_data(path_control, ex[1], data_scale)

    if data_control is None:
        print("Error: Invalid filetype")
    else:
        section_length_points = round(sampling_rate * section_length / 1000)
        num_sections = len(data_control) // section_length_points

        section_values = process_sections(data_control, num_sections, section_length_points, mode, sampling_rate)
        threshold = gumbel_dist_fit(section_values)
       
        return threshold
   

def read_data(path_control, filetype, data_scale):
    if filetype == '.bin':
        with open(path_control, mode='rb') as f:
            data = f.read()
        return np.frombuffer(data, np.float64)*data_scale #単位はpA
    elif filetype == '.abf':
        abf = pyabf.ABF(path_control)
        abf.setSweep(0)
        data = np.array(abf.sweepY, dtype=np.float64)
        data = data*data_scale #単位はpA
        return data
    else:
        return None


def process_sections(data_control, num_sections, section_length_points, mode, sampling_rate):
    section_values = []

    for i in range(num_sections):
        section_data = data_control[i*section_length_points:(i+1)*section_length_points]
        section_value = calculate_section_value(section_data, mode, sampling_rate)
        section_values.append(section_value)
   
    return section_values


def calculate_section_value(section_data, mode, sampling_rate):
    if mode == True:
        Ki = np.argmin(section_data)
    else:
        Ki = np.argmax(section_data)
    Pi = section_data[Ki]

    cutrange = round(5 * sampling_rate / 1000)
    if Ki <= cutrange:
        Bi = np.mean(section_data[Ki+cutrange:])
    elif Ki >= len(section_data) - cutrange:
        Bi = np.mean(section_data[:Ki-cutrange])
    else:
        Bi = np.mean(np.concatenate([section_data[:Ki-cutrange], section_data[Ki+cutrange:]]))
   
    if mode == True:
        Ai = Bi - Pi
    else:
        Ai = Pi - Bi

    return Ai


def gumbel_dist_fit(section_values):
    param = gumbel_r.fit(section_values)

    pv = kstest(section_values, 'gumbel_r', args=param)[1]

    return (pv, param)
   
#usage
#print(Autonanopore_control_train(r"D:\240626\pore2_1MKCl_control\-150\2024_06_26_0024.abf", "soaring"))