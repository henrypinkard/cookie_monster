# !pip install nvidia-ml-py3

import threading
import nvidia_smi
import numpy as np
import time
from datetime import datetime  
import subprocess
import os
import shutil
import argparse
from cookie_monster_lib import launch_training, check_GPU_status, create_saving_dir, check_for_kill_flag
import yaml

DEBUG = False

parser = argparse.ArgumentParser()
# Required arguments
parser.add_argument('train_script_path', help="train script to use", type=str)
parser.add_argument('config_dir', help="where config files kept", type=str)
parser.add_argument('save_dir', help="where trained models and data saved", type=str)
args = parser.parse_args()
train_script_path = args.train_script_path


# make root directory for model and logging
CONFIG_FILE_DIR = args.config_dir
SAVING_DIR_ROOT = args.save_dir


USAGE_THRESHOLD = 30 # need less than 30 percent usage
MEMORY_THRESHOLD = 6 # need more than 6 GB  

gpu_status_delta_t = 1 #seconds
gpu_status_average_window = 60 * 30  #seconds
# gpu_status_average_window = 0.2 * 30  #seconds


num_gpu_samples = gpu_status_average_window // gpu_status_delta_t

active_training_processes = {} #map from model/config file names to paths
gpu_indices = {}

#start up GPU status monitoring
nvidia_smi.nvmlInit()
num_GPUs = nvidia_smi.nvmlDeviceGetCount() 
gpu_usage = {index: [] for index in range(num_GPUs)}
gpu_memory = {index: [] for index in range(num_GPUs)}


while True:
    start_time = time.time()
            
    # clear completed processes
    for config_file in list(active_training_processes.keys()):
        process = active_training_processes[config_file]
        if process.poll() is not None:
            print('clearing process')
            # its finished
            del active_training_processes[config_file]
            del gpu_indices[config_file]


    # update config files
    configs_to_retry = []
    for config_file_name in os.listdir(CONFIG_FILE_DIR + 'training'):
        if config_file_name not in active_training_processes.keys():
            # if its not an active process its either complete or failed
            if 'complete.txt' in os.listdir(SAVING_DIR_ROOT + config_file_name[:-5]):
                # move config file to complete directory
                shutil.move(CONFIG_FILE_DIR + 'training/' + config_file_name, CONFIG_FILE_DIR + 'complete/' + config_file_name)
            else:
                configs_to_retry.append(config_file_name)

                
    #check for kill flag
    gpu_usage, gpu_memory = check_for_kill_flag(active_training_processes, gpu_indices, gpu_usage, gpu_memory, num_GPUs)
            
                
    available_GPUs = []
    for index in range(num_GPUs):
          
        # record usage
        memory, usage = check_GPU_status(index)
        gpu_usage[index].append(usage)
        gpu_memory[index].append(memory)
        if len(gpu_usage[index]) > num_gpu_samples:
            gpu_usage[index] = gpu_usage[index][-num_gpu_samples:]
            gpu_memory[index] = gpu_memory[index][-num_gpu_samples:]
        # compute averages
        average_usage = np.mean(gpu_usage[index])
        average_memory = np.mean(gpu_memory[index])
        if average_usage < USAGE_THRESHOLD and usage < USAGE_THRESHOLD and \
            memory > MEMORY_THRESHOLD and average_memory > MEMORY_THRESHOLD:
            available_GPUs.append(index)
        else:
            print('GPU ', index, ' unavailable')
    for gpu_index in gpu_indices.values():
        if gpu_index in available_GPUs:
            available_GPUs.remove(gpu_index)
    
    if len(available_GPUs) == 0:
        print('No GPUs available')
        time.sleep(gpu_status_delta_t)
        continue


    
    while True:
        if len(available_GPUs) == 0: 
            print('no GPUs available')
            break
        
        if len(available_GPUs) == 1 and len(gpu_usage[available_GPUs[0]]) < num_gpu_samples:
            print("waiting before hogging last GPU{:.2f}%".format(len(gpu_usage[available_GPUs[0]]) / num_gpu_samples *100))
            break
        
        print('available gpus', available_GPUs)
        # anything to train?
        #    Check for failed attempts
        #    Check for pending config files
        pending_configs = [s for s in os.listdir(CONFIG_FILE_DIR + 'pending') if s.endswith(".yaml")]
        if len(configs_to_retry) > 0:
            config_file_name = configs_to_retry.pop(0)
            config_path = CONFIG_FILE_DIR + 'training/' + config_file_name
            # load config
            with open(config_path, "r") as stream:
                config = yaml.safe_load(stream)
            overwrite =  not config['scheduler']['resume_training']
            saving_dir = create_saving_dir(SAVING_DIR_ROOT, config_file_name[:-5], overwrite=overwrite)
        elif len(pending_configs) > 0:
            config_file_name = pending_configs.pop(0)
            saving_dir = create_saving_dir(SAVING_DIR_ROOT, config_file_name[:-5], overwrite=False)
            print('moving pending config file to training')
            config_path = CONFIG_FILE_DIR + 'training/' + config_file_name
            shutil.move(CONFIG_FILE_DIR + 'pending/' + config_file_name, config_path)
            with open(config_path, "r") as stream:
                config = yaml.safe_load(stream)
        else:
            print('waiting for something to train')
            break #nothing to train
        
        # Update the number of attempts and resave config file

        config['scheduler']['attempt_number'] = config['scheduler']['attempt_number'] + 1
        config['scheduler']['date'] = datetime.now().strftime("%Y-%m-%d")
        if 'training' in config.keys():
            del config['training']
        with open( CONFIG_FILE_DIR + 'training/' + config_file_name , 'w') as file:
            documents = yaml.dump(config, file)

        # prefer GPUs with most available memory:
        free_mem = np.array([np.mean(gpu_memory[index]) for index in available_GPUs])
        to_use = np.argmax(free_mem)
        print('next to use ', available_GPUs[to_use])
        gpu_index = available_GPUs.pop(to_use)
        config_file_path = CONFIG_FILE_DIR + 'training/' + config_file_name
        active_training_processes[config_file_name] = launch_training(train_script_path, gpu_index, saving_dir, config_file_path, saving_dir)
        gpu_indices[config_file_name] = gpu_index
        print("Launching training {} on GPU {}".format(config_file_path, gpu_index))


        
    time.sleep(gpu_status_delta_t)

print('Exiting loop. This shouldnt happen. Why did this happen')
# nvidia_smi.nvmlShutdown()