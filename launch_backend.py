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
from cookie_monster_lib import launch_training, check_GPU_status, \
    create_saving_dir, check_for_kill_flag, print_status_update
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


USAGE_THRESHOLD = 60 # need less than 30 percent usage (average)
FREE_MEMORY_THRESHOLD = 6 # need more than 6 GB free memory at all times over last 10 minutes

UPDATE_INTERVAL = 30 #seconds

GPU_KILL_DELAY = 60 * 60 * 6 # 3 hours
GPU_LAUNCH_DELAY = 60 * 15 # Wait min before launching new process on same GPU


gpu_status_delta_t = 1 #seconds
gpu_status_average_window = 60 * 10  # how long it will wait to decide if a GPU is free
# gpu_status_average_window = 60 * 3  #seconds

num_gpu_samples = gpu_status_average_window // gpu_status_delta_t

active_training_processes = {} #map from model/config file names to paths
gpu_indices = {} #map from model/config file names to gpu indices
gpu_launch_times = {} #map from gpu indices to time of last launch on the

#start up GPU status monitoring
nvidia_smi.nvmlInit()
num_GPUs = nvidia_smi.nvmlDeviceGetCount() 
gpu_usage = {index: [] for index in range(num_GPUs)}
gpu_memory = {index: [] for index in range(num_GPUs)}
gpu_kill_times = {index: 0 for index in range(num_GPUs)}

last_status_time = -1

while True:
    start_time = time.time()
            
    # clear completed processes
    for config_file in list(active_training_processes.keys()):
        process = active_training_processes[config_file]
        if process.poll() is not None:
            print('clearing process ', config_file)
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
    killed_gpu_index = check_for_kill_flag(active_training_processes, gpu_indices)
    if killed_gpu_index is not None:
        print('killed gpu index: ', killed_gpu_index)
        gpu_kill_times[killed_gpu_index] = time.time()
                
    available_GPUs = []
    for gpu_index in range(num_GPUs):
          
        # record usage
        free_memory, usage = check_GPU_status(gpu_index)
        gpu_usage[gpu_index].append(usage)
        gpu_memory[gpu_index].append(free_memory)
        if len(gpu_usage[gpu_index]) > num_gpu_samples:
            gpu_usage[gpu_index] = gpu_usage[gpu_index][-num_gpu_samples:]
            gpu_memory[gpu_index] = gpu_memory[gpu_index][-num_gpu_samples:]
        # compute averages
        average_usage = np.mean(gpu_usage[gpu_index])
        average_memory = np.mean(gpu_memory[gpu_index])
        if average_usage < USAGE_THRESHOLD and usage < USAGE_THRESHOLD and \
            free_memory > FREE_MEMORY_THRESHOLD and average_memory > FREE_MEMORY_THRESHOLD:
            available_GPUs.append(gpu_index)
            if gpu_index in gpu_kill_times.keys():
                if time.time() - gpu_kill_times[gpu_index] < GPU_KILL_DELAY and gpu_index in available_GPUs:
                    available_GPUs.remove(gpu_index)
                else:
                    del gpu_kill_times[gpu_index] # reset kill time   
            if gpu_index in gpu_launch_times.keys() and \
                    time.time() - gpu_launch_times[gpu_index] < GPU_LAUNCH_DELAY and gpu_index in available_GPUs:
                available_GPUs.remove(gpu_index)
 

        if time.time() - last_status_time > UPDATE_INTERVAL:   
            print('GPU ', gpu_index, ' utilization: {:.2f}'.format( average_usage),
             '%, {:.2f}'.format(average_memory), 'GB free')

    if time.time() - last_status_time > UPDATE_INTERVAL:
        print_status_update(gpu_launch_times, gpu_kill_times, available_GPUs, gpu_indices, configs_to_retry, num_GPUs, 
                GPU_LAUNCH_DELAY, GPU_KILL_DELAY, CONFIG_FILE_DIR)
        
        last_status_time = time.time()


    if len(available_GPUs) == 0:
        # print('No GPUs available')
        time.sleep(gpu_status_delta_t)
        continue


    
    while True:
        if len(available_GPUs) == 0: 
            # print('no GPUs available')
            break
        
        # if len(available_GPUs) == 1 and len(gpu_usage[available_GPUs[0]]) < num_gpu_samples:
        #     print("waiting before hogging last GPU{:.2f}%".format(len(gpu_usage[available_GPUs[0]]) / num_gpu_samples *100))
        #     break
        
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
            should_resume =  not config['scheduler']['resume_training']
            # if it does exist, and youre not resuming, then delete the model dir
            dest = SAVING_DIR_ROOT + config_file_name[:-5]
            if os.path.exists(dest) and not should_resume:
                shutil.rmtree(dest)
            elif should_resume and not os.path.exists(dest):
                warnings.warn('trying to resume training but model dir does not exist')
            saving_dir = create_saving_dir(SAVING_DIR_ROOT, config_file_name[:-5])
            # else it should already exist
        elif len(pending_configs) > 0:
            config_file_name = pending_configs.pop(0)
            saving_dir = create_saving_dir(SAVING_DIR_ROOT, config_file_name[:-5])
            print('moving pending config file to training')
            config_path = CONFIG_FILE_DIR + 'training/' + config_file_name
            shutil.move(CONFIG_FILE_DIR + 'pending/' + config_file_name, config_path)
            with open(config_path, "r") as stream:
                config = yaml.safe_load(stream)
        else:
            # print('waiting for something to train')
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
        # print('next to use ', available_GPUs[to_use])
        gpu_index = available_GPUs.pop(to_use)
        config_file_path = CONFIG_FILE_DIR + 'training/' + config_file_name
        active_training_processes[config_file_name] = launch_training(train_script_path, gpu_index, saving_dir, config_file_path, saving_dir)
        gpu_indices[config_file_name] = gpu_index
        # reset memory and usage history for this GPU so that another training waits before launching
        print('\n######################################################')
        print("\nGPU {}: Launched training {}".format(gpu_index, config_file_path))
        print('######################################################\n')
        gpu_launch_times[int(gpu_index)] = time.time()
        # reset these because they are now out of date
        gpu_usage[gpu_index] = []
        gpu_memory[gpu_index] = []


        
    time.sleep(gpu_status_delta_t)

print('Exiting loop. This shouldnt happen. Why did this happen')
# nvidia_smi.nvmlShutdown()