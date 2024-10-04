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
from cookie_monster_backend_lib import launch_training, check_GPU_status, \
    check_for_kill_flag, print_status_update, run_server, load_config

import yaml
import psutil
import sys



# corectly number GPUs
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 

DEBUG = False

parser = argparse.ArgumentParser()
# # Required arguments
parser.add_argument('config_dir', help="where config files kept", type=str)
args = parser.parse_args()
CONFIG_FILE_DIR = args.config_dir
if CONFIG_FILE_DIR[-1] != os.sep:
    CONFIG_FILE_DIR += os.sep
# check if complete, pending, staging, traing, and abandoned directories exist and if not create them
for dir_name in ['complete', 'pending', 'staging', 'training', 'abandoned']:
    path = CONFIG_FILE_DIR + dir_name
    if not os.path.exists(path):
        # mkdir -p
        os.makedirs(path)
        print('created directory: ', CONFIG_FILE_DIR + dir_name)

# read port number from env variable
PORT_NUMBER = int(os.environ['COOKIE_MONSTER_PORT_NUMBER'])


# where to also write the STDOUT of this process
# COOKIE_MONSTER_LOGS_DIR = "/home/hpinkard_waller/10tb_extension/cookie_monster_logs" # CHANGE TO LOG PATH, #os.path.expanduser('~') + '/cookie_monster_logs'
# get it from env variable, or if it doesnt exist, throw an error
COOKIE_MONSTER_LOGS_DIR = os.environ['COOKIE_MONSTER_LOGS_DIR']
if COOKIE_MONSTER_LOGS_DIR is None:
    raise Exception('COOKIE_MONSTER_LOGS_DIR environment variable not set. Please set it to'
                    'the directory where you want to store logs in .bashrc or.zshrc')
# create it if it doesnt exist
if not os.path.exists(COOKIE_MONSTER_LOGS_DIR):
    os.mkdir(COOKIE_MONSTER_LOGS_DIR)

# check status of GPUs every RESOURCE_STATUS_CHECK_INTERVAL seconds and keep a history of RESOURCE_STATUS_CHECK_WINDOW seconds
RESOURCE_STATUS_CHECK_INTERVAL = 1 # seconds
RESOURCE_STATUS_CHECK_WINDOW = 60 * 30  # seconds

# Consider a GPU free if its usage, memory, and system RAM meet these thresholds
USAGE_THRESHOLD = 70 # need less than 70 percent usage (average)
FREE_GPU_MEMORY_THRESHOLD = 6 # need > 6 GB free memory at all times over last 10 minutes
FREE_RAM_THRESHOLD = 20 # Dont train if less than this amount of RAM is free

# print updates to console/file every
UPDATE_INTERVAL = 240 # seconds

# if someone kills a process using *stopit script, wait at least GPU_KILL_DELAY_H hours before launching a new process on the same GPU
GPU_KILL_DELAY_H = 2 # hours
# After launching a process on a given GPU, wait at least GPU_LAUNCH_DELAY seconds before launching another process on the same GPU
PROCESS_LAUNCH_DELAY = 60 * 8 # seconds, wait before launching new process



active_experiments = {} # map from model/config file names Objects
gpu_launch_times = {} # map from gpu indices to time of last launch on the

#start up GPU status monitoring
nvidia_smi.nvmlInit()
num_GPUs = nvidia_smi.nvmlDeviceGetCount() 
num_gpu_samples = RESOURCE_STATUS_CHECK_WINDOW // RESOURCE_STATUS_CHECK_INTERVAL
gpu_usage = {index: [] for index in range(num_GPUs)}
gpu_memory = {index: [] for index in range(num_GPUs)}
available_RAM = []
gpu_resume_times = {index: 0 for index in range(num_GPUs)}

last_status_time = -1


# create a cookie_monster_logs dir in the home dir if it doesn't exist
if not os.path.exists(COOKIE_MONSTER_LOGS_DIR):
    os.mkdir(COOKIE_MONSTER_LOGS_DIR)


# make a class that prints everything to a file as well as the console
class Logger:
 
    def __init__(self, filename):
        self.console = sys.stdout
        self.file = open(filename, 'w')
 
    def write(self, message):
        self.console.write(message)
        self.file.write(message)
 
    def flush(self):
        self.console.flush()
        self.file.flush()
 
# Open file for writing
filename = COOKIE_MONSTER_LOGS_DIR + '/cookie_monster_log_' + datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '.txt'
sys.stdout = Logger(filename)


request_queue, response_queue = run_server(port=PORT_NUMBER)

while True:
    start_time = time.time()
            
    # clear completed processes
    for config_file in list(active_experiments.keys()):
        process = active_experiments[config_file].process
        if process.poll() is not None:
            print('clearing process ', config_file)
            # its finished
            del active_experiments[config_file]


    # update config files
    configs_to_retry = []
    for config_file_name in os.listdir(CONFIG_FILE_DIR + 'training'):
        config = load_config(CONFIG_FILE_DIR + 'training/' + config_file_name)
        if config_file_name not in active_experiments.keys():
            # if its not an active process its either complete or failed
            saving_dir = config['saving_dir'] + config_file_name[:-5]
            if 'complete.txt' in os.listdir(saving_dir):
                # move config file to complete directory
                shutil.move(CONFIG_FILE_DIR + 'training/' + config_file_name, CONFIG_FILE_DIR + 'complete/' + config_file_name)
                # copy config file to saving dir
                shutil.copy(CONFIG_FILE_DIR + 'complete/' + config_file_name, saving_dir + os.sep + config_file_name)
            else:
                configs_to_retry.append(config_file_name)

                
    #check for kill flag
    killed_gpu_index, delay = check_for_kill_flag(request_queue, response_queue, active_experiments, GPU_KILL_DELAY_H)
    if killed_gpu_index is not None:
        print('killed gpu index: ', killed_gpu_index, '  with resume delay: ', delay)
        gpu_resume_times[killed_gpu_index] = time.time() / 60 ** 2 + delay
                
    available_GPUs = []
    time_since_last_process_launch = time.time() - max(gpu_launch_times.values() if len(gpu_launch_times) > 0 else [0])
    for gpu_index in range(num_GPUs):
        
        # record usage
        free_memory, usage = check_GPU_status(gpu_index)
        gpu_usage[gpu_index].append(usage)
        gpu_memory[gpu_index].append(free_memory)
        available_RAM.append(psutil.virtual_memory().available / 1024 ** 3)
        if len(gpu_usage[gpu_index]) > num_gpu_samples:
            gpu_usage[gpu_index] = gpu_usage[gpu_index][-num_gpu_samples:]
            gpu_memory[gpu_index] = gpu_memory[gpu_index][-num_gpu_samples:]
            available_RAM = available_RAM[-num_gpu_samples:]

        # compute averages
        average_usage = np.mean(gpu_usage[gpu_index])
        average_memory = np.mean(gpu_memory[gpu_index])
        min_free_ram = np.percentile(available_RAM, 5)
        enough_ram_for_launch = min_free_ram > FREE_RAM_THRESHOLD
        if average_usage < USAGE_THRESHOLD and usage < USAGE_THRESHOLD and \
            free_memory > FREE_GPU_MEMORY_THRESHOLD and average_memory > FREE_GPU_MEMORY_THRESHOLD:
            available_GPUs.append(gpu_index)
            if gpu_index in gpu_resume_times.keys():
                if gpu_resume_times[gpu_index] - time.time() / 60 ** 2 > 0 and gpu_index in available_GPUs:
                    available_GPUs.remove(gpu_index)
                else:
                    del gpu_resume_times[gpu_index] # the time has elapsed, clear it   
            if gpu_index in gpu_launch_times.keys() and \
                    time.time() - gpu_launch_times[gpu_index] < PROCESS_LAUNCH_DELAY and gpu_index in available_GPUs:
                available_GPUs.remove(gpu_index)


        if time.time() - last_status_time > UPDATE_INTERVAL:   
            print('GPU ', gpu_index, ' utilization: {:.2f}'.format( average_usage),
            '%, {:.2f}'.format(average_memory), 'GB free')

    if time.time() - last_status_time > UPDATE_INTERVAL:
        print_status_update(active_experiments, gpu_launch_times, gpu_resume_times, available_GPUs, configs_to_retry, num_GPUs, min_free_ram,
                PROCESS_LAUNCH_DELAY, CONFIG_FILE_DIR)
        
        last_status_time = time.time()


    if len(available_GPUs) == 0:
        # print('No GPUs available')
        time.sleep(RESOURCE_STATUS_CHECK_INTERVAL)
        continue


    if not enough_ram_for_launch:
        time.sleep(RESOURCE_STATUS_CHECK_INTERVAL)
        continue

    if time_since_last_process_launch < PROCESS_LAUNCH_DELAY:
        time.sleep(RESOURCE_STATUS_CHECK_INTERVAL)
        continue
    
    if len(available_GPUs) == 0: 
        # print('no GPUs available')
        continue
    
    # if len(available_GPUs) == 1 and len(gpu_usage[available_GPUs[0]]) < num_gpu_samples:
    #     print("waiting before hogging last GPU{:.2f}%".format(len(gpu_usage[available_GPUs[0]]) / num_gpu_samples *100))
    #     break
    
    # Determine if anything to train by
    #    Check for failed attempts
    #    Check for pending config files
    pending_configs = [s for s in os.listdir(CONFIG_FILE_DIR + 'pending') if s.endswith(".yaml")]
    # sort so the ones added to this folder first get launched first
    pending_configs.sort(key=lambda x: os.path.getctime(os.path.join(CONFIG_FILE_DIR + 'pending', x)))
    if len(configs_to_retry) > 0:
        # take the most recently modified one
        config_file_name = configs_to_retry.pop(-1)   
        config = load_config(CONFIG_FILE_DIR + '/training/' + config_file_name)
        experiment_saving_dir = config['saving_dir'] + config_file_name[:-5]
    elif len(pending_configs) > 0:
        config_file_name = pending_configs.pop(-1)
        config = load_config(CONFIG_FILE_DIR + '/pending/' + config_file_name)
        # always erase the model dir if the config is coming from pending
        experiment_saving_dir = config['saving_dir'] + config_file_name[:-5]
        if os.path.exists(experiment_saving_dir):
            shutil.rmtree(experiment_saving_dir)
        os.makedirs(experiment_saving_dir, exist_ok=True)
        print('moving pending config file to training')
        shutil.move(CONFIG_FILE_DIR + 'pending/' + config_file_name, CONFIG_FILE_DIR + 'training/' + config_file_name)
    else:
        # print('waiting for something to train')
        continue #nothing to train
    
    config_path = CONFIG_FILE_DIR + 'training/' + config_file_name

    # prefer empty GPUs
    to_use = None
    for i, gpu_index in enumerate(available_GPUs):
        handle = nvidia_smi.nvmlDeviceGetHandleByIndex(gpu_index)  # get handle to the first GPU
        info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)  # get memory info
        # get the memory used by each process
        processes = nvidia_smi.nvmlDeviceGetComputeRunningProcesses(handle)
        if len(processes) == 0:
            to_use = i
            break
    # if none are empty, prefer GPUs with most available memory:
    if to_use is None:
        free_mem = np.array([np.mean(gpu_memory[index]) for index in available_GPUs])
        to_use = np.argmax(free_mem)
    
    gpu_index = available_GPUs.pop(to_use)
    config_file_path = CONFIG_FILE_DIR + 'training/' + config_file_name
    active_experiments[config_file_name] = launch_training(gpu_index, experiment_saving_dir, config_file_path, experiment_saving_dir)
    # reset memory and usage history for this GPU so that another training waits before launching
    print('\n######################################################')
    print("\nGPU {}: Launched training {}".format(gpu_index, config_file_path))
    print('######################################################\n')
    gpu_launch_times[int(gpu_index)] = time.time()
    # reset these because they are now out of date
    gpu_usage[gpu_index] = []
    gpu_memory[gpu_index] = []

    time.sleep(RESOURCE_STATUS_CHECK_INTERVAL)

print('Exiting loop. This shouldnt happen. Why did this happen')
nvidia_smi.nvmlShutdown()

