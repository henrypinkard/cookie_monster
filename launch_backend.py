# !pip install nvidia-ml-py3

import threading
import nvidia_smi
import numpy as np
import time
import subprocess
import os
import shutil
import argparse

DEBUG = False

parser = argparse.ArgumentParser()
# Required arguments
parser.add_argument('train_script_path', help="train script to use", type=str)
parser.add_argument('status_dir', help="where config files kept", type=str)
parser.add_argument('save_dir', help="where trained models and data saved", type=str)
args = parser.parse_args()
train_script_path = args.train_script_path


# make root directory for model and logging
STATUS_DIR = args.status_dir
SAVING_DIR_ROOT = args.save_dir


USAGE_THRESHOLD = 30 # need less than 30 percent usage
MEMORY_THRESHOLD = 6 # need more than 6 GB

gpu_status_delta_t = 1 #seconds
gpu_status_average_window = 60 * 30  #seconds
# gpu_status_average_window = 0.2 * 30  #seconds


        
def check_GPU_status(index):
    handle = nvidia_smi.nvmlDeviceGetHandleByIndex(index)
    info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
    free_GB = info.free / 1024 ** 3
    gpu_usage = nvidia_smi.nvmlDeviceGetUtilizationRates(handle).gpu
    return free_GB, gpu_usage


def launch_training(gpu_index, saving_dir, config_file_path, stdout_path, debug=DEBUG):
    """
    Commence training of new model
    """
    if not os.path.exists(config_file_path):
        raise Exception("Config file doesnt exist", config_file_path)
    process = subprocess.Popen(["ipython", train_script_path, str(gpu_index), str(saving_dir), config_file_path],
                          stdout=subprocess.PIPE)     
    
    
    # launch a thread for monitoring 

    def read_process_output():
        # Read the stdout line by line
        
        
        with open(stdout_path + '/process_output.txt', 'a') as f:
            while True:
                line = process.stdout.readline()
                if not line:
                    # The process has exited, so shut down the thread
                    break       
                f.write(line.decode())
                f.flush()
                # Check if the process has exited (still needed?)
                exit_code = process.poll()
                if exit_code is not None:
                    # The process has exited
                    print('Process has exited with code', exit_code)  
                    return

    # Create and start the new thread
    thread = threading.Thread(target=read_process_output)
    thread.start()

    if debug:
        while True:
            line = process.stdout.readline()
            print(line)
            if not line: break
    return process


def create_saving_dir(saving_dir_root, model_name, overwrite):
    # Add unique suffix to experiment replicate
    dest = saving_dir_root + model_name
    if overwrite:
        if os.path.exists(dest):
            shutil.rmtree(dest)
    else:
        if os.path.exists(dest):
            raise Exception('saving dir already exists: ' + dest)
    os.mkdir(dest)
    return dest   
            

def check_for_kill_flag(active_training_processes, gpu_indices, gpu_usage, gpu_memory):
    flags = os.listdir(STATUS_DIR + '/stop_training')
    # clear flag
    with open(STATUS_DIR + '/stop_training/stop', 'r+') as f:
        gpu_index = f.readline()
        if len(gpu_index) == 0:
            return gpu_usage, gpu_memory
        f.truncate(0) #clear file
    # kill a process
    if len(active_training_processes) > 0:
        print("killing process")
        process_keys = list(active_training_processes.keys())
        unlucky_process = active_training_processes[process_keys[0]]
        for key in process_keys:
            if int(gpu_indices[key]) == int(gpu_index):
                unlucky_process = active_training_processes[key]
                print("killing process {}".format(key))
        unlucky_process.kill()
        # reset wait to start a new one
        gpu_usage = {index: [] for index in range(num_GPUs)}
        gpu_memory = {index: [] for index in range(num_GPUs)}

    return gpu_usage, gpu_memory

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
    # monitor for available GPUs

            
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
    for config_file_name in os.listdir(STATUS_DIR + 'training'):
        if config_file_name not in active_training_processes.keys():
            # if its not an active process its either complete or failed
            if 'complete.txt' in os.listdir(SAVING_DIR_ROOT + config_file_name[:-5]):
                # move config file to complete directory
                shutil.move(STATUS_DIR + 'training/' + config_file_name, STATUS_DIR + 'complete/' + config_file_name)
            else:
                configs_to_retry.append(config_file_name)

                
    #check for kill flag
    gpu_usage, gpu_memory = check_for_kill_flag(active_training_processes, gpu_indices, gpu_usage, gpu_memory)
            
                
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
        pending_configs = [s for s in os.listdir(STATUS_DIR + 'pending') if s.endswith(".yaml")]
        if len(configs_to_retry) > 0:
            config_file_name = configs_to_retry.pop(0)
            saving_dir = create_saving_dir(SAVING_DIR_ROOT, config_file_name[:-5], overwrite=True)
        elif len(pending_configs) > 0:
            config_file_name = pending_configs.pop(0)
            saving_dir = create_saving_dir(SAVING_DIR_ROOT, config_file_name[:-5], overwrite=False)
            print('moving pending config file to training')
            shutil.move(STATUS_DIR + 'pending/' + config_file_name, STATUS_DIR + 'training/' + config_file_name)
        else:
            print('waiting for something to train')
            break #nothing to train
        
        # prefer GPUs with most available memory:
        free_mem = np.array([np.mean(gpu_memory[index]) for index in available_GPUs])
        to_use = np.argmax(free_mem)
        print('next to use ', available_GPUs[to_use])
        gpu_index = available_GPUs.pop(to_use)
        config_file_path = STATUS_DIR + 'training/' + config_file_name
        active_training_processes[config_file_name] = launch_training(gpu_index, saving_dir, config_file_path, saving_dir)
        gpu_indices[config_file_name] = gpu_index
        print("Launching training {} on GPU {}".format(config_file_path, gpu_index))


        
    time.sleep(gpu_status_delta_t)

print('Exiting loop. This shouldnt happen. Why did this happen')
# nvidia_smi.nvmlShutdown()