# !pip install nvidia-ml-py3

import threading
import nvidia_smi
import numpy as np
import time
import subprocess
import os
import shutil
import argparse

        
def print_status_update(gpu_launch_times, gpu_kill_times, available_GPUs, gpu_indices, configs_to_retry, num_GPUs, 
                GPU_LAUNCH_DELAY, GPU_KILL_DELAY, CONFIG_FILE_DIR):
    for gpu_index in range(num_GPUs):
        # delay from it being killed
        if gpu_index in gpu_kill_times.keys() and \
            time.time() - gpu_kill_times[gpu_index] < GPU_KILL_DELAY:
            print('GPU ', gpu_index, ' is being left open for {:.2f}'.format(
                (GPU_KILL_DELAY - (time.time() - gpu_kill_times[gpu_index])) / 60), ' minutes')
        # delay from another process launching on this GPU
        if gpu_index in gpu_launch_times.keys() and \
                time.time() - gpu_launch_times[gpu_index] < GPU_LAUNCH_DELAY:
            print('GPU ', gpu_index, ' is being left open for {:.2f}'.format(
                (GPU_LAUNCH_DELAY - (time.time() - gpu_launch_times[gpu_index])) / 60), ' minutes')
    print('available GPUs: ', available_GPUs)
    # print which configs are training on which GPUs
    for index in range(num_GPUs):
        if index in gpu_indices.values():
            print('GPU ', index, ' is training ')
            for config_file, gpu_index in gpu_indices.items():
                if gpu_index == index:
                    print('\t', config_file)
    print('configs to retry: \n', configs_to_retry)
    for config in configs_to_retry:
        print('\t', config)
    pending_configs = [s for s in os.listdir(CONFIG_FILE_DIR + 'pending') if s.endswith(".yaml")]
    print(len(pending_configs), ' configs pending', '\n\n\n')


def check_GPU_status(index):
    handle = nvidia_smi.nvmlDeviceGetHandleByIndex(index)
    info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
    free_GB = info.free / 1024 ** 3
    gpu_usage = nvidia_smi.nvmlDeviceGetUtilizationRates(handle).gpu
    return free_GB, gpu_usage


def launch_training(train_script_path, gpu_index, saving_dir, config_file_path, stdout_path):
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

    # if debug:
    #     while True:
    #         line = process.stdout.readline()
    #         print(line)
    #         if not line: break
    return process


def create_saving_dir(saving_dir_root, model_name):
    """
    Create a directory for saving the model, overwriting if needed
    """
    dest = saving_dir_root + model_name
    if os.path.exists(dest):
        shutil.rmtree(dest)
    os.mkdir(dest)
    return dest   
            

def check_for_kill_flag(active_training_processes, gpu_indices):
    # Get home directory
    home = os.path.expanduser("~")
    flags = os.listdir(home + '/stop_training')
    # clear flag
    with open(home + '/stop_training/stop', 'r+') as f:
        gpu_index = f.readline()
        if len(gpu_index) == 0:
            return 
        f.truncate(0) #clear file
    # kill a process
    if len(active_training_processes) > 0:
        print("killing process")
        process_keys = list(active_training_processes.keys())
        
        if int(gpu_index) not in gpu_indices.values():
            unlucky_process = active_training_processes[process_keys[0]]
            unlucky_process.kill()
            return gpu_indices[process_keys[0]]    # return the gpu index of the unlucky process 
        for key in process_keys:
            # kill all processes on a specific GPU
            if int(gpu_indices[key]) == int(gpu_index):
                cleared_gpu_index = int(gpu_index)
                unlucky_process = active_training_processes[key]
                unlucky_process.kill()
                print("killing process {}".format(key))
        
        return cleared_gpu_index
