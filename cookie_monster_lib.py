# !pip install nvidia-ml-py3

import threading
import nvidia_smi
import numpy as np
import time
import subprocess
import os
import shutil
import argparse

        
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


def create_saving_dir(saving_dir_root, model_name, overwrite):
    # Add unique suffix to experiment replicate
    dest = saving_dir_root + model_name
    if overwrite:
        if os.path.exists(dest):
            shutil.rmtree(dest)
        os.mkdir(dest)
    else:
        if os.path.exists(dest):
            raise Exception("Experiment already exists")
        os.mkdir(dest)
    return dest   
            

def check_for_kill_flag(active_training_processes, gpu_indices, gpu_usage, gpu_memory, num_GPUs):
    # Get home directory
    home = os.path.expanduser("~")
    flags = os.listdir(home + '/stop_training')
    # clear flag
    with open(home + '/stop_training/stop', 'r+') as f:
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
