"""
Publicly shame users who are using too much GPU memory or compute.
"""

username_to_slack_names = {
    'hpinkard_waller': '@hpinkard',
    'vitran': '@@Vi Tran',
    'tiffany': '@Tiffany Chien',
    'mingxuan': '@Mingxuan',
    'rcao_waller': '@Ruiming',
    'clara_waller': '@Clara Hung',
    'keene_b': '@Keene Boondicharern'
}


import nvidia_smi
import psutil
import time
import numpy as np
import requests
import json

TIME_THRESHOLD_H = 3
GPU_MEM_THRESHOLD_GB = 5
GPU_COMPUTE_THRESHOLD = 20

INTERVAL_S = 30
MIN_MONITOR_DURATION_S = 60 

nvidia_smi.nvmlInit()  # initialize NVML library



def post_to_slack(message):

    # Set the API endpoint URL
    url = "https://hooks.slack.com/services/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"

    # Set the request headers and data
    headers = {'Content-type': 'application/json'}
    data = {'text': message}
    json_data = json.dumps(data)

    # Send the HTTP POST request
    response = requests.post(url, headers=headers, data=json_data)



def check_status():
    output = {}
    for gpu_index in range(nvidia_smi.nvmlDeviceGetCount()):  # iterate over all GPUs
        handle = nvidia_smi.nvmlDeviceGetHandleByIndex(gpu_index)  # get handle to the first GPU
        info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)  # get memory info
        # get the memory used by each process
        processes = nvidia_smi.nvmlDeviceGetComputeRunningProcesses(handle)
        for process in processes:
            # use the pid to get the user
            user = psutil.Process(process.pid).username()
            gb_memory = process.usedGpuMemory / 1024 / 1024 / 1024  # convert bytes to GB
            # get compute percentage
            compute = nvidia_smi.nvmlDeviceGetUtilizationRates(handle).gpu
            # print the user name, process id, and memory used by each process
            output[process.pid] = {'user': user, 
                                   'memory': [gb_memory], 
                                   'compute': [compute]}
    return output

accumulated = {}


while True:
    new_status = check_status()
    for pid in tuple(accumulated.keys()):
        if pid not in new_status.keys():
            del accumulated[pid] # its been cleared

    for pid in new_status.keys():
        if pid not in accumulated.keys():
            accumulated[pid] = new_status[pid]
        else:
            accumulated[pid]['memory'] += new_status[pid]['memory']
            accumulated[pid]['compute'] += new_status[pid]['compute']
    
    time.sleep(INTERVAL_S)  

    # get the average memory and compute usage for each process
    for pid in accumulated.keys():
        if len(accumulated[pid]['memory']) < MIN_MONITOR_DURATION_S / INTERVAL_S:
            continue
        average_memory = np.mean(accumulated[pid]['memory'])
        average_gpu_compute = np.mean(accumulated[pid]['compute'])
        # use the pid to figure out how long the process has been running
        process = psutil.Process(pid)
        process_running_time = time.time() - process.create_time()
        if average_memory > GPU_MEM_THRESHOLD_GB and average_gpu_compute < GPU_COMPUTE_THRESHOLD and process_running_time / 60 / 60 > TIME_THRESHOLD_H:
            print('{}\'s process has been running for {:.2f} hours and is using {:.2f} GB of memory and {:.2f}% of the GPU'.format(accumulated[pid]['user'], process_running_time / 60 / 60, average_memory, average_gpu_compute))
            post_to_slack('{}\'s process has been running for {:.2f} hours and is using {:.2f} GB of memory and {:.2f}% of the GPU'.format(accumulated[pid]['user'], process_running_time / 60 / 60, average_memory, average_gpu_compute))