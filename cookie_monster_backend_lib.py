# !pip install nvidia-ml-py3

import threading
import nvidia_smi
import numpy as np
import time
import subprocess
import os
import shutil
import argparse
import psutil
import builtins
import codecs
from datetime import datetime
import yaml
import requests
import queue


from http.server import BaseHTTPRequestHandler, HTTPServer
import json


class HTTPRequestHandler(BaseHTTPRequestHandler):
    
    request_queue = queue.Queue()
    response_queue = queue.Queue()

    def __init__(self, request, client_address, server):
        super().__init__(request, client_address, server)

    def do_POST(self):
        if self.path == '/messages':
            try:
                # Read the message data from the request
                content_length = int(self.headers['Content-Length'])
                message_data = self.rfile.read(content_length)
                message = json.loads(message_data)

                # Put a message into the queue
                self.request_queue.put((int(message["gpu_index"]), int(message["delay_time"]), message["user"]))

                # wait for there to be a message in the queue
                while self.response_queue.empty():
                    time.sleep(0.1)

                # Send a response back to the client
                self.send_response(200)
                self.send_header('Content-type', 'text/html')
                self.end_headers()
                response_content = self.response_queue.get()
                self.wfile.write(bytes(response_content, 'utf-8'))
            except:
                self.send_error(500)
                # print stacktrace
                import traceback
                traceback.print_exc()   
        else:
            self.send_error(404)


def run_server(port=7888):
    server_address = ('', port)
    httpd = HTTPServer(server_address, HTTPRequestHandler)
    print(f'Starting server on port {port}...')
    # strart on a new thread
    thread = threading.Thread(target=httpd.serve_forever)
    thread.daemon = True
    thread.start()
    return HTTPRequestHandler.request_queue, HTTPRequestHandler.response_queue
    




class TrainingProcess:

    def __init__(self, config, config_file_path, gpu_index, process):
        self.config = config
        self.name = config_file_path.split('/')[-1][:-5]
        self.config_file_path = config_file_path
        self.gpu_index = gpu_index
        self.process = process
        self.start_time = time.time()
        self.immortal = config['options']['immortal']

def load_config(config_path):
    with open(config_path, "r") as stream:
        config = yaml.safe_load(stream)
    return config

def train_script_complete(saving_dir):
    with open(saving_dir + "/complete.txt", mode="w") as f:
        f.write('complete')


def train_script_setup():

    ############# Directory and GPU setup #############
    parser = argparse.ArgumentParser()
    # Required arguments
    parser.add_argument('GPU_index', help="GPU_index", type=str)
    parser.add_argument('saving_dir', help="Directory where model and associated files saved", type=str)
    parser.add_argument('config_file_path', help="path to yaml file", type=str)

    #optional arguments
    # parser.add_argument('--othernamed', help="test", type=str)

    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
    os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU_index

    print(args.config_file_path)
    with open(args.config_file_path, "r") as stream:
        config = yaml.safe_load(stream)


    if 'training' in config and 'elapsed' in config['training']:
        if config['options']['resume_training']:
            already_elapsed_time = config['training']['elapsed']
        else:
            already_elapsed_time = 0
        attempt_number = config['training']['attempt_number']
        date = config['training']['start_date']
    else:
        already_elapsed_time = 0
        attempt_number = 0
        date = datetime.now().strftime("%Y-%m-%d")

    print('already elapsed from previous training: ' + str(already_elapsed_time / 60 ** 2) + 'h')

    config['training'] = {}

    config['training']['attempt_number'] = attempt_number + 1
    config['training']['start_date'] = date
    config['training']['elapsed'] = already_elapsed_time


    saving_dir = args.saving_dir
    hyperparameters = config["hyperparameters"]
    tensorboard_dir = saving_dir + '/tensorboard/'
    logging_dir = saving_dir + '/other_logs/'
    model_dir = saving_dir + '/model/'
    resume_backup_dir = saving_dir + '/resume_backup/'

    config['training']['tensorboard_dir'] = tensorboard_dir
        
    #Resave config file
    with open(args.config_file_path, 'w') as file:
        documents = yaml.dump(config, file)
        


    if config['options']['hog_memory'] == False:
        os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] ='false'
        os.environ['XLA_PYTHON_CLIENT_ALLOCATOR']='platform'
        os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'



    should_resume =  config['options']['resume_training']
    # if it does exist, and youre not resuming, then delete the model dir
    if os.path.exists(saving_dir) and not should_resume:
        # saving dir exists, but we are not resuming, so delete subdirs but keep process outputs from previous attemtpt
        for subfolder in ['model', 'tensorboard', 'other_logs', 'resume_backup']:
            if os.path.exists(saving_dir + '/' + subfolder):
                print(f'deleting: {saving_dir}/{subfolder}')
                shutil.rmtree(saving_dir + '/' + subfolder)
            
    elif os.path.exists(saving_dir) and should_resume:
        pass # resume the existing model
    else:
        # saving dir does not exist, so create it
        os.mkdir(saving_dir)



    if should_resume and os.path.exists(saving_dir + '/model/'):
        pass # Already created on prev run that was aborted
    else:
        # first attempt or fresh start
        os.mkdir(tensorboard_dir)
        os.mkdir(logging_dir)
        os.mkdir(resume_backup_dir)

    return args.config_file_path, saving_dir, config, hyperparameters, already_elapsed_time, \
            tensorboard_dir, logging_dir, model_dir, resume_backup_dir  



def print(*args, **kwargs):
    """
    enables writing to file and console in real time
    """
    kwargs['flush'] = True
    builtins.print(*args, **kwargs)

        
def print_status_update(active_experiments, gpu_launch_times, gpu_resume_times, available_GPUs, configs_to_retry, num_GPUs, 
                GPU_LAUNCH_DELAY, CONFIG_FILE_DIR):
    # print current data and time
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    for gpu_index in range(num_GPUs):
        # delay from it being killed
        if gpu_index in gpu_resume_times.keys() and \
             gpu_resume_times[gpu_index] * 60**2  - time.time() > 0:
            print('GPU ', gpu_index, ' is being left open for {:.2f}'.format(
                  ((60 ** 2 * gpu_resume_times[gpu_index]) - time.time()) / 60), ' minutes due to previous process kill')
        # delay from another process launching on this GPU
        if gpu_index in gpu_launch_times.keys() and \
                time.time() - gpu_launch_times[gpu_index] < GPU_LAUNCH_DELAY:
            print('GPU ', gpu_index, ' is being left open for {:.2f}'.format(
                (GPU_LAUNCH_DELAY - (time.time() - gpu_launch_times[gpu_index])) / 60), ' minutes (due to previous process launch)')
    print('available GPUs: ', available_GPUs)
    print('available RAM: {:.2f}'.format(psutil.virtual_memory().available / 1024 ** 3), 'GB')
     
    # print which configs are training on which GPUs
    for index in range(num_GPUs):
        if index in [e.gpu_index for e in active_experiments.values()]:
            print('GPU ', index, ' is training ')
            gpu_indices = {e.config_file_path: e.gpu_index for e in active_experiments.values()}
            for config_file, gpu_index in gpu_indices.items():
                if gpu_index == index:
                    print('\t', config_file)
    print('configs to retry:')
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


def launch_training(gpu_index, saving_dir, config_file_path, stdout_path):
    """
    Commence training of new model
    """
    with open(config_file_path, "r") as stream:
        config = yaml.safe_load(stream)

    if not os.path.exists(config_file_path):
        raise Exception("Config file doesnt exist", config_file_path)
    process = subprocess.Popen(["ipython", config['train_script_path'], str(gpu_index), str(saving_dir), config_file_path],
                          stdout=subprocess.PIPE)     
    
    
    # launch a thread for monitoring 

    def read_process_output():
        # Read the stdout line by line
        
        # find existing process_outputs in the directory
        process_output_files = [s for s in os.listdir(stdout_path) if s.endswith(".txt") and 'process_output' in s]
        if len(process_output_files) == 0:
            process_output_number = 1
        else:
            no_extensions = [name.split('.')[0] for name in process_output_files]
            # ignore the ones that don't have a _number appended, because they are from the old version
            new_version_no_extensions = list(filter(lambda x: x.count("_") == 2, no_extensions))
            process_output_number = np.max([int(name.split('_')[-1]) for name in new_version_no_extensions]) + 1

        with codecs.open(stdout_path + f'/process_output_{process_output_number}.txt', 'a', 'utf-8') as f: 
        # with open(stdout_path + f'/process_output_{process_output_number}.txt', 'a') as f:
            while True:
                line = process.stdout.readline()
                if not line:
                    # The process has exited, so shut down the thread
                    break       
                f.write(line.decode('utf-8'))  
                # f.write(line.decode())
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




    return TrainingProcess(config_file_path=config_file_path, process=process, config= config,
                           gpu_index=gpu_index )
            

def check_for_kill_flag(request_queue, response_queue, active_experiments, default_delay_time):

    # # Get home directory
    # home = os.path.expanduser("~")
    # flags = os.listdir(home + '/stop_training')
    # # clear flag
    # with open(home + '/stop_training/stop', 'r+') as f:
    #     line = f.readline()
    #     if len(line) == 0:
    #         return None, None
    #     gpu_index, delay_time_h = line.split(' ')
    #     if len(delay_time_h) == 0 or float(delay_time_h) == -1:
    #         delay_time_h = default_delay_time
    #     else:
    #         delay_time_h = float(delay_time_h)
    #     f.truncate(0) #clear file

    # check if request queue is empty
    if request_queue.empty():
        return None, None

    gpu_index, delay_time_h, username =  request_queue.get()
    print('got kill request from', username, 'gpu_index', gpu_index, 'delay_time_h', delay_time_h)
    if delay_time_h is None or float(delay_time_h) == -1:
        delay_time_h = default_delay_time
    if gpu_index is None:
        gpu_index = -1

    # kill a process
    if len(active_experiments) == 0:
        response_queue.put('There are no active experiments')
        return None, None
    if np.sum([not e.immortal for e in active_experiments.values()]) == 0:
        response_queue.put(f'The experiment(s) on GPU{gpu_index} are high priority. Please ask before killing.')
        return None, None
    
    process_keys = list(active_experiments.keys())
    gpu_indices = [e.gpu_index for e in active_experiments.values()]
    if int(gpu_index) == -1:
        unlucky_experiment = active_experiments[process_keys[0]]
        # iterate through active experiments and find the one that was started most recently
        for key in process_keys:
            if active_experiments[key].immortal:
                continue # dont kill immortal processes
            if active_experiments[key].start_time > unlucky_experiment.start_time:
                unlucky_experiment = active_experiments[key]

        unlucky_experiment.process.kill()
        response_queue.put(f'Killed process {unlucky_experiment.name}. I hope youre happy.')
        cleared_gpu_index = unlucky_experiment.gpu_index
        return cleared_gpu_index, delay_time_h    # return the gpu index of the unlucky process 
    # check if there are any processes on the specified GPU
    if int(gpu_index) not in gpu_indices:
        response_queue.put(f'There are no experiments on GPU {gpu_index}')
        return None, None
    elif len([e for e in active_experiments.values() if not e.immortal and
              int(e.gpu_index) == int(gpu_index)]) == 0:
        response_queue.put(f'The experiment(s) on GPU{gpu_index} are high priority. Please ask before killing.')
        return None, None
    else:
        # pick an abitrary process on the specified GPU
        unlucky_experiment = [e for e in active_experiments.values() if int(e.gpu_index) == int(gpu_index)][0]
        # kill the process on the specified GPU that started most recently
        for key in process_keys:
            if int(active_experiments[key].gpu_index) == int(gpu_index):
                unlucky_experiment = active_experiments[key]
                if active_experiments[key].start_time > unlucky_experiment.start_time:
                    unlucky_experiment = active_experiments[key]
        unlucky_experiment.process.kill()
        response_queue.put(f'Killed process {unlucky_experiment.name} on GPU {gpu_index}!!!')
        cleared_gpu_index = unlucky_experiment.gpu_index
        return cleared_gpu_index, delay_time_h
