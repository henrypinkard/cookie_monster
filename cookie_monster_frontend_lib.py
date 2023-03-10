import os
import pandas as pd
from itables import init_notebook_mode
import yaml
import time
from datetime import datetime
import warnings

from IPython.display import display, clear_output


import numpy as np
import os
# import tensorboard.backend.event_processing.event_accumulator as ea


def date_format(time_s):
    days = f"{int(time_s // (24 * 60**2))} days  "
    hours, remainder = divmod(time_s, 3600)
    minutes, seconds = divmod(remainder, 60)
    formatted_time = '{}:{:02d}:{:02d}'.format(int(hours), int(minutes), int(seconds))
    return formatted_time

def read_tensorboard_elasped_time(tensorboard_dir):
    event_files = os.listdir(tensorboard_dir + '/train')
    # sort by created date
    event_files.sort(key=lambda x: os.path.getmtime(os.path.join(tensorboard_dir + '/train', x)))
    # get the most recent file
    event_file = event_files[0]

    # print(event_file)
    event_acc = ea.EventAccumulator(tensorboard_dir + '/train/' + event_file)   
    event_acc.Reload()
    # Show all tags in the log file
    a = event_acc.Tensors('epoch_loss')


    wall_times = [event.wall_time for event in a]

    elapsed = np.max(wall_times) - np.min(wall_times)
    return elapsed
    # # convert elapsed time to hours, minutes, seconds
    # days = f"{int(elapsed // (24 * 60**2))} days  "
    # hours, remainder = divmod(elapsed, 3600)
    # minutes, seconds = divmod(remainder, 60)
    # formatted_time = '{}:{:02d}:{:02d}'.format(int(hours), int(minutes), int(seconds))
    # return formatted_time



def create_df(CONFIG_FILE_DIR, sort_columns=('experiment name', 'status', 'date')):

    names = []

    complete = os.listdir(CONFIG_FILE_DIR + 'complete')
    training = os.listdir(CONFIG_FILE_DIR + 'training')
    pending = os.listdir(CONFIG_FILE_DIR + 'pending')
    staging = os.listdir(CONFIG_FILE_DIR + 'staging')
    abandoned = os.listdir(CONFIG_FILE_DIR + 'abandoned')


    config_files = complete + training + pending + staging + abandoned
    statuses = len(complete) * ["complete"] + len(training) * ["training"] + len(pending) * ["pending"] + len(staging) * ["staging"] + len(abandoned) * ["abandoned"] 


    # Read stuff from its config file
    tensorboard_dirs = []
    config_paths = []
    dates = []
    # elapsed_times_tensorboard = []
    elapsed_times_config = []
    attempts = []

    experiment_names = []

    # hyperparameters
    all_marker_training = []
    arch = []
    batch_size = []
    density_output = []
    learning_rate = []
    max_epochs = []
    num_mixture_components = []
    overshoot_epochs = []
    single_marker_early_stopping = []
    single_marker_training = []
    train_priority = []


    for config_file, status in zip(config_files, statuses):
        config_file_path = CONFIG_FILE_DIR + status + '/' + config_file
        with open(config_file_path, "r") as stream:
            config = yaml.safe_load(stream)
            m_time = os.path.getmtime(config_file_path)
            
        if 'training' not in config or 'start_date' not in config['training']:
            dates.append(datetime.fromtimestamp(m_time).strftime("%Y-%m-%d"))
        else:
            dates.append(config['training']['start_date'])
        if 'training' not in config or 'attempt_number' not in config['training']:
            attempts.append('NA')
        else:
            attempts.append(int(config['training']['attempt_number']))

        if 'training' in config:
            if 'tensorboard_dir' not in config['training']:
                tensorboard_dirs.append('NA')
            else:
                tensorboard_dirs.append(config['training']['tensorboard_dir'])
        else:
            tensorboard_dirs.append('NA')
        
        # try:
        #     elapsed_times_tensorboard.append(None if 'training' not in config else read_tensorboard_elasped_time(config['training']['tensorboard_dir']))
        # except:
        #     warnings.warn(f"Can't read elapsed time from tensorboard for {config_file_path}")
        #     elapsed_times_tensorboard.append(None) 
        elapsed_times_config.append(None if 'training' not in config else config['training']['elapsed'])

        
        experiment_names.append(config['experiment_name'])
        # hyperparameters
        all_marker_training.append(config['hyperparameters']['all_marker_training'])
        arch.append(config['hyperparameters']['arch'])
        batch_size.append(config['hyperparameters']['batch_size'])
        density_output.append(config['hyperparameters']['density_output'])
        learning_rate.append(config['hyperparameters']['learning_rate'])
        max_epochs.append(config['hyperparameters']['max_epochs'])
        num_mixture_components.append(config['hyperparameters']['num_mixture_components'])
        overshoot_epochs.append(config['hyperparameters']['overshoot_epochs'])
        single_marker_early_stopping.append(config['hyperparameters']['single_marker_early_stopping'])
        single_marker_training.append(config['hyperparameters']['single_marker_training'])


        # if elapsed_times_tensorboard[-1] is not None and elapsed_times_tensorboard[-1] != 'NA':
        #     elapsed_times_tensorboard[-1] = date_format(elapsed_times_tensorboard[-1])
        if elapsed_times_config[-1] is not None and elapsed_times_config[-1] != 'NA':
            elapsed_times_config[-1] = date_format(elapsed_times_config[-1])


        if status == 'pending':
            pending_configs = [s for s in os.listdir(CONFIG_FILE_DIR + 'pending') if s.endswith(".yaml")]
            pending_configs.sort(key=lambda x: os.path.getctime(os.path.join(CONFIG_FILE_DIR + 'pending', x)))
            train_priority.append(len(pending_configs) - pending_configs.index(config_file))
        else:
            train_priority.append('NA')


        config_paths.append(CONFIG_FILE_DIR.replace(os.path.expanduser('~'), '') + status + '/' + config_file)



    d = {
        "date": dates,  
        "experiment name": experiment_names,
        "name": [cf.replace('_', " ") for cf in config_files],

        # "all_marker_training": all_marker_training,
        # "arch": arch,
        # "batch size": batch_size,
        # "density output": density_output,
        # "learning rate": learning_rate,
        # "max epochs": max_epochs,
        # "num mixture components": num_mixture_components,
        # "overshoot_epochs": overshoot_epochs,
        # "single marker early stopping": single_marker_early_stopping,
        # "single marker training": single_marker_training,

        "status": statuses, 
        "train priority": train_priority,

        "attempts": attempts, 
        # "elapsed_time (tensorboard)": elapsed_times_tensorboard,
        "elapsed_time (config)": elapsed_times_config,
        #  "config": config_paths, 
        #  "tensorboard": tensorboard_dirs}
    }
    df = pd.DataFrame(data=d)
    # df.style.set_properties(subset=['config_files'], **{'min-width': '500px'})
    pd.options.display.min_rows = 100



    df['status'] = pd.Categorical(df['status'], ["complete", "training", "pending", "staging", "abandoned"])
    df['date'] = pd.to_datetime(df['date'])
    # dont show time, only show the date
    df['date'] = df['date'].dt.date
    df = df.sort_values(by=list(sort_columns))

    def stylize(x):
        if x == 'complete':
            return 'color: green'
        elif x == 'training':
            return 'color: orange'
        elif x == 'pending':
            return 'color: blue'
        elif x == 'staging':
            return 'color: lightblue'
        elif x == 'abandoned':
            return 'color: gray'
        elif x == 'NA':
            return 'color: gray'
        return None
        

    df = df.style.applymap(stylize)  

        
    return df
