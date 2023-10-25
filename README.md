
# What does Cookie Monster do? 
Cookie Monster is a program that is continuously running on a process, allowing you to schedule tasks on a GPU. By inputting configuration files, it launches experiments, monitors them, and manages them throughout their duration on the GPU.

# Installation Guide

### 1. Clone Repo
``` 
git clone https://github.com/henrypinkard/cookie_monster.git
```
### 2. Install Dependencies
```
!pip install nvidia-ml-py3
!pip install numpy
!pip install itables
```
### 3. Add to python path
Add path to the git repo to `.bashrc` or `.zshrc`

# Setup
### 1. Make a `config_files` nested folder in your desired workspace
The necessary file structure of a cookie monster project folder is:
```
config_files
├── abandoned
├── complete
├── pending
├── staging
└── training
```

### 2. Make a `cookie_monster_logs` logging directory somewhere in your home directory 
This folder doesn't need any file structure.

### 3. Update `launch_backend.py` with the config and logging directories.
- Modify `CONFIG_FILE_DIR` to be the path to your `config_file` directory
- Modify `COOKIE_MONSTER_LOGS_DIR` to be the directory where the output logs are going to be. 

### 4. Set up GPU Scheduler and Server
Within `template_files` there is a `personstopit` file. This allows for multi-user control of the GPU scheduler. 
- Rename `personstopit` to your desired name, `****stopit`
- Within `****stopit`, replace the url port with a port that other users do not use.
- Replace the port in the `run_server` function in `cookie_monster_backend_lib.py` with the same port number
- Move `****stopit` to `/usr/local/bin` folder
``` 
sudo mv path_to_****stopit /usr/local/bin
```
- Change permissions for all users
```
sudo chmod +x /usr/local/bin/****stopit
```

# General use
Again, the necessary file structure of a cookie monster project folder is:
```
config_files
├── abandoned
├── complete
├── pending
├── staging
└── training

```
- `Staging` is a folder that you can store things you want nearby, but that aren't in the scheduler's view.
- The config files you want to run experiments for are put into the `pending` folder. 
- The scheduler pulls config files into the `training` folder.
- When an experiment is complete, the config file will be put into the `complete` folder.

### 1. Make sure your `config_files` folder is set up

### 2. Make a .py python script that will be passed into cookie_monster
This python script must follow the template in `template_files/template.py`, with the indicated headers and footers.

### 3. Make a .yaml file that cookie_monster can control
This .yaml file must follow the structure in `template_files/template.yaml`, copied below for clarity.

- Update `saving_dir` to the folder you want to save results to
This folder is where the .yaml file will store results, dictated by the `experiment_name` parameter.
- Update `train_script_path` to the .py script that you made above
- Update  `experiment_name` to the desired subfolder name to be created in `saving_dir`
```
config_file_version: ‘2.3’
saving_dir: /home/hpinkard_waller/models/ #update with your saving directory
train_script_path: /home/hpinkard_waller/GitRepos/microscoBayes/deep_density/train_model.py #update with your training script
metadata:
  experiment_name: template #subfolder in saving_dir where experiment results will be placed.
hyperparameters:
  # put your hyperparameters for training here

options:
  hog_memory: false
  resume_training: false
  immortal: false
```

### 4. Move the .yaml file to the `config_files/pending` folder to be prepared for the cookie_monster
### 5. Run Cookie Monster!
This will open a server in your current terminal. If you prefer, you can open a tmux or screen.
The command follows the structure :
```
python path_to_launch_backend "path_to_config_files_folder_in_quotes" "additional_arguments"
```
For example, 
```zsh
python /home/hpinkard_waller/GitRepos/cookie_monster/launch_backend.py "/home/hpinkard_waller/config_files_cookie_monster/" "/home/hpinkard_waller/models/"
```

### 6. Upon successful completion, check `config_files/complete` to see the .yaml file moved, and check `saving_dir` to see the output results.

### 7. Visualize all run experiments using `frontent_cookie_monster.ipynb`
-Change ```STATUS_DIR``` variable to point to your `config_files` directory.


# For use with jupyterlab

Add to ~/.jupyter

So that the checkpoints don't save in directory with yaml files

c.FileCheckpoints.checkpoint_dir = "/home/hpinkard_waller/jupyter_lab_checkpoints"