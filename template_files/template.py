"""
Script that trains a single model based on the info in a provided config file
"""
print("~~~~~~~~~~~~~running train script~~~~~~~~~~~~~~")
from cookie_monster_backend_lib import train_script_setup, train_script_complete
config_file_path, saving_dir, config, hyperparameters, already_elapsed_time, \
    tensorboard_dir, logging_dir, model_dir, resume_backup_dir  = train_script_setup()


#######################################################
### Enter your training code here #####################









#######################################################
##### Training complete file flag for scheduler #######
#######################################################
train_script_complete(saving_dir)