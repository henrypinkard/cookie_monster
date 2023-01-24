# General used

mkdir config_files_cookie_monster

.
├── complete
├── pending
├── stop_training
├── template.yaml
└── training

drwxrwxr-x 2 hpinkard_waller henry 4096 Jan 22 19:58 complete
drwxrwxr-x 2 hpinkard_waller henry 4096 Jan 22 20:01 pending
drwxrwxrwx 2 hpinkard_waller henry 4096 Nov 11 10:04 stop_training
-rw-rw-r-- 1 hpinkard_waller henry  862 Jan 23 17:55 template.yaml
drwxrwxr-x 2 hpinkard_waller henry 4096 Jan 22 20:01 training


# launch
python /home/hpinkard_waller/GitRepos/cookie_monster/launch_backend.py "/home/hpinkard_waller/GitRepos/microscoBayes/deep_density/train_model.py" "/home/hpinkard_waller/config_files_cookie_monster/" "/home/hpinkard_waller/models/"


# For use with jupyterlab

Add to ~/.jupyter

So that the checkpoints don't save in directory with yaml files

c.FileCheckpoints.checkpoint_dir = "/home/hpinkard_waller/jupyter_lab_checkpoints"