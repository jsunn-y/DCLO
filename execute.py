import argparse
import json
import os
from xmlrpc.client import Boolean
import torch
import sys
from time import gmtime, strftime

from src.optimization import run_dbas

class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        self.log = open(os.path.join(save_dir, 'log.txt'), 'a')
   
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass    

# Script starts here.
parser = argparse.ArgumentParser()
parser.add_argument('--config_file', type=str,
                    required=False, default='',
                    help='config file for experiments')
parser.add_argument('--exp_name', type=str,
                    required=False, default='',
                    help='experiment name (default will be config folder name)')
parser.add_argument('-d', '--device', type=int,
                    required=False, default=0,
                    help='device to run the experiment on')
parser.add_argument('--extract', action='store_true',
                    help='whether to extract the features')

args = parser.parse_args()

# Get JSON config file
config_file = os.path.join(os.getcwd(), 'configs', args.config_file)

# Get experiment name
exp_name = args.exp_name if len(args.exp_name) > 0 else args.config_file[:-5]

# Get save directory
save_dir = os.path.join(os.getcwd(), 'saved', exp_name)

# Create save folder
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

#Redirect output to log file
sys.stdout = Logger()
#sys.stdout = open(os.path.join(save_dir, 'log.txt'), 'w')

print('Config file:\t {}'.format(config_file))
print('Save directory:\t {}'.format(save_dir))

# Get device ID
if torch.cuda.is_available() and args.device >= 0:
    assert args.device < torch.cuda.device_count()
    device = 'cuda:{:d}'.format(args.device)
else:
    device = 'cpu'
print('Device:\t {}'.format(device))

# Load JSON config file
with open(config_file, 'r') as f:
    config = json.load(f)

#save the config file
with open(os.path.join(save_dir, args.config_file), 'w') as f:
    json.dump(config, f, indent=4)

# Start training
run_dbas(
    save_path=save_dir,
    data_config=config['data_config'],
    vae_model_config=config['vae_model_config'],
    vae_train_config=config['vae_train_config'],
    opt_config=config['opt_config'],
    device=device,
)
