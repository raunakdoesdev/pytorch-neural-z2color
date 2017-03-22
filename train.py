from __future__ import print_function
from nets.simple_net import SimpleNet
from import_utils import *
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy.random as random
import shelve
import sys


filename='/home/schowdhuri/working-directory/pytorch_neural/tmp/shelve.out'

print('Getting Shelved Data')
my_shelf = shelve.open(filename)

print('Loading run codes...')
load_run_codes()
print('Done loading run codes!')

# Load run data

print('Loading run data...')
pb = ProgressBar(1 + len(Segment_Data['run_codes']))
ctr = 0
for n in Segment_Data['run_codes'].keys():
	ctr+=1
	pb.animate(ctr)
	load_run_data(n)
pb.animate(len(Segment_Data['run_codes']))
print('\nFinished loading run data!')

if len(my_shelf) == 0 or 'reshelve' in sys.argv:
    
    # Load steering parameters
    print('Loading low_steer... (takes awhile)')
    low_steer = load_obj(join(hdf5_segment_metadata_path , 'low_steer'))
    
    print('Loading high steer... (takes awhile)')
    high_steer  = load_obj(join(hdf5_segment_metadata_path , 'high_steer'))
    print('Finished high steer data!')
    
    my_shelf = shelve.open(filename,'n') # 'n' for new
    
    my_shelf['low_steer'] = low_steer
    my_shelf['high_steer'] = high_steer
    
    my_shelf.close()
else:
    print('Loading shelved steer_data: ')
    shelfbar = ProgressBar(2)
    shelfbar.animate(0)
    for key in my_shelf:
        
        low_steer = my_shelf['low_steer']
        shelfbar.animate(1)
        high_steer = my_shelf['high_steer']
        shelfbar.animate(2)
    my_shelf.close()

# Update counters and initialize lens to maintain position in segment lists and when to reshuffle.
len_high_steer = len(high_steer)
len_low_steer = len(low_steer)
ctr_low = -1
ctr_high = -1

N_FRAMES = 2 # how many timesteps with images.
N_STEPS = 10 # how many timestamps with non-image data
ignore=['reject_run','left','out1_in2'] # runs with these labels are ignored
require_one=[] # at least one of this type of run lable is required
print_timer = Timer(5)
loss10000 = []
loss = []
rate_timer_interval = 10.
rate_timer = Timer(rate_timer_interval)
rate_ctr = 0

# Initialize Neural Net
print('Nerual net structure:')
net = SimpleNet()
print (net)
criterion = nn.MSELoss()  # define loss function
optimizer = torch.optim.SGD(net.parameters(), lr=0.005, momentum=0.0001)

print('Beginning Training...')
while True:
    if ctr_low >= len_low_steer:
        ctr_low = -1
    if ctr_high >= len_high_steer:
        ctr_high = -1
    if ctr_low == -1:
        random.shuffle(low_steer) # shuffle data before using (again)
        ctr_low = 0
    if ctr_high == -1:
        random.shuffle(high_steer)
        ctr_high = 0
        
    if random.random() > 0.5: # with some probability choose a low_steer element
        choice = low_steer[ctr_low]
        ctr_low += 1
    else:
        choice = high_steer[ctr_high]
        ctr_high += 1

    run_code = choice[3]
    seg_num = choice[0]
    offset = choice[1]
    data = get_data(run_code,seg_num,offset,N_STEPS,offset+0,N_FRAMES,ignore=ignore,require_one=require_one)

    if data == None: # if an ignore flag was found in the data, skip
        continue

    neural_input = None
    for c in range(3):
        for camera in ('left','right'):
            for t in range(N_FRAMES):
                if neural_input is None:
                    neural_input = torch.from_numpy(data[camera][t][:,:,c]) # Creates first channel
                else:
                    neural_input = torch.cat((neural_input,
                        torch.from_numpy(data[camera][t][:,:,c])), 2)  # Adds channel

    neural_input = torch.transpose(neural_input,0,2)
    neural_input = torch.transpose(neural_input,1,2)
    print(neural_input)

    break
