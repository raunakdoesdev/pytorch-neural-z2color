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

tab_count = 0
batch_size = 5 # Mini batch size
print_rate = 1000 # Print every 1000 mini-batches
num_training_sets = 10000 # Number of mini-batches

def run_job( fun, job_title): # Simply runs function and prints out data regarding it
    global tab_count
    print(tab_count * '\t' + 'Starting ' + job_title + '...' )
    tab_count += 1
    fun()
    tab_count -= 1
    print(tab_count * '\t' + 'Finshed ' + job_title + '!' )

def get_camera_data():
    global neural_input
    neural_input = torch.ByteTensor()
    for c in range(3):
        for camera in ('left','right'):
            for t in range(N_FRAMES):
                neural_input = torch.cat((neural_input, 
                    torch.from_numpy(data[camera][t][:,:,c])), 2)  # Adds channel

    # Switch dimensions to match neural net
    neural_input = torch.transpose(neural_input,0,2)
    neural_input = torch.transpose(neural_input,1,2)
    
def get_metadata():
    global metaData
    metaData = torch.ByteTensor()
    zero_matrix = torch.ByteTensor(1, 14,26).zero_() # Min value matrix
    one_matrix = torch.ByteTensor(1, 14,26).fill_(255) # Max value matrix
    
    for cur_label in ['racing',0,'follow','direct', 'play', 'furtive']:
        if data['labels'][cur_label]:
            metaData = torch.cat((one_matrix, metaData), 0)
        else:
            metaData = torch.cat((zero_matrix, metaData), 0)

def train():
    running_loss = 0.0
    total_runs = 0
    for batch_epoch in range(num_training_sets/batch_size):
        batch_metadata = torch.ByteTensor()
        batch_input = torch.ByteTensor()

        for batch in range(batch_size): # Construct batch
            run_job(pick_data, 'datapoint extraction')
            if data == None: # if an ignore flag was found in the data, skip
                print ('Skipping datapoint due to ignore flag')
                continue

            run_job(get_camera_data, "camera data extraction")
            run_job(get_metadata, "metadata extraction")

            # Creates batch
            batch_input = torch.cat((torch.unsqueeze(neural_input,0), batch_input), 0)
            batch_metadata = torch.cat((torch.unsqueeze(metaData, 0), batch_metadata), 0)

            total_runs += 1

        # Train and Backpropagate on Batch Data
        outputs = net(Variable(neural_input), Variable(metaData))
        loss = criterion(outputs, labels) # TODO: DEFINE LABELS
        loss.backward()
        optimizer.step()

        running_loss += loss.data[0]

        if i % print_rate  == print_rate - 1: # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' % (batch_epoch + 1, total_runs + 1, 
                running_loss / print_rate))
            running_loss = 0.0


def load_run_data_progress():
    pb = ProgressBar(1 + len(Segment_Data['run_codes']))
    ctr = 0
    for n in Segment_Data['run_codes'].keys():
    	ctr+=1
    	pb.animate(ctr)
    	load_run_data(n)
    pb.animate(len(Segment_Data['run_codes']))

def load_steer_data():
    global len_high_steer, len_low_steer, low_steer, high_steer
    filename='/home/schowdhuri/working-directory/pytorch_neural/tmp/shelve.out'
    my_shelf = shelve.open(filename)
    if len(my_shelf) == 0 or '-r' in sys.argv: # Load steer data from pickle and reshelve
        shelfbar = ProgressBar(4)
        shelfbar.animate(0)
        low_steer = load_obj(join(hdf5_segment_metadata_path , 'low_steer'))
        shelfbar.animate(1)
        high_steer  = load_obj(join(hdf5_segment_metadata_path , 'high_steer'))
        shelfbar.animate(2)
        my_shelf = shelve.open(filename,'n') # 'n' for new
        my_shelf['low_steer'] = low_steer
        shelfbar.animate(3)
        my_shelf['high_steer'] = high_steer
        shelfbar.animate(4)
        my_shelf.close()
    else: # Load steer data from previous shelve (faster)
        shelfbar = ProgressBar(2)
        shelfbar.animate(0)
        low_steer = my_shelf['low_steer']
        shelfbar.animate(1)
        high_steer = my_shelf['high_steer']
        shelfbar.animate(2)
        my_shelf.close()
    len_high_steer = len(high_steer)
    len_low_steer = len(low_steer)

def model_init_params():
    global ctr_low, ctr_high, N_FRAMES, N_STEPS, ignore, require_one, print_timer, loss10000, loss
    global rate_timer_interval, rate_timer, rate_ctr

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

def instantiate_net():
    global net, criterion, optimizer
    net = SimpleNet()
    criterion = nn.MSELoss()  # define loss function
    optimizer = torch.optim.SGD(net.parameters(), lr=0.005, momentum=0.0001)

def print_net():
    print (net)

def pick_data():
    global choice, run_code, seg_num, offset, data

    global ctr_low, ctr_high, N_FRAMES, N_STEPS, ignore, require_one, print_timer, loss10000, loss
    global rate_timer_interval, rate_timer, rate_ctr

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
    data = get_data(run_code, seg_num,offset,N_STEPS,offset+0,N_FRAMES,ignore=ignore,
        require_one=require_one)
# Nice way to see what's happening in the file, and allows easy removal of debug statements
# through the run_job function.
run_job( load_run_codes, 'loading run codes')
run_job( load_run_data_progress, 'loading run data')
run_job( load_steer_data, 'load steer data')
run_job( model_init_params, 'initializing train parameters')
run_job( instantiate_net, 'instantiating neural network')
run_job( print_net, 'printing neural network layers')
run_job( train, 'Training')
