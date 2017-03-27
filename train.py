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
import time

# Parameters
batch_size = 20 # Mini batch size
print_rate = 10 # Print every 1000 mini-batches (also the save rate)
num_training_sets = 1000000
torch.cuda.device(0) # Cuda device ID

# Var Initializations
tab_count = 0

def run_job( fun, job_title, debug=True): # Simply runs function and prints out data regarding it
    global tab_count
    if(debug):
        print(tab_count * '\t' + 'Starting ' + job_title + '...' )
        tab_count += 1
    fun()
    if(debug):
        tab_count -= 1
        print(tab_count * '\t' + 'Finshed ' + job_title + '!' )

def get_camera_data():
    global neural_input
    neural_input = torch.DoubleTensor()
    for c in range(3):
        for camera in ('left','right'):
            for t in range(N_FRAMES):
                raw_input_data = torch.from_numpy(data[camera][t][:,:,c]).double()
                neural_input = torch.cat((neural_input, raw_input_data/255.), 2)  # Adds channel

    # Switch dimensions to match neural net
    neural_input = torch.transpose(neural_input,0,2)
    neural_input = torch.transpose(neural_input,1,2)
   
def get_metadata():
    global metaData
    metaData = torch.DoubleTensor()
    zero_matrix = torch.DoubleTensor(1, 13,26).zero_() # Min value matrix
    one_matrix = torch.DoubleTensor(1, 13,26).fill_(1) # Max value matrix
    
    for cur_label in ['racing','caffe','follow','direct', 'play', 'furtive']:
        if cur_label == 'caffe':
            if data['states'][0]:
                metaData = torch.cat((one_matrix, metaData), 0)
            else:
                metaData = torch.cat((zero_matrix, metaData), 0)
        else:
            if data['labels'][cur_label]:
                metaData = torch.cat((one_matrix, metaData), 0)
            else:
                metaData = torch.cat((zero_matrix, metaData), 0)

def validation(tenpercent=True):
    global len_high_steer, len_low_steer
    running_loss = 0.0
    total_runs = 2000

    if tenpercent:
        total_runs = (len_high_steer / 10) + (len_low_steer / 10) # Print total runs

    epoch_progress = ProgressBar(total_runs())
    for batch_epoch in range(total_runs):
        batch_metadata = torch.DoubleTensor()
        batch_input = torch.DoubleTensor()
        batch_labels = torch.DoubleTensor()
        
        pick_validation_data()
        while data == None: # Iterate until valid datapoint is picked
            pick_validation_data()

        get_camera_data()
        get_metadata()
        

        # Get Labels
        labels = torch.DoubleTensor()
        steer = torch.from_numpy(data['steer'][-N_STEPS:])/99.
        motor = torch.from_numpy(data['motor'][-N_STEPS:])/99.
        labels = torch.cat((steer, labels), 0)
        labels = torch.cat((motor, labels), 0)
        
        # Creates batch
        batch_input = torch.cat((torch.unsqueeze(neural_input ,0), batch_input), 0)
        batch_metadata = torch.cat((torch.unsqueeze(metaData, 0), batch_metadata), 0)
        batch_labels = torch.cat((torch.unsqueeze(labels, 0), batch_labels), 0)

        # Train and Backpropagate on Batch Data
        outputs = net(Variable(batch_input.cuda().float()), Variable(batch_metadata.cuda().float()))
        loss = criterion(outputs, Variable(batch_labels.cuda().float())) 
        running_loss += loss.data[0]
        epoch_progress.animate(batch_epoch+1)

    print('Average on Validation Set Loss = ' + str(running_loss/total_runs))
    return running_loss/total_runs


def train():
    global cur_steer_choice, validation_ctr_low, validation_ctr_high
    global len_high_steer, len_low_steer

    num_epochs = 10
    total_runs = (((len_high_steer / 10) + (len_low_steer / 10)) * 9) - 2 # Print total runs
    running_loss = 0.0
    total_runs = 0
    # start_time = timer()
    start_time = time.time()

    num_batches_per_epoch = (len_high_steer + len_low_steer)*9/(10*batch_size)

    save_dir = time.strftime("save/%m-%d--%H-%M-%S")
    os.makedirs(save_dir)
    os.makedirs(save_dir+'/validation')

    for epoch in range(num_epochs):
        print('Epoch # ' + str(epoch) + ' Progress Bar:')
        epoch_progress = ProgressBar(num_batches_per_epoch)

        for run in range(num_batches_per_epoch):
            batch_metadata = torch.DoubleTensor()
            batch_input = torch.DoubleTensor()
            batch_labels = torch.DoubleTensor()
            
            for batch in range(batch_size): # Construct batch
                pick_data()
                while data == None: # Iterate until valid datapoint is picked
                    pick_data() # TODO: FACTOR REPEAT INTO CODE
                get_camera_data()
                get_metadata()

                # get labels
                labels = torch.DoubleTensor()

                steer = torch.from_numpy(data['steer'][-N_STEPS:])/99.
                motor = torch.from_numpy(data['motor'][-N_STEPS:])/99.

                labels = torch.cat((steer, labels), 0)
                labels = torch.cat((motor, labels), 0)
                
                # Creates batch
                batch_input = torch.cat((torch.unsqueeze(neural_input ,0), batch_input), 0)
                batch_metadata = torch.cat((torch.unsqueeze(metaData, 0), batch_metadata), 0)
                batch_labels = torch.cat((torch.unsqueeze(labels, 0), batch_labels), 0)
                total_runs += 1

            # Train and Backpropagate on Batch Data
            outputs = net(Variable(batch_input.cuda().float()), Variable(batch_metadata.cuda().float()))
            loss = criterion(outputs, Variable(batch_labels.cuda().float())) 
            loss.backward()
            optimizer.step()
            epoch_progress.animate(run + 1)

            if run % print_rate == print_rate - 1:
                file_name = save_dir + '/epoch_'+ str(epoch)+'-batch_'+ str(run + 1)
                torch.save(net.state_dict(), file_name)
                

        print ('Validating Epoch # ' + str(epoch) + ' Training:')
        cur_steer_choice = 0
        validation_ctr_low = -1
        validation_ctr_high = -1
        val_loss = validation()
        file_name = save_dir + '/validation/epoch_'+ str(epoch)+'--valloss_'+ str(val_loss) 
        torch.save(net.state_dict(), file_name)


def load_run_data_progress():
    pb = ProgressBar(1 + len(Segment_Data['run_codes']))
    ctr = 0
    for n in Segment_Data['run_codes'].keys():
    	ctr+=1
    	pb.animate(ctr)
    	load_run_data(n)
    pb.animate(len(Segment_Data['run_codes']))
    print()

def load_steer_data():
    global len_high_steer, len_low_steer, low_steer, high_steer
    reload_data = False
    
    filename='/home/schowdhuri/working-directory/pytorch_neural/tmp/shelve.out'
    try:
        my_shelf = shelve.open(filename)
    except:
        reload_data = True
    if reload_data or len(my_shelf) == 0 or '-r' in sys.argv: # Load steer data from pickle and reshelve
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
    print()

def model_init_params():
    global ctr_low, ctr_high, N_FRAMES, N_STEPS, ignore, require_one, print_timer, loss10000, loss
    global cur_steer_choice_training

    ctr_low = -1
    ctr_high = -1
    cur_steer_choice_training = 0
    
    N_FRAMES = 2 # how many timesteps with images.
    N_STEPS = 10 # how many timestamps with non-image data
    ignore=['reject_run','left','out1_in2'] # runs with these labels are ignored
    require_one=[] # at least one of this type of run lable is required
    loss10000 = []
    loss = []

def instantiate_net():
    global net, criterion, optimizer
    net = SimpleNet()
    criterion = nn.MSELoss()  # define loss function
    optimizer = torch.optim.SGD(net.parameters(), lr=0.005, momentum=0.0001)
    net = net.cuda()
    criterion = criterion.cuda()

def print_net():
    print (net)

def pick_validation_data():
    global choice, run_code, seg_num, offset, data, cur_steer_choice
    global validation_ctr_high, validation_ctr_low, N_FRAMES, N_STEPS, ignore, require_one, print_timer, loss10000, loss
    
    low_start = len_low_steer * 9 / 10
    high_start = len_high_steer * 9 / 10

    if validation_ctr_low >= len_low_steer:
        cur_steer_choice = 1 # Stop using it if it's used up
    if validation_ctr_high >= len_high_steer:
        cur_steer_choice = 0 # Stop using it if it's used up
    if validation_ctr_low == -1:
        validation_ctr_low = low_start
    if validation_ctr_high == -1:
        validation_ctr_high = high_start

    if cur_steer_choice == 0: # alternate steer choices
        choice = low_steer[validation_ctr_low]
        cur_steer_choice = 1
        validation_ctr_low += 1
    else:
        choice = high_steer[validation_ctr_high]
        cur_steer_choice = 0
        validation_ctr_high += 1

    run_code = choice[3]
    seg_num = choice[0]
    offset = choice[1]
    data = get_data(run_code, seg_num,offset,N_STEPS,offset+0,N_FRAMES,ignore=ignore,
        require_one=require_one)

def pick_data():
    global choice, run_code, seg_num, offset, data, cur_steer_choice_training
    global ctr_low, ctr_high, N_FRAMES, N_STEPS, ignore, require_one, print_timer, loss10000, loss

    low_bound = len_low_steer * 9 / 10
    high_bound = len_high_steer * 9 / 10

    if ctr_low > low_bound:
        cur_steer_choice_training = 1
    if ctr_high > high_bound:
        cur_steer_choice_training = 0
        ctr_high = -1
    if ctr_low == -1:
        ctr_low = 0
    if ctr_high == -1:
        ctr_high = 0

    if cur_steer_choice_training == 0: # with some probability choose a low_steer element
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

run_job( load_run_codes, 'loading run codes')
run_job( load_run_data_progress, 'loading run data')
run_job( load_steer_data, 'load steer data')
run_job( model_init_params, 'initializing train parameters')
run_job( instantiate_net, 'instantiating neural network')
run_job( print_net, 'printing neural network layers')
run_job( train, 'Training')
