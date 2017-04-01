from __future__ import print_function

import argparse

import torch
import torch.nn as nn
from torch.autograd import Variable

from import_utils import *
from nets.simple_net import Z2Color

# Define Arguements and Default Values
parser = argparse.ArgumentParser(description='PyTorch z2_color Training')
parser.add_argument('--validate', default='', type=str, metavar='PATH',
                    help='path to model for validation')
parser.add_argument('--nframes', default=2, type=int, help='Number of timesteps with images.')
parser.add_argument('--nsteps', default=10, type=int, help='Number of timesteps with non-image data.')
parser.add_argument('--ignore', default=['reject_run', 'left', 'out1_in2'], type=str, nargs='+',
                    help='Runs with these labels are ignored')
parser.add_argument('--require_one', default=[], type=str, nargs='+',
                    help='Mandatory run labels, runs without these labels will be ignored.')
parser.add_argument('--cuda_device', default=0, type=int, help='Cuda GPU ID to use for GPU Acceleration.')
parser.add_argument('--batch-size', default=5, type=int, help='Number of datapoints in a mini-batch for training.')
args = parser.parse_args()


# def validation(tenpercent=True):
#     global len_high_steer, len_low_steer
#     running_loss = 0.0
#     total_runs = 2000
#
#     if tenpercent:
#         total_runs = (len_high_steer / 10) + (len_low_steer / 10)  # Print total runs
#
#     epoch_progress = ProgressBar(total_runs)
#     for batch_epoch in range(total_runs):
#         batch_metadata = torch.DoubleTensor()
#         batch_input = torch.DoubleTensor()
#         batch_labels = torch.DoubleTensor()
#
#         pick_validation_data()
#         while data == None:  # Iterate until valid datapoint is picked
#             pick_validation_data()
#
#         get_camera_data()
#         get_metadata()
#
#         # Get Labels
#         labels = torch.DoubleTensor()
#         steer = torch.from_numpy(data['steer'][-args.nsteps:]) / 99.
#         motor = torch.from_numpy(data['motor'][-args.nsteps:]) / 99.
#         labels = torch.cat((steer, labels), 0)
#         labels = torch.cat((motor, labels), 0)
#
#         # Creates batch
#         batch_input = torch.cat((torch.unsqueeze(neural_input, 0), batch_input), 0)
#         batch_metadata = torch.cat((torch.unsqueeze(metaData, 0), batch_metadata), 0)
#         batch_labels = torch.cat((torch.unsqueeze(labels, 0), batch_labels), 0)
#
#         # Train and Backpropagate on Batch Data
#         outputs = net(Variable(batch_input.cuda().float()), Variable(batch_metadata.cuda().float()))
#         print(outputs)
#      print(batch_labels)
#      loss = criterion(outputs, Variable(batch_labels.cuda().float()))
#      running_loss += loss.data[0]
#      print('Error: ' + str(loss.data[0]))
#      # epoch_progress.animate(batch_epoch + 1)
#  print('Average on Validation Set Loss = ' + str(running_loss / total_runs))
#  return running_loss / total_runs


# def train(low_steer, high_steer, data):
#     num_epochs = 10
#     total_runs = (((len(high_steer) / 10) + (len(low_steer) / 10)) * 9) - 2  # Print total runs
#     running_loss = 0.0
#     total_runs = 0
#     # start_time = timer()
#     start_time = time.time()
#
#     # num_batches_per_epoch = (len_high_steer + len_low_steer) * 9 / (10 * batch_size)
#     num_batches_per_epoch = 1000
#
#     save_dir = time.strftime("save/%m-%d--%H-%M-%S")
#     os.makedirs(save_dir)
#     os.makedirs(save_dir + '/validation')
#
#     for epoch in range(num_epochs):
#         print('Epoch # ' + str(epoch) + ' Progress Bar:')
#         epoch_progress = ProgressBar(num_batches_per_epoch)
#
#         for run in range(num_batches_per_epoch):
#             batch_metadata = torch.DoubleTensor()
#             batch_input = torch.DoubleTensor()
#             batch_labels = torch.DoubleTensor()
#
#             for batch in range(args.batch_size):  # Construct batch
#                 pick_data()
#                 while data == None:  # Iterate until valid datapoint is picked
#                     pick_data()  # TODO: FACTOR REPEAT INTO CODE
#                 get_camera_data()
#                 get_metadata()
#
#                 # get labels
#                 labels = torch.DoubleTensor()
#
#                 steer = torch.from_numpy(data['steer'][-args.nsteps:]) / 99.
#                 motor = torch.from_numpy(data['motor'][-args.nsteps:]) / 99.
#
#                 labels = torch.cat((steer, labels), 0)
#                 labels = torch.cat((motor, labels), 0)
#
#                 # Creates batch
#                 batch_input = torch.cat((torch.unsqueeze(neural_input, 0), batch_input), 0)
#                 batch_metadata = torch.cat((torch.unsqueeze(metaData, 0), batch_metadata), 0)
#                 batch_labels = torch.cat((torch.unsqueeze(labels, 0), batch_labels), 0)
#                 total_runs += 1
#
#             # Train and Backpropagate on Batch Data
#             outputs = net(Variable(batch_input.cuda().float()), Variable(batch_metadata.cuda().float()))
#             loss = criterion(outputs, Variable(batch_labels.cuda().float()))
#             loss.backward()
#             optimizer.step()
#             epoch_progress.animate(run + 1)
#             print('Saving Data:\t')
#
#             if run % print_rate == print_rate - 1:
#                 file_name = save_dir + '/epoch_' + str(epoch) + '-batch_' + str(run + 1)
#                 torch.save({'net': net.state_dict(), 'optimizer': optimizer.state_dict()}, file_name)
#
#         print('Validating Epoch # ' + str(epoch) + ' Training:')
#         cur_steer_choice = 0
#         validation_ctr_low = -1
#         validation_ctr_high = -1
#         val_loss = validation()
#         file_name = save_dir + '/validation/epoch_' + str(epoch) + '--valloss_' + str(val_loss)
#         torch.save(net.state_dict(), file_name)


def load_full_run_data():
    pb = ProgressBar(1 + len(Segment_Data['run_codes']))
    ctr = 0
    for n in Segment_Data['run_codes'].keys():
        ctr += 1
        pb.animate(ctr)
        load_run_data(n)
    pb.animate(len(Segment_Data['run_codes']))


def load_steer_data():
    load_steer_data_progress = ProgressBar(2)
    load_steer_data_progress.animate(0)
    low_steer = load_obj(join(hdf5_segment_metadata_path, 'low_steer'))
    load_steer_data_progress.animate(1)
    high_steer = load_obj(join(hdf5_segment_metadata_path, 'high_steer'))
    load_steer_data_progress.animate(2)
    return low_steer, high_steer


def instantiate_net():
    net = Z2Color().cuda()
    criterion = nn.MSELoss().cuda()  # define loss function
    optimizer = torch.optim.SGD(net.parameters(), lr=0.005, momentum=0.0001)
    return net, criterion, optimizer


@static_vars(ctr_low=0, ctr_high=0, cur_steer_choice=0)
def pick_data(low_steer, high_steer):
    low_bound = len(low_steer) * 9 / 10
    high_bound = len(high_steer) * 9 / 10

    if pick_data.ctr_low > low_bound and pick_data.ctr_high > high_bound:
        return 'DONE'  # Finished processing data

    if pick_data.ctr_low > low_bound:
        pick_data.cur_steer_choice = 1
    if pick_data.ctr_high > high_bound:
        pick_data.cur_steer_choice = 0

    if pick_data.cur_steer_choice == 0:  # with some probability choose a low_steer element
        choice = low_steer[pick_data.ctr_low]
        pick_data.ctr_low += 1
        pick_data.cur_steer_choice = 1
    else:
        choice = high_steer[pick_data.ctr_high]
        pick_data.ctr_high += 1
        pick_data.cur_steer_choice = 0

    run_code = choice[3]
    seg_num = choice[0]
    offset = choice[1]

    return get_data(run_code, seg_num, offset, args.nsteps, offset + 0, args.nframes, ignore=args.ignore,
                    require_one=args.require_one)


# def pick_validation_data():
#     global choice, run_code, seg_num, offset, data, cur_steer_choice
#
#     low_start = len_low_steer * 9 / 10
#     high_start = len_high_steer * 9 / 10
#
#     if validation_ctr_low >= len_low_steer:
#         cur_steer_choice = 1  # Stop using it if it's used up
#     if validation_ctr_high >= len_high_steer:
#         cur_steer_choice = 0  # Stop using it if it's used up
#     if validation_ctr_low == -1:
#         validation_ctr_low = low_start
#     if validation_ctr_high == -1:
#         validation_ctr_high = high_start
#
#     if cur_steer_choice == 0:  # alternate steer choices
#         choice = low_steer[validation_ctr_low]
#         cur_steer_choice = 1
#         validation_ctr_low += 1
#     else:
#         choice = high_steer[validation_ctr_high]
#         cur_steer_choice = 0
#         validation_ctr_high += 1
#
#     run_code = choice[3]
#    seg_num = choice[0]
#    offset = choice[1]
#    data = get_data(run_code, seg_num, offset, args.nsteps, offset + 0, args.nframes, ignore=args.ignore,
#                     require_one=args.require_one)


def get_camera_data(data):
    camera_data = torch.FloatTensor()
    for c in range(3):
        for camera in ('left', 'right'):
            for t in range(args.nframes):
                raw_input_data = torch.from_numpy(data[camera][t][:, :, c]).float()
                camera_data = torch.cat((camera_data, raw_input_data / 255.), 2)  # Adds channel

    # Switch dimensions to match neural net
    camera_data = torch.transpose(camera_data, 0, 2)
    camera_data = torch.transpose(camera_data, 1, 2)

    return camera_data


def get_metadata(data):
    metadata = torch.FloatTensor()
    zero_matrix = torch.FloatTensor(1, 13, 26).zero_()  # Min value matrix
    one_matrix = torch.FloatTensor(1, 13, 26).fill_(1)  # Max value matrix

    for cur_label in ['racing', 'caffe', 'follow', 'direct', 'play', 'furtive']:
        if cur_label == 'caffe':
            if data['states'][0]:
                metadata = torch.cat((one_matrix, metadata), 0)
            else:
                metadata = torch.cat((zero_matrix, metadata), 0)
        else:
            if data['labels'][cur_label]:
                metadata = torch.cat((one_matrix, metadata), 0)
            else:
                metadata = torch.cat((zero_matrix, metadata), 0)

    return metadata


def get_labels(data):
    labels = torch.FloatTensor()

    steer = torch.from_numpy(data['steer'][-args.nsteps:]).float() / 99.
    motor = torch.from_numpy(data['motor'][-args.nsteps:]).float() / 99.

    labels = torch.cat((steer, labels), 0)
    labels = torch.cat((motor, labels), 0)

    return labels


def train_batch(batch_size):
    print('Training !!!')
    batch_metadata = torch.FloatTensor()
    batch_input = torch.FloatTensor()
    batch_labels = torch.FloatTensor()

    for batch in range(batch_size):  # Construct batch
        while 'data' not in locals() or data is None:
            data = pick_data(low_steer, high_steer)

        if data == 'DONE':  # If out of data, return done and skip batch
            return 'DONE'

        camera_data = get_camera_data(data)
        metadata = get_metadata(data)
        labels = get_labels(data)

        # Creates batch
        batch_input = torch.cat((torch.unsqueeze(camera_data, 0), batch_input), 0)
        batch_metadata = torch.cat((torch.unsqueeze(metadata, 0), batch_metadata), 0)
        batch_labels = torch.cat((torch.unsqueeze(labels, 0), batch_labels), 0)

    # Train and Backpropagate on Batch Data
    outputs = net(Variable(batch_input.cuda()), Variable(batch_metadata.cuda()))
    loss = criterion(outputs.cuda(), Variable(batch_labels.cuda()))
    loss.backward()
    optimizer.step()
    print(loss.data[0])


torch.set_default_tensor_type('torch.FloatTensor')  # Default tensor to float for consistency
torch.cuda.device(args.cuda_device)  # Cuda device ID

# Load Data
print('Loading run codes')
load_run_codes()
print('Loading run data')
load_full_run_data()
print()
print('Loading steer data')
low_steer, high_steer = load_steer_data()
print()
# Instantiate and Print Neural Net
net, criterion, optimizer = instantiate_net()  # TODO: Load neural net from file
print(net)

print('Training!')

for epoch in range(10):  # Iterate through epochs
    isFinished = 'NOT_DONE'
    while isFinished != 'DONE':
        isFinished = train_batch(args.batch_size)

# cur_steer_choice = 0
# validation_ctr_low = -1
# validation_ctr_high = -1
# net = Z2Color()
# criterion = nn.MSELoss()  # define loss function
# criterion = criterion.cuda()
# loaded = torch.load(args.validate)
# net = net.load_state_dict(loaded['net'])
# net = net.load_state_dict(loaded['optimizer'])
# net = net.cuda()
# validation(False)
