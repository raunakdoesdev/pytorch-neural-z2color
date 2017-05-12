import argparse
import traceback
import datetime
import random

import torch
import torch.nn as nn
import torch.nn.utils as nnutils
from torch.autograd import Variable
from libs.import_utils import *
from nets.squeezenet import SqueezeNet
from nets.z2_color import Z2Color
from nets.z2_color_batchnorm import Z2ColorBatchNorm

# Define Arguments and Default Values
parser = argparse.ArgumentParser(description='PyTorch z2_color Training',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--validate', type=str, metavar='PATH',
                    help='path to model for validation')
# parser.add_argument('--skipfirstval', type=str, metavar='PATH',
#                     help='Skip the first validation (if restoring an end of epoch save file)')
parser.add_argument('--resume', type=str, metavar='PATH',
                    help='path to model for training resume')
parser.add_argument('--ignore', default=['reject_run', 'left', 'out1_in2', 'racing', 'Smyth'], type=str, nargs='+',
                    help='Runs with these labels are ignored')
parser.add_argument('--require-one', default=[], type=str, nargs='+',
                    help='Mandatory run labels, runs without these labels will be ignored.')
parser.add_argument('--cuda-device', default=0, type=int, help='Cuda GPU ID to use for GPU Acceleration.')
parser.add_argument('--batch-size', default=64, type=int, help='Number of datapoints in a mini-batch for training.')
parser.add_argument('--saverate', default=700, type=int,
                    help='Number of batches after which a progress save is done.')
args = parser.parse_args()

start_ctrl_low = 0
start_ctrl_high = 0
if args.resume is not None:
    save_data = torch.load(args.resume)
    start_ctrl_low = save_data['low_ctr']
    start_ctrl_high = save_data['high_ctr']


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

@static_vars(ctr_low=-1, ctr_high=-1)
def pick_validate_data(low_steer_train, high_steer_train, low_steer_val, high_steer_val):
    low_bound = len(low_steer_val)
    high_bound = len(high_steer_val)

    if pick_validate_data.ctr_low == -1 and pick_validate_data.ctr_high == -1:
        pick_validate_data.ctr_low = 0
        pick_validate_data.ctr_high = 0
    if pick_validate_data.ctr_low >= low_bound or pick_validate_data.ctr_high >= high_bound:
        pick_validate_data.ctr_low = 0
        pick_validate_data.ctr_high = 0
        return pick_validate_data.ctr_low + pick_validate_data.ctr_high, 0  # Finished processing data

    if random.random() > 0.5:  # with some probability choose a low_steer element
        choice = low_steer_val[pick_validate_data.ctr_low]
        pick_validate_data.ctr_low += 1
    else:
        choice = high_steer_val[pick_validate_data.ctr_high]
        pick_validate_data.ctr_high += 1

    run_code = choice[3]
    seg_num = choice[0]
    offset = choice[1]

    return (pick_validate_data.ctr_high + pick_validate_data.ctr_low), get_data(run_code, seg_num, offset, net.N_STEPS,
                                                                                offset + 0, net.N_FRAMES,
                                                                                ignore=args.ignore,
                                                                                require_one=args.require_one)


@static_vars(ctr_low=start_ctrl_low, ctr_high=start_ctrl_high)
def pick_data(low_steer_train=None, high_steer_train=None, low_steer_val=None, high_steer_val=None):
    if low_steer_train is None and high_steer_train is None:
        return pick_data.ctr_low, pick_data.ctr_high
    low_bound = len(low_steer_train)
    high_bound = len(high_steer_train)

    if pick_data.ctr_low >= low_bound or pick_data.ctr_high >= high_bound:
        # Reset counters and say you're done
        pick_data.ctr_low = 0
        pick_data.ctr_high = 0
        return pick_data.ctr_low + pick_data.ctr_high, 0  # Finished processing data

    if random.random() > 0.5:  # with some probability choose a low_steer element
        choice = low_steer_train[pick_data.ctr_low]
        pick_data.ctr_low += 1
    else:
        choice = high_steer_train[pick_data.ctr_high]
        pick_data.ctr_high += 1

    run_code = choice[3]
    seg_num = choice[0]
    offset = choice[1]

    return (pick_data.ctr_high + pick_data.ctr_low), get_data(run_code, seg_num, offset, net.N_STEPS, offset + 0,
                                                              net.N_FRAMES, ignore=args.ignore,
                                                              require_one=args.require_one)


def get_camera_data(data):
    camera_data = torch.FloatTensor().cuda()
    for c in range(3):
        for camera in ('left', 'right'):
            for t in range(net.N_FRAMES):
                raw_input_data = torch.from_numpy(data[camera][t][:, :, c]).cuda().float()
                camera_data = torch.cat((camera_data, (raw_input_data.unsqueeze(2) / 255.) - 0.5), 2)  # Adds channel

    # Switch dimensions to match neural net
    camera_data = torch.transpose(camera_data, 0, 2)
    camera_data = torch.transpose(camera_data, 1, 2)

    return camera_data


def get_metadata(data):
    metadata = torch.FloatTensor().cuda()
    zero_matrix = torch.FloatTensor(1, 13, 26).zero_().cuda()  # Min value matrix
    one_matrix = torch.FloatTensor(1, 13, 26).fill_(1).cuda()  # Max value matrix

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

def get_squeeze_metadata(data):
    metadata = torch.FloatTensor().cuda()
    for batch in range(data.size()[0]):
        one_batch_data = torch.FloatTensor().cuda()
        for label in range(data.size()[1]):
            one_batch_data = torch.cat((torch.FloatTensor(1, 1, 13, 26).fill_(data.data[0][label]).cuda(), one_batch_data), 1)
        metadata = torch.cat((metadata, one_batch_data), 0)
    return metadata
    

def get_labels(data):
    steer = torch.from_numpy(data['steer'][-net.N_STEPS:]).cuda().float() / 99.
    motor = torch.from_numpy(data['motor'][-net.N_STEPS:]).cuda().float() / 99.

    return torch.cat((steer, motor), 0)


def get_batch_data(batch_size, data_function):
    batch_metadata = torch.FloatTensor().cuda()
    batch_input = torch.FloatTensor().cuda()
    batch_labels = torch.FloatTensor().cuda()

    for batch in range(batch_size):  # Construct batch
        data = None
        while 'data' not in locals() or data is None:
            progress, data = data_function(low_steer_train, high_steer_train, low_steer_val, high_steer_val)

        if data == 0:  # If out of data, return done and skip batch
            return progress, False, None, None, None

        camera_data = get_camera_data(data)
        metadata = get_metadata(data)
        labels = get_labels(data)

        # Creates batch
        batch_input = torch.cat((torch.unsqueeze(camera_data, 0), batch_input), 0)
        batch_metadata = torch.cat((torch.unsqueeze(metadata, 0), batch_metadata), 0)
        batch_labels = torch.cat((torch.unsqueeze(labels, 0), batch_labels), 0)

    # Train and Backpropagate on Batch Data
    return progress, True, batch_input, batch_metadata, batch_labels


torch.set_default_tensor_type('torch.FloatTensor')  # Default tensor to float for consistency

torch.cuda.set_device(args.cuda_device)  # Cuda device ID

# Load Data
print('Loading run codes')
load_run_codes()
print('Loading run data')
load_full_run_data()
print()
print('Loading steer data')
low_steer, high_steer = load_steer_data()
random.shuffle(low_steer)
random.shuffle(high_steer)
low_steer_train = low_steer[:int(0.9*len(low_steer))]
high_steer_train = high_steer[:int(0.9*len(high_steer))]
low_steer_val = low_steer[int(0.9*len(low_steer)):]
high_steer_val = high_steer[int(0.9*len(high_steer)):]

net = SqueezeNet().cuda()
criterion = nn.MSELoss().cuda()  # define loss function
optimizer = torch.optim.Adam(net.parameters(), lr=net.lr)

drive_net = Z2ColorBatchNorm().cuda()
drive_net.load_state_dict(torch.load('batch_norm_epoch7')['net'])
drive_net.eval() # Set network to evaluate mode

cur_epoch = 0
if args.resume is not None:
    save_data = torch.load(args.resume)
    net.load_state_dict(save_data['net'])
    optimizer.load_state_dict(save_data['optim'])
    cur_epoch = save_data['epoch']
if args.validate is not None:
    save_data = torch.load(args.validate)
    net.load_state_dict(save_data['net'])

    sum = 0
    count = 0
    notFinished = True  # Checks if finished with dataset
    net.eval()
    while notFinished:
        random.shuffle(low_steer_val)
        random.shuffle(high_steer_val)
        # Load batch
        progress, notFinished, batch_input, batch_metadata, batch_labels = get_batch_data(1, pick_validate_data)
        if not notFinished:
            break

        infer_metadata = net(Variable(batch_input))
        outputs = drive_net(Variable(batch_input), Variable(get_squeeze_metadata(infer_metadata))).cuda()

        loss = criterion(outputs, Variable(batch_labels))
        count += 1
        sum += loss.data[0]

        # print('Output:\n' + str(outputs) + '\nLabels:\n' + str(batch_labels))
        print('Average Loss: ' + str(sum / count))
else:
    print(net)
    log_file = open('logs/log_file' + str(datetime.datetime.now().isoformat()), 'w')
    log_file.truncate()
    try:
        for epoch in range(cur_epoch, 10):  # Iterate through epochs
            cur_epoch = epoch
            # Training
            notFinished = True  # Checks if finished with dataset
            random.shuffle(low_steer_train)
            random.shuffle(high_steer_train)
            pb = ProgressBar(len(low_steer_train) + len(high_steer_train))
            batch_counter = 0
            sum = 0
            sum_counter = 0
            start = time.time()
            net.train()
            while notFinished:
                # Load batch
                progress, notFinished, batch_input, batch_metadata, batch_labels = get_batch_data(args.batch_size,
                                                                                                  pick_data)
                if not notFinished:
                    break

                # zero the parameter gradients
                optimizer.zero_grad()

                # Run neural net + Calculate Loss
                infer_metadata = net(Variable(batch_input))
                outputs = drive_net(Variable(batch_input), Variable(get_squeeze_metadata(infer_metadata))).cuda()
                loss = criterion(outputs, Variable(batch_labels))

                # Backprop
                loss.backward()
                # nnutils.clip_grad_norm(net.parameters(), 1.0)
                optimizer.step()

                # Update progress bar
                pb.animate(progress)
                batch_counter += 1
                sum_counter += 1
                sum += loss.data[0]

                if sum_counter == 80:
                    log_file.write(
                        '\n' + str(batch_counter) + ',' + str(sum / sum_counter))
                    log_file.flush()
                    sum = 0
                    sum_counter = 0

                if batch_counter % args.saverate == 0 and batch_counter != 0:
                    low, high = pick_data()
                    save_data = {'low_ctr': low, 'high_ctr': high, 'net': net.state_dict(),
                                 'optim': optimizer.state_dict(), 'epoch': cur_epoch}
                    torch.save(save_data, 'save/progress_save_' + str(epoch) + '-' + str(batch_counter))

            sum = 0
            count = 0
            notFinished = True  # Checks if finished with dataset
            pb = ProgressBar(len(low_steer_val) + len(high_steer_val))
            net.eval()
            while notFinished:
                # Load batch
                progress, notFinished, batch_input, batch_metadata, batch_labels = get_batch_data(1, pick_validate_data)

                if not notFinished:
                    break

                # Run neural net + Calculate Loss
                infer_metadata = net(Variable(batch_input))
                outputs = drive_net(Variable(batch_input), Variable(get_squeeze_metadata(infer_metadata))).cuda()
                loss = criterion(outputs, Variable(batch_labels))
                count += 1
                sum += loss.data[0]

                if count % 1000 == 0:
                    pb.animate(progress)
                    log_file.write('\nAverage Validation Loss,' + str(sum / count))
                    log_file.flush()

            log_file.write('\nFinish cross validation! Average Validation Error = ' + str(sum / count))
            log_file.flush()
            save_data = {'low_ctr': 0, 'high_ctr': 0, 'net': net.state_dict(),
                         'optim': optimizer.state_dict(), 'epoch': cur_epoch}
            torch.save(save_data, 'save/epoch_save_' + str(cur_epoch) + '.' + str(sum / count))
    except Exception as e:  # In case of any exception or error, save the model.
        log_file.write('\nError Recieved while training. Saved model and terminated code:\n' + traceback.format_exc())
        low, high = pick_data()
        save_data = {'low_ctr': low, 'high_ctr': high, 'net': net.state_dict(),
                     'optim': optimizer.state_dict(), 'epoch': cur_epoch}
        torch.save(save_data, 'interrupt_save')
        print('\nError Recieved, Saved model!')
    finally:
        log_file.close()
