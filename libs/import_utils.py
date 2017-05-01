import pickle

import h5py

from utils import *

Segment_Data = {}
hdf5_runs_path = desktop_dir('bair_car_data/hdf5/runs')
hdf5_segment_metadata_path = desktop_dir('bair_car_data/hdf5/segment_metadata')


def load_hdf5(path):
    F = h5py.File(path)
    labels = {}
    Lb = F['labels']
    for k in Lb.keys():
        if Lb[k][0]:
            labels[k] = True
        else:
            labels[k] = False
    S = F['segments']
    return labels, S


def load_obj(name):
    if not name.endswith('.pkl'):
        name += '.pkl'
    with open(name, 'rb') as f:
        return pickle.load(f)


def load_run_codes():
    run_codes = load_obj(join(hdf5_segment_metadata_path, 'run_codes.pkl'))  # Unpickle run codes
    # Segment run codes.
    Segment_Data['run_codes'] = run_codes
    Segment_Data['runs'] = {}
    for n in run_codes.keys():
        run_name = run_codes[n]
        Segment_Data['runs'][run_name] = {}
        Segment_Data['runs'][run_name]['run_code'] = n


def load_run_data(run_code_num):
    run_name = Segment_Data['run_codes'][run_code_num]
    assert (run_name in Segment_Data['runs'])
    labels, segments = load_hdf5(join(hdf5_runs_path, run_name + '.hdf5'))
    high_steer = load_obj(join(hdf5_segment_metadata_path, run_name + '.high_steer.pkl'))
    low_steer = load_obj(join(hdf5_segment_metadata_path, run_name + '.low_steer.pkl'))
    state_hist_list = load_obj(join(hdf5_segment_metadata_path, run_name + '.state_hist_list.pkl'))
    Segment_Data['runs'][run_name]['labels'] = labels
    Segment_Data['runs'][run_name]['segments'] = segments
    Segment_Data['runs'][run_name]['high_steer'] = high_steer
    Segment_Data['runs'][run_name]['low_steer'] = low_steer
    Segment_Data['runs'][run_name]['state_hist_list'] = state_hist_list
    return run_name


def get_data(run_code_num, seg_num, offset, slen, img_offset, img_slen, ignore=[], require_one=[], smooth_steer=True):
    """
    def get_data(run_code_num,seg_num,offset,slen,img_offset,img_slen,ignore=[left,out1_in2],require_one=[],smooth_steer=True):

    This is the function that delivers segment data to load into Caffe. A run, segment and offset from segement beginning are
    specified. If there are insufficient data following the offset, None is returned.
    """
    run_name = Segment_Data['run_codes'][run_code_num]
    labels = Segment_Data['runs'][run_name]['labels']
    for ig in ignore:
        if labels[ig]:
            return None
    require_one_okay = True
    if len(require_one) > 0:
        require_one_okay = False
        for ro in require_one:
            if labels[ro]:
                require_one_okay = True
    if not require_one_okay:
        return None
    a = offset
    b = offset + slen
    ia = img_offset
    ib = img_offset + img_slen
    seg_num_str = str(seg_num)
    if not (b - a <= len(Segment_Data['runs'][run_name]['segments'][seg_num_str]['steer'][:])):
        return None
    if not (ib - ia <= len(Segment_Data['runs'][run_name]['segments'][seg_num_str]['steer'][:])):
        return None
    steers = Segment_Data['runs'][run_name]['segments'][seg_num_str]['steer'][a:b]
    if len(steers) != slen:
        return None
    motors = Segment_Data['runs'][run_name]['segments'][seg_num_str]['motor'][a:b]
    if len(motors) != slen:
        return None
    states = Segment_Data['runs'][run_name]['segments'][str(seg_num)]['state'][a:b]
    if len(states) != slen:
        return None
    left_images = Segment_Data['runs'][run_name]['segments'][seg_num_str]['left'][ia:ib]
    right_images = Segment_Data['runs'][run_name]['segments'][seg_num_str]['right'][ia:ib]
    if smooth_steer:
        for i in range(2, len(steers)):
            steers[i] = (3 / 6.) * steers[i] + (2 / 6.) * steers[i - 1] + (1 / 6.) * steers[i - 2]

    data = {
        'name': run_name,
        'steer': steers,
        'motor': motors,
        'states': states,
        'left': left_images,
        'right': right_images,
        'labels': labels
    }

    if motors[-1] > 40 and motors[-1] < 60:
        return None
    return data
