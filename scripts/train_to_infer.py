import sys
import torch

filename = sys.argv[1]
weights = torch.load(filename)

for key in weights['net']:
    weights['net'][key] = weights['net'][key].cuda(device=0)

newweights = dict()
newweights['net'] = weights['net']

torch.save(newweights, filename + ".infer")
