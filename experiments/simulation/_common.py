import os, sys
os.environ.setdefault('OMP_NUM_THREADS', '2')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
import torch
torch.set_num_threads(int(os.environ['OMP_NUM_THREADS']))
