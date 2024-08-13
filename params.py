import torch


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

LOG_B = '\033[1;34m'
LOG_Y = '\033[1;33m'
LOG_G = '\033[1;36m'
LOG_END = '\033[m'

