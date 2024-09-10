import torch
import torch.nn as nn
from ADMLP.model import ADMLP
from ADMLP.config import GlobalConfig
from collections import OrderedDict
from torch.utils.data import DataLoader
from ADMLP.data import CARLA_Data
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

config = GlobalConfig()
net = ADMLP(config)
ckpt = torch.load('admlp_b2d.ckpt', map_location="cuda")
ckpt = ckpt["state_dict"]
new_state_dict = OrderedDict()
for key, value in ckpt.items():
	new_key = key.replace("model.","")
	new_state_dict[new_key] = value
net.load_state_dict(new_state_dict, strict = False)
net.cuda()
net.eval()

config.val_data = 'admlp_bench2drive-val.npy'
val_set = CARLA_Data(data_path=config.val_data)
val_loader = DataLoader(val_set, batch_size=300, shuffle=False)

# Iterate over the validation set
l2_05 = []
l2_1 = []
l2_15 = []
l2_2 = []
length = val_set.__len__()

with torch.no_grad():
	for index, batch in enumerate(tqdm(val_loader)):
		batch['input'] = batch['input'].to('cuda')
		predict = net(batch)
		waypoints = batch['waypoints']
		theta = batch['thetas']
		l2_05.extend(np.linalg.norm(predict[:, 0, :2].detach().cpu().numpy() - waypoints[:, 0, :2].numpy(), axis=1).tolist())
		l2_1.extend(np.linalg.norm(predict[:, 1, :2].detach().cpu().numpy() - waypoints[:, 1, :2].numpy(), axis=1).tolist())
		l2_15.extend(np.linalg.norm(predict[:, 2, :2].detach().cpu().numpy() - waypoints[:, 2, :2].numpy(), axis=1).tolist())
		l2_2.extend(np.linalg.norm(predict[:, 3, :2].detach().cpu().numpy() - waypoints[:, 3, :2].numpy(), axis=1).tolist())

print((sum(l2_05)/length + sum(l2_1)/length + sum(l2_15)/length + sum(l2_2)/length)/4)
