import torch
import torch.nn as nn
from TCP.model import TCP
from TCP.config import GlobalConfig
from collections import OrderedDict
from torch.utils.data import DataLoader
from TCP.data import CARLA_Data
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

config = GlobalConfig()
net = TCP(config)
ckpt = torch.load('CKPY_PATH', map_location="cuda")
ckpt = ckpt["state_dict"]
new_state_dict = OrderedDict()
for key, value in ckpt.items():
	new_key = key.replace("model.","")
	new_state_dict[new_key] = value
net.load_state_dict(new_state_dict, strict = False)
net.cuda()
net.eval()

config.val_data = 'tcp_bench2drive-val.npy'
val_set = CARLA_Data(root=config.root_dir_all, data_path=config.val_data, img_aug=False)
val_loader = DataLoader(val_set, batch_size=300, shuffle=False)

# Iterate over the validation set
l2_05 = []
l2_1 = []
l2_15 = []
l2_2 = []
length = val_set.__len__()

with torch.no_grad():
	for index, batch in enumerate(tqdm(val_loader)):
		front_img = batch['front_img'].to('cuda')
		speed = batch['speed'].to(dtype=torch.float32).view(-1,1) / 12.
		target_point = batch['target_point'].to(dtype=torch.float32)
		command = batch['target_command']
		state = torch.cat([speed, target_point, command], 1).to('cuda')
		gt_waypoints = batch['waypoints']

		pred = net(front_img, state, target_point.to('cuda'))
		l2_05.extend(np.linalg.norm(pred['pred_wp'][:, 0].detach().cpu().numpy() - gt_waypoints[:, 0].numpy(), axis=1).tolist())
		l2_1.extend(np.linalg.norm(pred['pred_wp'][:, 1].detach().cpu().numpy() - gt_waypoints[:, 1].numpy(), axis=1).tolist())
		l2_15.extend(np.linalg.norm(pred['pred_wp'][:, 2].detach().cpu().numpy() - gt_waypoints[:, 2].numpy(), axis=1).tolist())
		l2_2.extend(np.linalg.norm(pred['pred_wp'][:, 3].detach().cpu().numpy() - gt_waypoints[:, 3].numpy(), axis=1).tolist())

print((sum(l2_05)/length + sum(l2_1)/length + sum(l2_15)/length + sum(l2_2)/length)/4)
