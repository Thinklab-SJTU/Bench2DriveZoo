import numpy as np
import torch 
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

class CARLA_Data(Dataset):
	def __init__(self, data_path):
		self.x = []
		self.y = []
		self.theta = []
		self.command = []
		self.speed = []
		self.speed_acc = []

		self.future_x = []
		self.future_y = []
		self.future_theta = []

		data = np.load(data_path, allow_pickle=True).item()
		self.x += data['input_x']
		self.y += data['input_y']
		self.theta += data['input_theta']
		self.command += data['input_command']
		self.speed += data['input_speed']
		self.speed_acc += data['input_speed_acc']

		self.future_x += data['future_x']
		self.future_y += data['future_y']
		self.future_theta += data['future_theta']

	def __getitem__(self, index):
		data = dict()
		data['x'] = self.x[index]
		data['y'] = self.y[index]
		data['theta'] = self.theta[index]
		data['speed'] = self.speed[index][-1]
		data['speed_acc'] = self.speed_acc[index][-1]
		data['future_x'] = self.future_x[index]
		data['future_y'] = self.future_y[index]
		data['future_theta'] = self.future_theta[index]

		ego_x = self.x[index][-1]
		ego_y = self.y[index][-1]
		if np.isnan(self.theta[index][-1]):
			ego_theta = 0
		else:
			ego_theta = self.theta[index][-1] - np.pi/2 # compass on left hand (0, -1)
		
		waypoints = []
		for i in range(6):
			R = np.array([
			[np.cos(ego_theta), np.sin(ego_theta)],
			[-np.sin(ego_theta),  np.cos(ego_theta)]
			])
			local_command_point = np.array([self.future_x[index][i]-ego_x, self.future_y[index][i]-ego_y])
			local_command_point = R.dot(local_command_point) # left hand
			waypoints.append([local_command_point[0], local_command_point[1]])
		
		thetas = [(t - np.pi/2) - ego_theta for t in self.future_theta[index]]
		thetas = [0 if np.isnan(t) else t for t in thetas]
		data['waypoints'] = torch.tensor(waypoints)
		data['thetas'] = torch.tensor(thetas)

		hist_waypoints = []
		for i in range(4):
			R = np.array([
			[np.cos(ego_theta), np.sin(ego_theta)],
			[-np.sin(ego_theta),  np.cos(ego_theta)]
			])
			local_command_point = np.array([self.x[index][i]-ego_x, self.y[index][i]-ego_y])
			local_command_point = R.dot(local_command_point) # left hand
			hist_waypoints.append([local_command_point[0], local_command_point[1]])
		
		hist_thetas = [(t - np.pi/2) - ego_theta for t in self.theta[index][:-1]]
		hist_thetas = [0 if np.isnan(t) else t for t in hist_thetas]
		data['hist_waypoints'] = torch.tensor(hist_waypoints).flatten()
		data['hist_thetas'] = torch.tensor(hist_thetas)
		data['speed'] = torch.tensor([data['speed']])
		data['speed_acc'] = torch.tensor(data['speed_acc'])

		debug_plot_local = False
		if debug_plot_local:
			plt.figure(figsize=(10, 10))
			points = hist_waypoints + [[0, 0]] + waypoints
			x, y = zip(*points)
			plt.scatter(x, y, color='red', zorder=5)
			for i, (px, py) in enumerate(points):
				plt.text(px, py, f'P{i}', fontsize=12, ha='right' if i % 2 == 0 else 'left', va='top' if i % 2 == 0 else 'bottom')
			plt.plot(x, y, 'b--')
			plt.xlabel('X coordinate')
			plt.ylabel('Y coordinate')
			plt.grid(True)
			plt.axis('equal') 
			plt.savefig(f"{index}_local.png")
		
		command = self.command[index][-1]
		# VOID = -1
		# LEFT = 1
		# RIGHT = 2
		# STRAIGHT = 3
		# LANEFOLLOW = 4
		# CHANGELANELEFT = 5
		# CHANGELANERIGHT = 6
		if command < 0:
			command = 4
		command -= 1
		assert command in [0, 1, 2, 3, 4, 5]
		cmd_one_hot = [0] * 6
		cmd_one_hot[command] = 1
		data['command'] = torch.tensor(cmd_one_hot)	
		data['input'] = torch.cat((data['hist_waypoints'], data['hist_thetas'], data['speed'], data['speed_acc'], data['command']))
		return data

	def __len__(self):
		print(len(self.x))
		return len(self.x)