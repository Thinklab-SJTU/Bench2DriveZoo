import os
import json
import datetime
import pathlib
import time
import cv2
import carla
from collections import deque
import math
from collections import OrderedDict

import torch
import carla
import numpy as np
from PIL import Image
from torchvision import transforms as T

from leaderboard.autoagents import autonomous_agent

from ADMLP.model import ADMLP
from ADMLP.config import GlobalConfig
from team_code.planner import RoutePlanner
from scipy.optimize import fsolve

SAVE_PATH = os.environ.get('SAVE_PATH', None)
IS_BENCH2DRIVE = os.environ.get('IS_BENCH2DRIVE', None)
PLANNER_TYPE = os.environ.get('PLANNER_TYPE', None)
print('*'*10)
print(PLANNER_TYPE)
print('*'*10)

EARTH_RADIUS_EQUA = 6378137.0

def get_entry_point():
	return 'ADMLPAgent'


class ADMLPAgent(autonomous_agent.AutonomousAgent):
	def setup(self, path_to_conf_file):
		self.track = autonomous_agent.Track.SENSORS
		self.data_queue = deque()
		self.data_queue_len = 21 ### 20 Hz!!!
		if IS_BENCH2DRIVE:
			self.save_name = path_to_conf_file.split('+')[-1]
			self.config_path = path_to_conf_file.split('+')[0]
		else:
			self.config_path = path_to_conf_file
			self.save_name = '_'.join(map(lambda x: '%02d' % x, (now.month, now.day, now.hour, now.minute, now.second)))
		self.step = -1
		self.wall_start = time.time()
		self.initialized = False

		self.config = GlobalConfig()
		self.net = ADMLP(self.config)

		ckpt = torch.load(self.config_path, map_location="cuda")
		ckpt = ckpt["state_dict"]
		new_state_dict = OrderedDict()
		for key, value in ckpt.items():
			new_key = key.replace("model.","")
			new_state_dict[new_key] = value
		self.net.load_state_dict(new_state_dict, strict = False)
		self.net.cuda()
		self.net.eval()

		self.save_path = None
		# self.lat_ref, self.lon_ref = 42.0, 2.0
		if SAVE_PATH is not None:
			now = datetime.datetime.now()
			# string = pathlib.Path(os.environ['ROUTES']).stem + '_'
			# string += self.save_name
			string = self.save_name

			print (string)

			self.save_path = pathlib.Path(os.environ['SAVE_PATH']) / string
			self.save_path.mkdir(parents=True, exist_ok=False)

			(self.save_path / 'rgb').mkdir()
			(self.save_path / 'rgb_front').mkdir()
			(self.save_path / 'meta').mkdir()
			(self.save_path / 'bev').mkdir()
	
	def _init(self):
		try:
			locx, locy = self._global_plan_world_coord[0][0].location.x, self._global_plan_world_coord[0][0].location.y
			lon, lat = self._global_plan[0][0]['lon'], self._global_plan[0][0]['lat']
			E = EARTH_RADIUS_EQUA
			def equations(vars):
				x, y = vars
				eq1 = lon * math.cos(x * math.pi / 180) - (locx * x * 180) / (math.pi * E) - math.cos(x * math.pi / 180) * y
				eq2 = math.log(math.tan((lat + 90) * math.pi / 360)) * E * math.cos(x * math.pi / 180) + locy - math.cos(x * math.pi / 180) * E * math.log(math.tan((90 + x) * math.pi / 360))
				return [eq1, eq2]
			initial_guess = [0, 0]
			solution = fsolve(equations, initial_guess)
			self.lat_ref, self.lon_ref = solution[0], solution[1]
		except Exception as e:
			print(e, flush=True)
			self.lat_ref, self.lon_ref = 0, 0
		print(self.lat_ref, self.lon_ref, self.save_name)
		#
		self._route_planner = RoutePlanner(4.0, 50.0, lat_ref=self.lat_ref, lon_ref=self.lon_ref)
		self._route_planner.set_route(self._global_plan, True)
		self.initialized = True
		self.metric_info = {}

	def sensors(self):
		sensors =  [
				{
					'type': 'sensor.camera.rgb',
					'x': 0.80, 'y': 0.0, 'z': 1.60,
					'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
					'width': 1600, 'height': 900, 'fov': 70,
					'id': 'CAM_FRONT'
					},
				# imu
				{
					'type': 'sensor.other.imu',
					'x': -1.4, 'y': 0.0, 'z': 0.0,
					'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
					'sensor_tick': 0.05,
					'id': 'IMU'
					},
				# gps
				{
					'type': 'sensor.other.gnss',
					'x': -1.4, 'y': 0.0, 'z': 0.0,
					'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
					'sensor_tick': 0.01,
					'id': 'GPS'
					},
				# speed
				{
					'type': 'sensor.speedometer',
					'reading_frequency': 20,
					'id': 'SPEED'
					},
				]
		
		if IS_BENCH2DRIVE:
			sensors += [
					{	
						'type': 'sensor.camera.rgb',
						'x': 0.0, 'y': 0.0, 'z': 50.0,
						'roll': 0.0, 'pitch': -90.0, 'yaw': 0.0,
						'width': 512, 'height': 512, 'fov': 5 * 10.0,
						'id': 'bev'
					}]
		return sensors

	def tick(self, input_data):
		self.step += 1
		
		rgb = cv2.cvtColor(input_data['CAM_FRONT'][1][:, :, :3], cv2.COLOR_BGR2RGB)
		bev = cv2.cvtColor(input_data['bev'][1][:, :, :3], cv2.COLOR_BGR2RGB)
		gps = input_data['GPS'][1][:2]
		speed = input_data['SPEED'][1]['speed']
		acceleration = input_data['IMU'][1][:3]
		compass = input_data['IMU'][1][-1]
		if (math.isnan(compass) == True): #It can happen that the compass sends nan for a few frames
			compass = 0.0

		result = {
				'rgb': rgb,
				'gps': gps,
				'speed': speed,
				'compass': compass,
				'bev': bev,
				'acceleration':acceleration,
				}
		pos = self.gps_to_location(result['gps'])
		result['x'] = pos[0]
		result['y'] = pos[1]
		result['theta'] = compass - np.pi/2
		next_wp, next_cmd = self._route_planner.run_step(pos)
		command = next_cmd.value
		result['next_command'] = next_cmd.value
		result['x_target'] = next_wp[0]
		result['y_target'] = next_wp[1]
		return result
	
	@torch.no_grad()
	def run_step(self, input_data, timestamp):
		if not self.initialized:
			self._init()
		tick_data = self.tick(input_data)

		if len(self.data_queue) >= self.data_queue_len:
			self.data_queue.popleft()
		self.data_queue.append(tick_data)
		
		if self.step < self.data_queue_len:
			control = carla.VehicleControl()
			control.steer = 0.0
			control.throttle = 0.0
			control.brake = 0.0
			self.last_control = control
			return control
		
		if (self.step -  self.data_queue_len) % 10 != 0:
			return self.last_control
		
		# Process
		ego_x = self.data_queue[-1]['x']
		ego_y = self.data_queue[-1]['y']
		ego_theta = self.data_queue[-1]['theta']
		next_wp_x = self.data_queue[-1]['x_target']
		next_wp_y = self.data_queue[-1]['y_target']
		speed = self.data_queue[-1]['speed']
		acceleration = self.data_queue[-1]['acceleration']
		command =  self.data_queue[-1]['next_command']

		waypoints = []
		theta = []
		print(len(self.data_queue), self.step)
		for index in range(len(self.data_queue)-1, len(self.data_queue)-1-20-1, -5):
			print(index)
			R = np.array([
			[np.cos(ego_theta), np.sin(ego_theta)],
			[-np.sin(ego_theta),  np.cos(ego_theta)]
			])
			local_command_point = np.array([self.data_queue[index]['x']-ego_x, self.data_queue[index]['y']-ego_y])
			local_command_point = R.dot(local_command_point) # left hand
			waypoints.append([local_command_point[0], local_command_point[1]])
			theta.append(self.data_queue[index]['theta']-ego_theta)

		waypoints = waypoints[::-1]
		theta = theta[::-1]
		R = np.array([
			[np.cos(ego_theta), np.sin(ego_theta)],
			[-np.sin(ego_theta),  np.cos(ego_theta)]
			])

		local_command_point = np.array([next_wp_x-ego_x, next_wp_y-ego_y])
		local_command_point = R.dot(local_command_point)
		target_point = tuple(local_command_point)
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
		command = torch.tensor(cmd_one_hot)
		result = {}
		try:
			result['input'] = torch.cat((torch.tensor(waypoints[:-1]).flatten(), torch.tensor(theta[:-1]), torch.tensor([speed]), torch.tensor(acceleration), command))
		except:
			import pdb;pdb.set_trace()
		result['input'] = result['input'].unsqueeze(0).to('cuda', dtype=torch.float32)
		
		# if self.step % 10 != 0:
		# 	return self.last_control
		pred=self.net(result)
		steer_traj, throttle_traj, brake_traj, metadata_traj = self.net.control_pid(pred, speed, target_point)
		# if brake_traj < 0.05: brake_traj = 0.0
		# if throttle_traj > brake_traj: brake_traj = 0.0

		self.pid_metadata = metadata_traj
		self.pid_metadata['agent'] = 'only_traj'
		control = carla.VehicleControl()
		control.steer = np.clip(float(steer_traj), -1, 1)
		control.throttle = np.clip(float(throttle_traj), 0, 0.75)
		control.brake = np.clip(float(brake_traj), 0, 1)
		self.pid_metadata['steer'] = control.steer
		self.pid_metadata['throttle'] = control.throttle
		self.pid_metadata['brake'] = control.brake

		self.pid_metadata['steer_traj'] = float(steer_traj)
		self.pid_metadata['throttle_traj'] = float(throttle_traj)
		self.pid_metadata['brake_traj'] = float(brake_traj)

		if abs(control.steer) > 0.07: ## In turning
			speed_threshold = 2.5 ## Avoid stuck during turning
		else:
			speed_threshold = 3.0 ## Avoid pass stop/red light/collision
		if float(tick_data['speed']) > speed_threshold:
			max_throttle = 0.05
		else:
			max_throttle = 0.75
		control.throttle = np.clip(control.throttle, a_min=0.0, a_max=max_throttle)

		if control.brake > 0:
			control.brake = 1.0
		if control.brake > 0.5:
			control.throttle = float(0)
		
		metric_info = self.get_metric_info()
		self.metric_info[self.step] = metric_info
		if SAVE_PATH is not None and (self.step -  self.data_queue_len) % 1 == 0:
			self.save(tick_data)
		self.last_control = control
		return control

	def save(self, tick_data):
		frame = self.step

		Image.fromarray(tick_data['rgb']).save(self.save_path / 'rgb' / ('%04d.png' % frame))

		Image.fromarray(tick_data['bev']).save(self.save_path / 'bev' / ('%04d.png' % frame))

		outfile = open(self.save_path / 'meta' / ('%04d.json' % frame), 'w')
		json.dump(self.pid_metadata, outfile, indent=4)
		outfile.close()

		# metric info
		outfile = open(self.save_path / 'metric_info.json', 'w')
		json.dump(self.metric_info, outfile, indent=4)
		outfile.close()

	def destroy(self):
		del self.net
		torch.cuda.empty_cache()

	def gps_to_location(self, gps):
		# gps content: numpy array: [lat, lon, alt]
		lat, lon = gps
		scale = math.cos(self.lat_ref * math.pi / 180.0)
		my = math.log(math.tan((lat+90) * math.pi / 360.0)) * (EARTH_RADIUS_EQUA * scale)
		mx = (lon * (math.pi * EARTH_RADIUS_EQUA * scale)) / 180.0
		y = scale * EARTH_RADIUS_EQUA * math.log(math.tan((90.0 + self.lat_ref) * math.pi / 360.0)) - my
		x = mx - scale * self.lon_ref * math.pi * EARTH_RADIUS_EQUA / 180.0
		return np.array([x, y])