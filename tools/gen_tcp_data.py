import os
import json
import numpy as np
from tqdm import trange
import gzip
import multiprocessing as mp 
import time


INPUT_FRAMES = 1
FUTURE_FRAMES = 4*5 # 10hz --> 2hz
TRAIN = True

val_list = [
    'StaticCutIn_Town05_Route226_Weather18',
    'MergerIntoSlowTrafficV2_Town12_Route857_Weather25',
    'YieldToEmergencyVehicle_Town04_Route166_Weather10',
    'ConstructionObstacle_Town10HD_Route74_Weather22',
    'VehicleTurningRoutePedestrian_Town15_Route445_Weather11',
    'VanillaSignalizedTurnEncounterRedLight_Town07_Route359_Weather21',
    'SignalizedJunctionLeftTurnEnterFlow_Town13_Route657_Weather2',
    'LaneChange_Town06_Route307_Weather21',
    'ConstructionObstacleTwoWays_Town12_Route1093_Weather1',
    'HazardAtSideLaneTwoWays_Town12_Route1151_Weather7',
    'OppositeVehicleTakingPriority_Town04_Route214_Weather6',
    'NonSignalizedJunctionRightTurn_Town03_Route126_Weather18',
    'VanillaNonSignalizedTurnEncounterStopsign_Town12_Route979_Weather9',
    'ParkedObstacle_Town06_Route282_Weather22',
    'ControlLoss_Town10HD_Route378_Weather14',
    'ControlLoss_Town04_Route170_Weather14',
    'OppositeVehicleRunningRedLight_Town04_Route180_Weather23',
    'InterurbanAdvancedActorFlow_Town06_Route324_Weather2',
    'HighwayCutIn_Town12_Route1029_Weather15',
    'MergerIntoSlowTraffic_Town06_Route317_Weather5',
    'NonSignalizedJunctionLeftTurn_Town07_Route342_Weather3',
    'AccidentTwoWays_Town12_Route1115_Weather23',
    'ParkingCrossingPedestrian_Town13_Route545_Weather25',
    'VanillaSignalizedTurnEncounterGreenLight_Town07_Route354_Weather8',
    'ParkingExit_Town12_Route922_Weather12',
    'VanillaSignalizedTurnEncounterRedLight_Town15_Route491_Weather23',
    'HardBreakRoute_Town01_Route32_Weather6',
    'DynamicObjectCrossing_Town01_Route3_Weather3',
    'ConstructionObstacle_Town12_Route78_Weather0',
    'EnterActorFlow_Town03_Route132_Weather2',
    'HazardAtSideLane_Town10HD_Route373_Weather9',
    'InvadingTurn_Town02_Route95_Weather9',
    'TJunction_Town05_Route260_Weather0',
    'VehicleTurningRoute_Town15_Route504_Weather10',
    'DynamicObjectCrossing_Town02_Route11_Weather11',
    'TJunction_Town06_Route306_Weather20',
    'ParkedObstacleTwoWays_Town13_Route1333_Weather26',
    'SignalizedJunctionRightTurn_Town03_Route118_Weather14',
    'NonSignalizedJunctionLeftTurnEnterFlow_Town12_Route949_Weather13',
    'VehicleOpensDoorTwoWays_Town12_Route1203_Weather7',
    'CrossingBicycleFlow_Town12_Route977_Weather15',
    'SignalizedJunctionLeftTurn_Town04_Route173_Weather26',
    'HighwayExit_Town06_Route312_Weather0',
    'Accident_Town05_Route218_Weather10',
    'ParkedObstacle_Town10HD_Route372_Weather8',
    'InterurbanActorFlow_Town12_Route1291_Weather1',
    'ParkingCutIn_Town13_Route1343_Weather1',
    'VehicleTurningRoutePedestrian_Town15_Route481_Weather19',
    'PedestrianCrossing_Town13_Route747_Weather19',
    'BlockedIntersection_Town03_Route135_Weather5',
]

def get_action(index):
	Discrete_Actions_DICT = {
		0:  (0, 0, 1, False),
		1:  (0.7, -0.5, 0, False),
		2:  (0.7, -0.3, 0, False),
		3:  (0.7, -0.2, 0, False),
		4:  (0.7, -0.1, 0, False),
		5:  (0.7, 0, 0, False),
		6:  (0.7, 0.1, 0, False),
		7:  (0.7, 0.2, 0, False),
		8:  (0.7, 0.3, 0, False),
		9:  (0.7, 0.5, 0, False),
		10: (0.3, -0.7, 0, False),
		11: (0.3, -0.5, 0, False),
		12: (0.3, -0.3, 0, False),
		13: (0.3, -0.2, 0, False),
		14: (0.3, -0.1, 0, False),
		15: (0.3, 0, 0, False),
		16: (0.3, 0.1, 0, False),
		17: (0.3, 0.2, 0, False),
		18: (0.3, 0.3, 0, False),
		19: (0.3, 0.5, 0, False),
		20: (0.3, 0.7, 0, False),
		21: (0, -1, 0, False),
		22: (0, -0.6, 0, False),
		23: (0, -0.3, 0, False),
		24: (0, -0.1, 0, False),
		25: (1, 0, 0, False),
		26: (0, 0.1, 0, False),
		27: (0, 0.3, 0, False),
		28: (0, 0.6, 0, False),
		29: (0, 1.0, 0, False),
		30: (0.5, -0.5, 0, True),
		31: (0.5, -0.3, 0, True),
		32: (0.5, -0.2, 0, True),
		33: (0.5, -0.1, 0, True),
		34: (0.5, 0, 0, True),
		35: (0.5, 0.1, 0, True),
		36: (0.5, 0.2, 0, True),
		37: (0.5, 0.3, 0, True),
		38: (0.5, 0.5, 0, True),
		}
	throttle, steer, brake, reverse = Discrete_Actions_DICT[index]
	return throttle, steer, brake

class Colors:
	RED = '\033[91m'
	GREEN = '\033[92m'
	YELLOW = '\033[93m'
	BLUE = '\033[94m'
	MAGENTA = '\033[95m'
	CYAN = '\033[96m'
	WHITE = '\033[97m'
	RESET = '\033[0m'

def gen_single_route(route_folder, count):

	folder_path = os.path.join(route_folder, 'anno')
	length = len([name for name in os.listdir(folder_path)]) - 1 # drop last frame
	
	if length < INPUT_FRAMES + FUTURE_FRAMES:
		return

	seq_future_x = []
	seq_future_y = []
	seq_future_theta = []
	seq_future_feature = []
	seq_future_action = []
	seq_future_action_index = []

	seq_future_only_ap_brake = []


	seq_input_x = []
	seq_input_y = []
	seq_input_theta = []

	seq_front_img = []
	seq_feature = []
	seq_value = []
	seq_speed = []

	seq_action = []
	seq_action_index = []

	seq_x_target = []
	seq_y_target = []
	seq_target_command = []

	seq_only_ap_brake = []

	full_seq_x = []
	full_seq_y = []
	full_seq_theta = []

	full_seq_feature = []
	full_seq_action = []
	full_seq_action_index = []
	full_seq_only_ap_brake = []

	for i in trange(length):
		with gzip.open(os.path.join(route_folder, f'anno/{i:05}.json.gz'), 'rt', encoding='utf-8') as gz_file:
			anno = json.load(gz_file)

		expert_feature = np.load(os.path.join(route_folder, f'expert_assessment/{i:05}.npz'), allow_pickle=True)['arr_0']

		full_seq_x.append(anno['x'])
		full_seq_y.append(anno['y'])  # TODO(yzj): need to align sign
		full_seq_theta.append(anno['theta'])
		full_seq_feature.append(expert_feature[:-2])
		throttle, steer, brake = get_action(int(expert_feature[-1]))
		full_seq_action.append(np.array([throttle, steer, brake], dtype=np.float32))
		full_seq_action_index.append(int(expert_feature[-1]))
		full_seq_only_ap_brake.append(anno['only_ap_brake'])

	for i in trange(INPUT_FRAMES-1, length-FUTURE_FRAMES-5):
		with gzip.open(os.path.join(route_folder, f'anno/{i:05}.json.gz'), 'rt', encoding='utf-8') as gz_file:
			anno = json.load(gz_file)

		expert_feature = np.load(os.path.join(route_folder, f'expert_assessment/{i:05}.npz'), allow_pickle=True)['arr_0']

		seq_input_x.append(full_seq_x[i-(INPUT_FRAMES-1):i+5:5])
		seq_input_y.append(full_seq_y[i-(INPUT_FRAMES-1):i+5:5])
		seq_input_theta.append(full_seq_theta[i-(INPUT_FRAMES-1):i+5:5])

		seq_future_x.append(full_seq_x[i+5:i+FUTURE_FRAMES+5:5])
		seq_future_y.append(full_seq_y[i+5:i+FUTURE_FRAMES+5:5])
		seq_future_theta.append(full_seq_theta[i+5:i+FUTURE_FRAMES+5:5])

		seq_future_feature.append(full_seq_feature[i+5:i+FUTURE_FRAMES+5:5])
		seq_future_action.append(full_seq_action[i+5:i+FUTURE_FRAMES+5:5])
		seq_future_action_index.append(full_seq_action_index[i+5:i+FUTURE_FRAMES+5:5])
		seq_future_only_ap_brake.append(full_seq_only_ap_brake[i+5:i+FUTURE_FRAMES+5:5])

		seq_feature.append(expert_feature[:-2])
		seq_value.append(expert_feature[-2])
		
		front_img_list = [os.path.join(route_folder, f'camera/rgb_front/{i:05}.jpg') for _ in range(INPUT_FRAMES-1, -1, -1)]
		seq_front_img.append(front_img_list)

		seq_speed.append(anno["speed"])

		throttle, steer, brake = get_action(int(expert_feature[-1]))
		seq_action.append(np.array([throttle, steer, brake], dtype=np.float32))  # step + action = next_step
		seq_action_index.append(int(expert_feature[-1]))

		seq_x_target.append(anno["x_target"])
		seq_y_target.append(anno["y_target"])
		seq_target_command.append(anno["next_command"])
		seq_only_ap_brake.append(anno["only_ap_brake"])

	with count.get_lock():
		count.value += 1
	return seq_future_x, seq_future_y, seq_future_theta, seq_future_feature, seq_future_action, seq_future_action_index, seq_future_only_ap_brake, seq_input_x, seq_input_y, seq_input_theta, seq_front_img, seq_feature, seq_value, seq_speed, seq_action, seq_action_index, seq_x_target, seq_y_target, seq_target_command, seq_only_ap_brake

def gen_sub_folder(seq_data_list):
	print('begin saving...', flush=True)
	total_future_x = []
	total_future_y = []
	total_future_theta = []

	total_future_feature = []
	total_future_action = []
	total_future_action_index = []
	total_future_only_ap_brake = []

	total_input_x = []
	total_input_y = []
	total_input_theta = []

	total_front_img = []
	total_feature = []
	total_value = []
	total_speed = []

	total_action = []
	total_action_index = []

	total_x_target = []
	total_y_target = []
	total_target_command = []

	total_only_ap_brake = []

	for seq_data in seq_data_list:
		# seq_data = gen_single_route(os.path.join(folder_path, route))
		if not seq_data:
			continue
		seq_future_x, seq_future_y, seq_future_theta, seq_future_feature, seq_future_action, seq_future_action_index, seq_future_only_ap_brake, seq_input_x, seq_input_y, seq_input_theta, seq_front_img, seq_feature, seq_value, seq_speed, seq_action, seq_action_index, seq_x_target, seq_y_target, seq_target_command, seq_only_ap_brake = seq_data
		total_future_x.extend(seq_future_x)
		total_future_y.extend(seq_future_y)
		total_future_theta.extend(seq_future_theta)
		total_future_feature.extend(seq_future_feature)
		total_future_action.extend(seq_future_action)
		total_future_action_index.extend(seq_future_action_index)
		total_future_only_ap_brake.extend(seq_future_only_ap_brake)
		total_input_x.extend(seq_input_x)
		total_input_y.extend(seq_input_y)
		total_input_theta.extend(seq_input_theta)
		total_front_img.extend(seq_front_img)
		total_feature.extend(seq_feature)
		total_value.extend(seq_value)
		total_speed.extend(seq_speed)
		total_action.extend(seq_action)
		total_action_index.extend(seq_action_index)
		total_x_target.extend(seq_x_target)
		total_y_target.extend(seq_y_target)
		total_target_command.extend(seq_target_command)
		total_only_ap_brake.extend(seq_only_ap_brake)

	data_dict = {}
	data_dict['future_x'] = total_future_x
	data_dict['future_y'] = total_future_y
	data_dict['future_theta'] = total_future_theta
	data_dict['future_feature'] = total_future_feature
	data_dict['future_action'] = total_future_action
	data_dict['future_action_index'] = total_future_action_index
	data_dict['future_only_ap_brake'] = total_future_only_ap_brake
	data_dict['input_x'] = total_input_x
	data_dict['input_y'] = total_input_y
	data_dict['input_theta'] = total_input_theta
	data_dict['front_img'] = total_front_img
	data_dict['feature'] = total_feature
	data_dict['value'] = total_value
	data_dict['speed'] = total_speed
	data_dict['action'] = total_action
	data_dict['action_index'] = total_action_index
	data_dict['x_target'] = total_x_target
	data_dict['y_target'] = total_y_target
	data_dict['target_command'] = total_target_command
	data_dict['only_ap_brake'] = total_only_ap_brake

	if TRAIN:
		file_path = os.path.join("tcp_bench2drive-train")
	else:
		file_path = os.path.join("tcp_bench2drive-val")
	np.save(file_path, data_dict)
	print(f'begin saving, length={len(total_future_x)}', flush=True)

def get_folder_path(folder_paths, total):
	path = 'YOUR_PATH'
	for d0 in os.listdir(path):
		if TRAIN:
			if d0 not in val_list:
				folder_paths.put(os.path.join(path, d0))
				with total.get_lock():
					total.value += 1
		else:
			if d0 in val_list:
				folder_paths.put(os.path.join(path, d0))
				with total.get_lock():
					total.value += 1
	return folder_paths

def worker(folder_paths, count, seq_data_list, stop_event, worker_num, completed_workers):
	while True:
		if folder_paths.qsize()<=0:
			with completed_workers.get_lock():
				completed_workers.value += 1
				if completed_workers.value == worker_num:
					stop_event.set()
			break
		folder_path = folder_paths.get()
		seq_data = gen_single_route(folder_path, count)
		seq_data_list.append(seq_data)

def display(count, total, stop_event, completed_workers):
	t1 = time.time()
	while True:
		print(f'{Colors.GREEN}[count/total]=[{count.value}/{total.value}, {count.value/(time.time()-t1):.2f}it/s, completed_workers={completed_workers.value}]{Colors.RESET}', flush=True)
		time.sleep(3)
		if stop_event.is_set():
			break

if __name__ == '__main__':
	folder_paths = mp.Queue()
	seq_data_list = mp.Manager().list()
	count = mp.Value('d', 0)
	total = mp.Value('d', 0)
	stop_event = mp.Event()
	completed_workers = mp.Value('d', 0)

	get_folder_path(folder_paths, total)
	ps = []
	worker_num = 64
	for i in range(worker_num):
		p = mp.Process(target=worker, args=(folder_paths, count, seq_data_list, stop_event, worker_num, completed_workers, ))
		p.daemon = True
		p.start()
		ps.append(p)
	
	p = mp.Process(target=display, args=(count, total, stop_event, completed_workers))
	p.daemon = True
	p.start()
	ps.append(p)
	
	for p in ps:
		p.join()
	
	display(count, total, stop_event, completed_workers)
	gen_sub_folder(seq_data_list)
	display(count, total, stop_event, completed_workers)