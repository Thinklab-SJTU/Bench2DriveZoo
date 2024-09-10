import os
import json
import numpy as np
from tqdm import trange
import gzip
import multiprocessing as mp 
import time


INPUT_FRAMES = 5*5
FUTURE_FRAMES = 6*5

TRAIN = False

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

	seq_input_x = []
	seq_input_y = []
	seq_input_theta = []

	seq_input_speed = []
	seq_input_speed_acc = []
	seq_input_command = []

	full_seq_x = []
	full_seq_y = []
	full_seq_theta = []
	full_seq_speed = []
	full_seq_speed_acc = []
	full_seq_command = []

	for i in trange(length):
		with gzip.open(os.path.join(route_folder, f'anno/{i:05}.json.gz'), 'rt', encoding='utf-8') as gz_file:
			anno = json.load(gz_file)
		
		full_seq_x.append(anno['x'])
		full_seq_y.append(anno['y'])  # TODO(yzj): need to align sign
		full_seq_speed.append(anno['speed'])
		full_seq_speed_acc.append(anno['acceleration'])
		full_seq_theta.append(anno['theta'])
		full_seq_command.append(anno['next_command'])

	for i in trange(INPUT_FRAMES-5, length-FUTURE_FRAMES):
		with gzip.open(os.path.join(route_folder, f'anno/{i:05}.json.gz'), 'rt', encoding='utf-8') as gz_file:
			anno = json.load(gz_file)

		seq_input_x.append(full_seq_x[i-(INPUT_FRAMES-5):i+5:5])
		seq_input_y.append(full_seq_y[i-(INPUT_FRAMES-5):i+5:5])
		seq_input_theta.append(full_seq_theta[i-(INPUT_FRAMES-5):i+5:5])

		seq_input_speed.append(full_seq_speed[i-(INPUT_FRAMES-5):i+5:5])
		seq_input_speed_acc.append(full_seq_speed_acc[i-(INPUT_FRAMES-5):i+5:5])
		seq_input_command.append(full_seq_command[i-(INPUT_FRAMES-5):i+5:5])

		seq_future_x.append(full_seq_x[i+5:i+FUTURE_FRAMES+5:5])
		seq_future_y.append(full_seq_y[i+5:i+FUTURE_FRAMES+5:5])
		seq_future_theta.append(full_seq_theta[i+5:i+FUTURE_FRAMES+5:5])

	with count.get_lock():
		count.value += 1
	return seq_future_x, seq_future_y, seq_future_theta, seq_input_x, seq_input_y, seq_input_theta, seq_input_speed, seq_input_speed_acc, seq_input_command, full_seq_x, full_seq_y, full_seq_theta, full_seq_speed, full_seq_speed_acc, full_seq_command 

def gen_sub_folder(seq_data_list):
	print('begin saving...')
	total_future_x = []
	total_future_y = []
	total_future_theta = []

	total_input_x = []
	total_input_y = []
	total_input_theta = []

	total_input_speed = []
	total_input_speed_acc = []
	total_input_command = []

	for seq_data in seq_data_list:
		if not seq_data:
			continue
		seq_future_x, seq_future_y, seq_future_theta, seq_input_x, seq_input_y, seq_input_theta, seq_input_speed, seq_input_speed_acc, seq_input_command, full_seq_x, full_seq_y, full_seq_theta, full_seq_speed, full_seq_speed_acc, full_seq_command  = seq_data
		total_future_x.extend(seq_future_x)
		total_future_y.extend(seq_future_y)
		total_future_theta.extend(seq_future_theta)
		total_input_x.extend(seq_input_x)
		total_input_y.extend(seq_input_y)
		total_input_theta.extend(seq_input_theta)
		total_input_speed.extend(seq_input_speed)
		total_input_speed_acc.extend(seq_input_speed_acc)
		total_input_command.extend(seq_input_command)

	data_dict = {}
	data_dict['future_x'] = total_future_x
	data_dict['future_y'] = total_future_y
	data_dict['future_theta'] = total_future_theta
	data_dict['input_x'] = total_input_x
	data_dict['input_y'] = total_input_y
	data_dict['input_theta'] = total_input_theta
	data_dict['input_speed'] = total_input_speed
	data_dict['input_speed_acc'] = total_input_speed_acc
	data_dict['input_command'] = total_input_command		
	if TRAIN:
		file_path = os.path.join("admlp_bench2drive-train")
	else:
		file_path = os.path.join("admlp_bench2drive-val")
	np.save(file_path, data_dict)
	print(f'begin saving, length={len(total_future_x)}')

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