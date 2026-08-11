import os
from os.path import join
import gzip, json, pickle
import numpy as np
from pyquaternion import Quaternion
from tqdm import tqdm
from vis_utils import calculate_cube_vertices,calculate_occlusion_stats,edges,DIS_CAR_SAVE
import cv2
import multiprocessing
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
# All data in the Bench2Drive dataset are in the left-handed coordinate system.
# This code converts all coordinate systems (world coordinate system, vehicle coordinate system,
# camera coordinate system, and lidar coordinate system) to the right-handed coordinate system
# consistent with the nuscenes dataset.

DATA_ROOT = './data/bench2drive/trainval'
MAP_ROOT = './data/bench2drive/maps'
OUT_DIR = './data/infos'

MAX_DISTANCE = 75              # Filter bounding boxes that are too far from the vehicle
FILTER_Z_SHRESHOLD = 10        # Filter bounding boxes that are too high/low from the vehicle
FILTER_INVISINLE = True        # Filter bounding boxes based on visibility
NUM_VISIBLE_SHRESHOLD = 1      # Filter bounding boxes with fewer visible vertices than this value
NUM_OUTPOINT_SHRESHOLD = 7     # Filter bounding boxes where the number of vertices outside the frame is greater than this value in all cameras
CAMERAS = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
CAMERA_TO_FOLDER_MAP = {'CAM_FRONT':'rgb_front', 'CAM_FRONT_LEFT':'rgb_front_left', 'CAM_FRONT_RIGHT':'rgb_front_right', 'CAM_BACK':'rgb_back', 'CAM_BACK_LEFT':'rgb_back_left', 'CAM_BACK_RIGHT':'rgb_back_right'}

stand_to_ue4_rotate = np.array([[ 0, 0, 1, 0],
                                [ 1, 0, 0, 0],
                                [ 0,-1, 0, 0],
                                [ 0, 0, 0, 1]])

lidar_to_righthand_ego = np.array([[  0, 1, 0, 0],
                                   [ -1, 0, 0, 0],
                                   [  0, 0, 1, 0],
                                   [  0, 0, 0, 1]])

lefthand_ego_to_lidar = np.array([[ 0, 1, 0, 0],
                                  [ 1, 0, 0, 0],
                                  [ 0, 0, 1, 0],
                                  [ 0, 0, 0, 1]])

left2right = np.eye(4)
left2right[1,1] = -1

def apply_trans(vec,world2ego):
    vec = np.concatenate((vec,np.array([1])))
    t = world2ego @ vec
    return t[0:3]

def get_pose_matrix(dic):
    new_matrix = np.zeros((4,4))
    new_matrix[0:3,0:3] = Quaternion(axis=[0, 0, 1], radians=dic['theta']-np.pi/2).rotation_matrix
    new_matrix[0,3] = dic['x']
    new_matrix[1,3] = dic['y']
    new_matrix[3,3] = 1
    return new_matrix

def get_npc2world(npc):
    for key in ['world2vehicle','world2ego','world2sign','world2ped']:
        if key in npc.keys():
            npc2world = np.linalg.inv(np.array(npc[key]))
            yaw_from_matrix = np.arctan2(npc2world[1,0], npc2world[0,0])
            yaw = npc['rotation'][-1] / 180 * np.pi
            if abs(yaw-yaw_from_matrix)> 0.01:
                npc2world[0:3,0:3] = Quaternion(axis=[0, 0, 1], radians=yaw).rotation_matrix
            npc2world = left2right @ npc2world @ left2right
            return npc2world
    npc2world = np.eye(4)
    npc2world[0:3,0:3] = Quaternion(axis=[0, 0, 1], radians=npc['rotation'][-1]/180*np.pi).rotation_matrix
    npc2world[0:3,3] = np.array(npc['location'])
    return left2right @ npc2world @ left2right


def get_global_trigger_vertex(center,extent,yaw_in_degree):
    x,y = center[0],-center[1]
    dx,dy = extent[0],extent[1]
    yaw_in_radians = -yaw_in_degree/180*np.pi
    vertex_in_self = np.array([[ dx, dy],
                               [-dx, dy],
                               [-dx,-dy],
                               [ dx,-dy]])
    rotate_matrix = np.array([[np.cos(yaw_in_radians),-np.sin(yaw_in_radians)],
                              [np.sin(yaw_in_radians), np.cos(yaw_in_radians)]])
    rotated_vertex = (rotate_matrix @ vertex_in_self.T).T
    vertex_in_global = np.array([[x,y]]).repeat(4,axis=0) + rotated_vertex
    return vertex_in_global



def get_image_point(loc, K, w2c):
    point = np.array([loc[0], loc[1], loc[2], 1])
    point_camera = np.dot(w2c, point)
    point_camera = point_camera[0:3]
    depth = point_camera[2]
    point_img = np.dot(K, point_camera)
    point_img[0] /= point_img[2]
    point_img[1] /= point_img[2]
    return point_img[0:2], depth

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



def preprocess_map(map_file_name):
    raw_map = dict(np.load(join(MAP_ROOT, map_file_name), allow_pickle=True)['arr'])
    town_name = map_file_name.split('_')[0]
    map_info = {} 
    lane_points = []
    lane_types = []
    lane_sample_points = []
    trigger_volumes_points = []
    trigger_volumes_types = []
    trigger_volumes_sample_points = []
    for road_id, road in raw_map.items():
        for lane_id, lane in road.items():
            if lane_id == 'Trigger_Volumes':
                for single_trigger_volume in lane:
                    points = np.array(single_trigger_volume['Points'])
                    points[:,1] *= -1 #left2right
                    trigger_volumes_points.append(points)
                    trigger_volumes_sample_points.append(points.mean(axis=0))
                    trigger_volumes_types.append(single_trigger_volume['Type'])
            else:
                for single_lane in lane:
                    points = np.array([raw_point[0] for raw_point in single_lane['Points']])
                    points[:,1] *= -1
                    lane_points.append(points)
                    lane_types.append(single_lane['Type'])
                    lane_lenth = points.shape[0]
                    if lane_lenth % 50 != 0:
                        devide_points = [50*i for i in range(lane_lenth//50+1)]
                    else:
                        devide_points = [50*i for i in range(lane_lenth//50)]
                    devide_points.append(lane_lenth-1)
                    lane_sample_points_tmp = points[devide_points]
                    lane_sample_points.append(lane_sample_points_tmp)
    map_info['lane_points'] = lane_points
    map_info['lane_sample_points'] = lane_sample_points
    map_info['lane_types'] = lane_types
    map_info['trigger_volumes_points'] = trigger_volumes_points
    map_info['trigger_volumes_sample_points'] = trigger_volumes_sample_points
    map_info['trigger_volumes_types'] = trigger_volumes_types
    
    os.makedirs(join(OUT_DIR, 'b2d_infos_map'), exist_ok=True)
    with open(join(OUT_DIR, 'b2d_infos_map', town_name + '.pkl'),'wb') as f:
        pickle.dump(map_info, f)
        
def generate_map_infos(map_workers):
    map_file_names = [ f for f in os.listdir(MAP_ROOT) if '.npz' in f]
    with ProcessPoolExecutor(max_workers=map_workers) as executor:
        futures = [executor.submit(preprocess_map, map_file_name) for map_file_name in map_file_names]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing maps"):
            pass

def preprocess(folder_name, split):
    route_data = []
    folder_path = join(DATA_ROOT, folder_name)
    last_position_dict = {}
    for ann_name in sorted(os.listdir(join(folder_path,'anno')),key= lambda x: int(x.split('.')[0])):
        position_dict = {}
        frame_data = {}
        cam_gray_depth = {}
        with gzip.open(join(folder_path,'anno',ann_name), 'rt', encoding='utf-8') as gz_file:
            anno = json.load(gz_file) 
        frame_data['folder'] = folder_name
        frame_data['town_name'] =  folder_name.split('_')[1] if 'Town' in folder_name.split('_')[1] else folder_name.split('_')[2]
        frame_data['command_far_xy'] = np.array([anno['x_command_far'],-anno['y_command_far']])
        frame_data['command_far'] = anno['command_far']
        frame_data['command_near_xy'] = np.array([anno['x_command_near'],-anno['y_command_near']])
        frame_data['command_near'] = anno['command_near']
        frame_data['frame_idx'] = int(ann_name.split('.')[0])
        frame_data['ego_yaw'] = -np.nan_to_num(anno['theta'],nan=np.pi)+np.pi/2 # align with LiDARInstance3DBoxes in mmcv/core/bbox/structures/lidar_box3d.py
        frame_data['ego_translation'] = np.array([anno['x'],-anno['y'],0])
        frame_data['ego_vel'] = np.array([anno['speed'],0,0])
        frame_data['ego_accel'] = np.array([anno['acceleration'][0],-anno['acceleration'][1],anno['acceleration'][2]])
        frame_data['ego_rotation_rate'] = -np.array(anno['angular_velocity'])
        frame_data['ego_size'] = np.array([anno['bounding_boxes'][0]['extent'][1],anno['bounding_boxes'][0]['extent'][0],anno['bounding_boxes'][0]['extent'][2]])*2
        world2ego = left2right @ anno['bounding_boxes'][0]['world2ego'] @ left2right
        frame_data['world2ego'] = world2ego
        if frame_data['frame_idx'] == 0:
            expert_file_path = join(folder_path,'expert_assessment','-0001.npz')
        else:
            expert_file_path = join(folder_path,'expert_assessment',str(frame_data['frame_idx']-1).zfill(5)+'.npz')
        expert_data = np.load(expert_file_path,allow_pickle=True)['arr_0']
        action_id = expert_data[-1]
        value = expert_data[-2]
        expert_feature = expert_data[:-2]
        throttle, steer, brake = get_action(action_id)
        frame_data['brake'] = brake
        frame_data['throttle'] = throttle
        frame_data['steer'] = steer
        frame_data['action_id'] = action_id
        frame_data['value'] = value
        frame_data['expert_feature'] = expert_feature
        ###get sensor infos###
        sensor_infos = {}
        for cam in CAMERAS:
            sensor_infos[cam] = {}
            sensor_infos[cam]['intrinsic'] = np.array(anno['sensors'][cam]['intrinsic'])
            sensor_infos[cam]['world2cam'] = np.linalg.inv(stand_to_ue4_rotate) @ np.array(anno['sensors'][cam]['world2cam']) @left2right
            cam2ego = world2ego @ np.linalg.inv(sensor_infos[cam]['world2cam'])
            sensor_infos[cam]['cam2ego'] = cam2ego
            sensor_infos[cam]['data_path'] = join(folder_name,'camera',CAMERA_TO_FOLDER_MAP[cam],ann_name.split('.')[0]+'.jpg')
            depth_path = join(
                DATA_ROOT,
                sensor_infos[cam]['data_path']
                    .replace('rgb_', 'depth_')
                    .replace('.jpg', '.npz')
            )
            cam_gray_depth[cam] = None
            if os.path.exists(depth_path):
                try:
                    with np.load(depth_path) as data:
                        key = list(data.keys())[0]
                        cam_gray_depth[cam] = data[key].astype(np.float32)
                except Exception as e:
                    print(f"Depth load failed: {depth_path}", e)
            else:
                print(f"[error] Depth does not exist!")
        sensor_infos['LIDAR_TOP'] = {}
        world2lidar = lefthand_ego_to_lidar @ np.array(anno['sensors']['LIDAR_TOP']['world2lidar']) @ left2right
        sensor_infos['LIDAR_TOP']['world2lidar'] = world2lidar
        lidar2ego = world2ego @ np.linalg.inv(sensor_infos['LIDAR_TOP']['world2lidar'])
        sensor_infos['LIDAR_TOP']['lidar2ego'] = lidar2ego
        frame_data['sensors'] = sensor_infos
        ###get bounding_boxes infos###
        gt_boxes = []
        gt_names = []
        gt_ids = []
        num_points_list = []
        npc2world_list = []
        for npc in anno['bounding_boxes']:
            if npc['class'] == 'ego_vehicle': continue
            if npc['distance'] > MAX_DISTANCE: continue
            if abs(npc['location'][2] - anno['bounding_boxes'][0]['location'][2]) > FILTER_Z_SHRESHOLD: continue
            center = np.array([npc['center'][0],-npc['center'][1],npc['center'][2]]) # left hand -> right hand
            extent = np.array([npc['extent'][1],npc['extent'][0],npc['extent'][2]])  # lwh -> wlh
            position_dict[npc['id']] = center
            local_center = apply_trans(center, world2lidar)
            size = extent * 2 
            if 'world2vehicle' in npc.keys():
                world2vehicle = left2right @ np.array(npc['world2vehicle'])@left2right
                vehicle2lidar = world2lidar @ np.linalg.inv(world2vehicle) 
                yaw_local = np.arctan2(vehicle2lidar[1,0], vehicle2lidar[0,0])

            else:
                yaw_local = -npc['rotation'][-1]/180*np.pi - frame_data['ego_yaw'] +np.pi / 2  
            yaw_local_in_lidar_box = -yaw_local - np.pi / 2  
            while yaw_local < -np.pi:
                yaw_local += 2*np.pi
            while yaw_local > np.pi:
                yaw_local -= 2*np.pi  
            if 'speed' in npc.keys():
                if 'vehicle' in npc['class']:  # only vehicles have correct speed
                    speed = npc['speed']
                else:
                    if npc['id'] in last_position_dict.keys():  #calculate speed for other object
                        speed = np.linalg.norm((center-last_position_dict[npc['id']])[0:2]) * 10
                    else:
                        speed = 0
            else:
                speed = 0
            if 'num_points' in npc.keys():
                num_points = npc['num_points']
            else:
                num_points = -1
            npc2world = get_npc2world(npc)
            speed_x = speed * np.cos(yaw_local)
            speed_y = speed * np.sin(yaw_local)

            ###fliter_bounding_boxes###
            if FILTER_INVISINLE:
                valid = False
                box2lidar = np.eye(4)
                box2lidar[0:3,0:3] = Quaternion(axis=[0, 0, 1], radians=yaw_local).rotation_matrix
                box2lidar[0:3,3] = local_center
                lidar2box = np.linalg.inv(box2lidar)
                raw_verts = calculate_cube_vertices(local_center,extent)
                verts = []
                for raw_vert in raw_verts:
                    tmp = np.dot(lidar2box, [raw_vert[0], raw_vert[1], raw_vert[2],1])
                    tmp[0:3] += local_center
                    verts.append(tmp.tolist()[:-1])
                for cam in CAMERAS:
                    lidar2cam = np.linalg.inv(frame_data['sensors'][cam]['cam2ego']) @ sensor_infos['LIDAR_TOP']['lidar2ego']
                    test_points = [] 
                    test_depth = []
                    for vert in verts:
                        point, depth = get_image_point(vert, frame_data['sensors'][cam]['intrinsic'], lidar2cam)
                        if depth > 0:
                            test_points.append(point)
                            test_depth.append(depth)
                    num_visible_vertices, num_invisible_vertices, num_vertices_outside_camera, colored_points = calculate_occlusion_stats(np.array(test_points), np.array(test_depth),  cam_gray_depth[cam], max_render_depth=MAX_DISTANCE)
                    if num_visible_vertices>NUM_VISIBLE_SHRESHOLD and num_vertices_outside_camera<NUM_OUTPOINT_SHRESHOLD:
                        valid = True
                        break
            else:
                valid = True
            if valid:
                npc2world_list.append(npc2world)
                num_points_list.append(num_points)            
                gt_boxes.append(np.concatenate([local_center,size,np.array([yaw_local_in_lidar_box,speed_x,speed_y])]))
                gt_names.append(npc['type_id'])
                gt_ids.append(int(npc['id']))

        if len(gt_boxes) == 0:
            continue
        last_position_dict = position_dict.copy()    
        gt_ids = np.array(gt_ids)
        gt_names = np.array(gt_names)
        num_points_list = np.array(num_points_list)
        gt_boxes = np.stack(gt_boxes)
        npc2world = np.stack(npc2world_list)
        frame_data['gt_ids'] = gt_ids
        frame_data['gt_boxes'] = gt_boxes
        frame_data['gt_names'] = gt_names
        frame_data['num_points'] = num_points_list
        frame_data['npc2world'] = npc2world
        # frame_data['desired_speed'] = speed_data[folder_name.split('/')[1]][frame_data['frame_idx']]    
        route_data.append(frame_data)
        
    os.makedirs(join(OUT_DIR, 'b2d_infos_' + split), exist_ok=True)
    with open(join(OUT_DIR, 'b2d_infos_' + split, folder_name+'.pkl'),'wb') as f:
        pickle.dump(route_data,f)
    meta = {'route_name':folder_name, 'lenth':len(route_data)}
    return meta


def generate_infos(folder_list, workers, split):
    metas = {}
    metas['routes_names'] = []
    metas['divide_nums'] = []    
    total_lenth = 0   
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(preprocess, file, split) for file in folder_list]
        for future in tqdm(as_completed(futures), total=len(folder_list), desc="Processing files"):
            try:
                result = future.result()
                total_lenth += result['lenth']
                metas['routes_names'].append(result['route_name'])
                metas['divide_nums'].append(total_lenth)                 
            except Exception as e:
                print(f"Error processing file: {e}")

    metas['total_lenth'] = total_lenth
    metas['split'] =  split
    metas['infos_dir_name'] = join(OUT_DIR,'b2d_infos_' + split)

    with open(join(OUT_DIR,'b2d_infos_' + split + '_meta.pkl'),'wb') as f:
        pickle.dump(metas,f)

if __name__ == "__main__":

    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument('--info_workers', type=int, default=4, help='num of workers to prepare infos')
    argparser.add_argument('--map_workers', type=int, default=4, help='num of workers to prepare maps')
    args = argparser.parse_args()    
    info_workers = args.info_workers
    map_workers = args.map_workers
    os.makedirs(OUT_DIR, exist_ok=True)
    process_list = []
    with open('./data/splits/bench2drive_train_val_split_v04.json','r') as f:
        train_val_split = json.load(f)
    all_folder = os.listdir(DATA_ROOT)
    train_list = []
    for foldername in all_folder:
        if 'Town' in foldername and 'Route' in foldername and 'Weather' in foldername and not foldername in train_val_split['val']:
            train_list.append(foldername)
    print('processing train data...')
    generate_infos(train_list, info_workers, 'train')
    process_list = []
    print('processing val data...')
    generate_infos(train_val_split['val'], info_workers, 'val')
    print('processing map data...')
    generate_map_infos(map_workers)
    print('finish!')