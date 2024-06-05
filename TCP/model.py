from collections import deque
import numpy as np
import torch 
from torch import nn
from TCP.resnet import *

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

class PIDController(object):
    def __init__(self, K_P=1.0, K_I=0.0, K_D=0.0, n=20):
        self._K_P = K_P
        self._K_I = K_I
        self._K_D = K_D

        self._window = deque([0 for _ in range(n)], maxlen=n)
        self._max = 0.0
        self._min = 0.0

    def step(self, error):
        self._window.append(error)
        self._max = max(self._max, abs(error))
        self._min = -abs(self._max)

        if len(self._window) >= 2:
            integral = np.mean(self._window)
            derivative = (self._window[-1] - self._window[-2])
        else:
            integral = 0.0
            derivative = 0.0

        return self._K_P * error + self._K_I * integral + self._K_D * derivative

class TCP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.turn_controller = PIDController(K_P=config.turn_KP, K_I=config.turn_KI, K_D=config.turn_KD, n=config.turn_n)
        self.speed_controller = PIDController(K_P=config.speed_KP, K_I=config.speed_KI, K_D=config.speed_KD, n=config.speed_n)

        self.perception = resnet34(pretrained=True)

        self.measurements = nn.Sequential(
                            nn.Linear(1+2+6, 128),
                            nn.ReLU(inplace=True),
                            nn.Linear(128, 128),
                            nn.ReLU(inplace=True),
                        )

        self.join_traj = nn.Sequential(
                            nn.Linear(128+1000, 512),
                            nn.ReLU(inplace=True),
                            nn.Linear(512, 512),
                            nn.ReLU(inplace=True),
                            nn.Linear(512, 256),
                            nn.ReLU(inplace=True),
                        )

        self.join_ctrl = nn.Sequential(
                            nn.Linear(128+512, 512),
                            nn.ReLU(inplace=True),
                            nn.Linear(512, 512),
                            nn.ReLU(inplace=True),
                            nn.Linear(512, 256),
                            nn.ReLU(inplace=True),
                        )

        self.speed_branch = nn.Sequential(
                            nn.Linear(1000, 256),
                            nn.ReLU(inplace=True),
                            nn.Linear(256, 256),
                            nn.Dropout2d(p=0.5),
                            nn.ReLU(inplace=True),
                            nn.Linear(256, 1),
                        )

        self.value_branch_traj = nn.Sequential(
                    nn.Linear(256, 256),
                    nn.ReLU(inplace=True),
                    nn.Linear(256, 256),
                    nn.Dropout2d(p=0.5),
                    nn.ReLU(inplace=True),
                    nn.Linear(256, 1),
                )
        
        self.feature_branch_traj = nn.Sequential(
                    nn.Linear(256, 1536),
                    nn.ReLU(inplace=True),
        )

        self.feature_branch_ctrl = nn.Sequential(
                    nn.Linear(256, 1536),
                    nn.ReLU(inplace=True),
        )

        self.value_branch_ctrl = nn.Sequential(
                    nn.Linear(256, 256),
                    nn.ReLU(inplace=True),
                    nn.Linear(256, 256),
                    nn.Dropout2d(p=0.5),
                    nn.ReLU(inplace=True),
                    nn.Linear(256, 1),
                )
        # shared branches_neurons

        self.policy_head = nn.Sequential(
                nn.Linear(256, 256),
                nn.ReLU(inplace=True),
                nn.Linear(256, 256),
                nn.Dropout2d(p=0.5),
                nn.ReLU(inplace=True),
            )
        self.decoder_ctrl = nn.GRUCell(input_size=256+39, hidden_size=256)
        self.output_ctrl = nn.Sequential(
                nn.Linear(256, 256),
                nn.ReLU(inplace=True),
                nn.Linear(256, 256),
                nn.ReLU(inplace=True),
            )

        self.num_actions = 39
        self.action_head = nn.Linear(256, self.num_actions)

        self.decoder_traj = nn.GRUCell(input_size=4, hidden_size=256)
        self.output_traj = nn.Linear(256, 2)

        self.init_att = nn.Sequential(
                nn.Linear(128, 256),
                nn.ReLU(inplace=True),
                nn.Linear(256, 29*8),
                nn.Softmax(1)
            )

        self.wp_att = nn.Sequential(
                nn.Linear(256+256, 256),
                nn.ReLU(inplace=True),
                nn.Linear(256, 29*8),
                nn.Softmax(1)
            )

        self.merge = nn.Sequential(
                nn.Linear(512+256, 512),
                nn.ReLU(inplace=True),
                nn.Linear(512, 256),
            )
        

    def forward(self, img, state, target_point):
        feature_emb, cnn_feature = self.perception(img)
        outputs = {}
        outputs['pred_speed'] = self.speed_branch(feature_emb)
        measurement_feature = self.measurements(state)

        j_traj = self.join_traj(torch.cat([feature_emb, measurement_feature], 1))
        outputs['pred_value_traj'] = self.value_branch_traj(j_traj)
        outputs['pred_features_traj'] = self.feature_branch_traj(j_traj)
        z = j_traj
        output_wp = list()
        traj_hidden_state = list()

        # initial input variable to GRU
        x = torch.zeros(size=(z.shape[0], 2), dtype=z.dtype).type_as(z)

        # autoregressive generation of output waypoints
        for _ in range(self.config.pred_len):
            x_in = torch.cat([x, target_point], dim=1)
            z = self.decoder_traj(x_in, z)
            traj_hidden_state.append(z)
            dx = self.output_traj(z)
            x = dx + x
            output_wp.append(x)

        pred_wp = torch.stack(output_wp, dim=1)
        outputs['pred_wp'] = pred_wp

        traj_hidden_state = torch.stack(traj_hidden_state, dim=1)
        init_att = self.init_att(measurement_feature).view(-1, 1, 8, 29)
        feature_emb = torch.sum(cnn_feature*init_att, dim=(2, 3))
        j_ctrl = self.join_ctrl(torch.cat([feature_emb, measurement_feature], 1))
        outputs['pred_value_ctrl'] = self.value_branch_ctrl(j_ctrl)
        outputs['pred_features_ctrl'] = self.feature_branch_ctrl(j_ctrl)
        policy = self.policy_head(j_ctrl)
        outputs['action_index'] = self.action_head(policy)

        x = j_ctrl
        action_index = outputs['action_index']
        future_feature, future_action_index = [], []

        # initial hidden variable to GRU
        h = torch.zeros(size=(x.shape[0], 256), dtype=x.dtype).type_as(x)

        for _ in range(self.config.pred_len):
            x_in = torch.cat([x, action_index], dim=1)
            h = self.decoder_ctrl(x_in, h)
            wp_att = self.wp_att(torch.cat([h, traj_hidden_state[:, _]], 1)).view(-1, 1, 8, 29)
            new_feature_emb = torch.sum(cnn_feature*wp_att, dim=(2, 3))
            merged_feature = self.merge(torch.cat([h, new_feature_emb], 1))
            dx = self.output_ctrl(merged_feature)
            x = dx + x

            policy = self.policy_head(x)
            action_index = self.action_head(policy)
            future_feature.append(self.feature_branch_ctrl(x))
            future_action_index.append(action_index)

        outputs['future_feature'] = future_feature
        outputs['future_action_index'] = future_action_index
        return outputs

    def process_action(self, pred, command, speed, target_point):
        action_index = pred['action_index'].argmax().item()
        action = Discrete_Actions_DICT[action_index]
        throttle, steer, brake, reverse = action
        metadata = {
            'speed': float(speed.cpu().numpy().astype(np.float64)),
            'steer': float(steer),
            'throttle': float(throttle),
            'brake': float(brake),
            'reverse': float(reverse),
            'command': command,
            'target_point': tuple(target_point[0].data.cpu().numpy().astype(np.float64)),
        }
        return steer, throttle, brake, metadata

    def _get_action_beta(self, alpha, beta):
        x = torch.zeros_like(alpha)
        x[:, 1] += 0.5
        mask1 = (alpha > 1) & (beta > 1)
        x[mask1] = (alpha[mask1]-1)/(alpha[mask1]+beta[mask1]-2)

        mask2 = (alpha <= 1) & (beta > 1)
        x[mask2] = 0.0

        mask3 = (alpha > 1) & (beta <= 1)
        x[mask3] = 1.0

        # mean
        mask4 = (alpha <= 1) & (beta <= 1)
        x[mask4] = alpha[mask4]/torch.clamp((alpha[mask4]+beta[mask4]), min=1e-5)

        x = x * 2 - 1

        return x

    def control_pid(self, waypoints, velocity, target):
        ''' Predicts vehicle control with a PID controller.
        Args:
            waypoints (tensor): output of self.plan()
            velocity (tensor): speedometer input
        '''
        assert(waypoints.size(0)==1)
        waypoints = waypoints[0].data.cpu().numpy()
        target = target.squeeze().data.cpu().numpy()

        waypoints[:, [0, 1]] = waypoints[:, [1, 0]]  
        target[[0, 1]] = target[[1, 0]]

        # iterate over vectors between predicted waypoints
        num_pairs = len(waypoints) - 1
        best_norm = 1e5
        desired_speed = 0
        aim = waypoints[0]
        for i in range(num_pairs):
            # magnitude of vectors, used for speed
            desired_speed += np.linalg.norm(
                    waypoints[i+1] - waypoints[i]) * 2.0 / num_pairs

            # norm of vector midpoints, used for steering
            norm = np.linalg.norm((waypoints[i+1] + waypoints[i]) / 2.0)
            if abs(self.config.aim_dist-best_norm) > abs(self.config.aim_dist-norm):
                aim = waypoints[i]
                best_norm = norm

        aim_last = waypoints[-1] - waypoints[-2]

        angle = np.degrees(np.pi / 2 - np.arctan2(aim[1], aim[0])) / 90
        angle_last = np.degrees(np.pi / 2 - np.arctan2(aim_last[1], aim_last[0])) / 90
        angle_target = np.degrees(np.pi / 2 - np.arctan2(target[1], target[0])) / 90

        # choice of point to aim for steering, removing outlier predictions
        # use target point if it has a smaller angle or if error is large
        # predicted point otherwise
        # (reduces noise in eg. straight roads, helps with sudden turn commands)
        use_target_to_aim = np.abs(angle_target) < np.abs(angle)
        use_target_to_aim = use_target_to_aim or (np.abs(angle_target-angle_last) > self.config.angle_thresh and target[1] < self.config.dist_thresh)
        if use_target_to_aim:
            angle_final = angle_target
        else:
            angle_final = angle

        steer = self.turn_controller.step(angle_final)
        steer = np.clip(steer, -1.0, 1.0)

        speed = velocity[0].data.cpu().numpy()
        brake = desired_speed < self.config.brake_speed or (speed / desired_speed) > self.config.brake_ratio

        delta = np.clip(desired_speed - speed, 0.0, self.config.clip_delta)
        throttle = self.speed_controller.step(delta)
        throttle = np.clip(throttle, 0.0, self.config.max_throttle)
        throttle = throttle if not brake else 0.0

        metadata = {
            'speed': float(speed.astype(np.float64)),
            'steer': float(steer),
            'throttle': float(throttle),
            'brake': float(brake),
            'wp_4': tuple(waypoints[3].astype(np.float64)),
            'wp_3': tuple(waypoints[2].astype(np.float64)),
            'wp_2': tuple(waypoints[1].astype(np.float64)),
            'wp_1': tuple(waypoints[0].astype(np.float64)),
            'aim': tuple(aim.astype(np.float64)),
            'target': tuple(target.astype(np.float64)),
            'desired_speed': float(desired_speed.astype(np.float64)),
            'angle': float(angle.astype(np.float64)),
            'angle_last': float(angle_last.astype(np.float64)),
            'angle_target': float(angle_target.astype(np.float64)),
            'angle_final': float(angle_final.astype(np.float64)),
            'delta': float(delta.astype(np.float64)),
        }

        return steer, throttle, brake, metadata


    def get_action(self, action_index):
        # action = self._get_action_beta(mu.view(1,2), sigma.view(1,2))
        # acc, steer = action[:, 0], action[:, 1]
        # if acc >= 0.0:
        # 	throttle = acc
        # 	brake = torch.zeros_like(acc)
        # else:
        # 	throttle = torch.zeros_like(acc)
        # 	brake = torch.abs(acc)
        index = np.argmax(action_index.cpu().numpy())
        throttle, steer, brake, reverse = Discrete_Actions_DICT[index]
        throttle = torch.tensor(throttle).to(action_index.device)
        steer = torch.tensor(steer).to(action_index.device)
        brake = torch.tensor(brake).to(action_index.device)

        throttle = torch.clamp(throttle, 0, 1)
        steer = torch.clamp(steer, -1, 1)
        brake = torch.clamp(brake, 0, 1)

        return throttle, steer, brake