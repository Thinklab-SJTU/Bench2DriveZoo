import argparse
import os
from collections import OrderedDict
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy
from TCP.model import TCP
from TCP.data import CARLA_Data
from TCP.config import GlobalConfig


class TCP_planner(pl.LightningModule):
    def __init__(self, config, lr):
        super().__init__()
        self.lr = lr
        self.config = config
        self.model = TCP(config)

    def _load_state_dict(self, il_net, rl_state_dict, key_word):
        rl_keys = [k for k in rl_state_dict.keys() if key_word in k]
        il_keys = il_net.state_dict().keys()
        assert len(rl_keys) == len(il_net.state_dict().keys()), f'mismatch number of layers loading {key_word}'
        new_state_dict = OrderedDict()
        for k_il, k_rl in zip(il_keys, rl_keys):
            new_state_dict[k_il] = rl_state_dict[k_rl]
        il_net.load_state_dict(new_state_dict)
    
    def forward(self, batch):
        pass

    def training_step(self, batch, batch_idx):
        front_img = batch['front_img']
        speed = batch['speed'].to(dtype=torch.float32).view(-1,1) / 12.
        target_point = batch['target_point'].to(dtype=torch.float32)
        command = batch['target_command']
        
        state = torch.cat([speed, target_point, command], 1)
        value = batch['value'].view(-1,1)
        feature = batch['feature']

        gt_waypoints = batch['waypoints']

        pred = self.model(front_img, state, target_point)
        action_loss = F.nll_loss(F.log_softmax(pred['action_index'], dim=1), batch['action_index'])
        speed_loss = F.l1_loss(pred['pred_speed'], speed) * self.config.speed_weight
        value_loss = (F.mse_loss(pred['pred_value_traj'], value) + F.mse_loss(pred['pred_value_ctrl'], value)) * self.config.value_weight
        feature_loss = (F.mse_loss(pred['pred_features_traj'], feature) +F.mse_loss(pred['pred_features_ctrl'], feature))* self.config.features_weight

        future_feature_loss = 0
        future_action_loss = 0
        for i in range(self.config.pred_len):
            action_loss = F.nll_loss(F.log_softmax(pred['future_action_index'][i], dim=1), batch['future_action_index'][i])
            future_action_loss += action_loss
            future_feature_loss += F.mse_loss(pred['future_feature'][i], batch['future_feature'][i]) * self.config.features_weight
        future_feature_loss /= self.config.pred_len
        future_action_loss /= self.config.pred_len
        wp_loss = F.l1_loss(pred['pred_wp'], gt_waypoints, reduction='none').mean()
        loss = action_loss + speed_loss + value_loss + feature_loss + wp_loss+ future_feature_loss + future_action_loss
        with torch.no_grad():
            fb_error_mean = torch.abs(pred['pred_wp'][:,:,0] - gt_waypoints[:,:,0]).mean()
            lr_error_mean = torch.abs(pred['pred_wp'][:,:,1] - gt_waypoints[:,:,1]).mean()
        self.log('train_action_loss', action_loss.item())
        self.log('train_wp_loss_loss', wp_loss.item())
        self.log('train_speed_loss', speed_loss.item())
        self.log('train_value_loss', value_loss.item())
        self.log('train_feature_loss', feature_loss.item())
        self.log('train_future_feature_loss', future_feature_loss.item())
        self.log('train_future_action_loss', future_action_loss.item())
        self.log('left_right_error_mean', lr_error_mean.item())
        self.log('front_back_error_mean', fb_error_mean.item())
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-7)
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, 30, 0.5)
        return [optimizer], [lr_scheduler]

    def validation_step(self, batch, batch_idx):
        front_img = batch['front_img']
        speed = batch['speed'].to(dtype=torch.float32).view(-1,1) / 12.
        target_point = batch['target_point'].to(dtype=torch.float32)
        command = batch['target_command']
        state = torch.cat([speed, target_point, command], 1)
        value = batch['value'].view(-1,1)
        feature = batch['feature']
        gt_waypoints = batch['waypoints']

        pred = self.model(front_img, state, target_point)
        action_loss = F.nll_loss(F.log_softmax(pred['action_index'], dim=1), batch['action_index'])
        speed_loss = F.l1_loss(pred['pred_speed'], speed) * self.config.speed_weight
        value_loss = (F.mse_loss(pred['pred_value_traj'], value) + F.mse_loss(pred['pred_value_ctrl'], value)) * self.config.value_weight
        feature_loss = (F.mse_loss(pred['pred_features_traj'], feature) +F.mse_loss(pred['pred_features_ctrl'], feature))* self.config.features_weight
        wp_loss = F.l1_loss(pred['pred_wp'], gt_waypoints, reduction='none').mean()
        with torch.no_grad():
            fb_error_mean = torch.abs(pred['pred_wp'][:,:,0] - gt_waypoints[:,:,0]).mean()
            lr_error_mean = torch.abs(pred['pred_wp'][:,:,1] - gt_waypoints[:,:,1]).mean()

            _, predicted_indices = torch.max(F.log_softmax(pred['action_index'], dim=1), 1)
            correct = (predicted_indices == batch['action_index']).float()
            accuracy = correct.sum() / len(correct) 

        B = batch['action_index'].shape[0]
        batch_steer_l1 = 0 
        batch_brake_l1 = 0
        batch_throttle_l1 = 0
        for i in range(B):
            throttle, steer, brake = self.model.get_action(pred['action_index'][i])
            batch_throttle_l1 += torch.abs(throttle-batch['action'][i][0])
            batch_steer_l1 += torch.abs(steer-batch['action'][i][1])
            batch_brake_l1 += torch.abs(brake-batch['action'][i][2])

        batch_throttle_l1 /= B
        batch_steer_l1 /= B
        batch_brake_l1 /= B

        future_feature_loss = 0
        future_action_loss = 0
        for i in range(self.config.pred_len-1):
            future_action_loss += F.nll_loss(F.log_softmax(pred['future_action_index'][i], dim=1), batch['future_action_index'][i])
            future_feature_loss += F.mse_loss(pred['future_feature'][i], batch['future_feature'][i]) * self.config.features_weight
        future_feature_loss /= self.config.pred_len
        future_action_loss /= self.config.pred_len

        val_loss = wp_loss + batch_throttle_l1+5*batch_steer_l1+batch_brake_l1

        self.log("val_action_loss", action_loss.item(), sync_dist=True)
        self.log('val_speed_loss', speed_loss.item(), sync_dist=True)
        self.log('val_value_loss', value_loss.item(), sync_dist=True)
        self.log('val_feature_loss', feature_loss.item(), sync_dist=True)
        self.log('val_wp_loss_loss', wp_loss.item(), sync_dist=True)
        self.log('val_future_feature_loss', future_feature_loss.item(), sync_dist=True)
        self.log('val_future_action_loss', future_action_loss.item(), sync_dist=True)
        self.log('val_loss', val_loss.item(), sync_dist=True)
        # add
        self.log('lr_error_mean', lr_error_mean.item(), sync_dist=True)
        self.log('fb_error_mean', fb_error_mean.item(), sync_dist=True)
        self.log('current_acc', accuracy.item(), sync_dist=True)
        self.log('current_throttle_error', batch_throttle_l1.item(), sync_dist=True)
        self.log('current_steer_error', batch_steer_l1.item(), sync_dist=True)
        self.log('current_brake_error', batch_brake_l1.item(), sync_dist=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--id', type=str, default='TCP', help='Unique experiment identifier.')
    parser.add_argument('--epochs', type=int, default=27, help='Number of train epochs.')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate.')
    parser.add_argument('--val_every', type=int, default=2, help='Validation frequency (epochs).')
    parser.add_argument('--batch_size', type=int, default=300, help='Batch size')
    parser.add_argument('--logdir', type=str, default='log', help='Directory to log data to.')
    parser.add_argument('--gpus', type=int, default=1, help='number of gpus')

    args = parser.parse_args()
    args.logdir = os.path.join(args.logdir, args.id)

    # Config
    config = GlobalConfig()

    # Data
    train_set = CARLA_Data(root=config.root_dir_all, data_path=config.train_data, img_aug = config.img_aug)
    print(len(train_set))
    val_set = CARLA_Data(root=config.root_dir_all, data_path=config.val_data,)
    print(len(val_set))

    dataloader_train = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=16)
    dataloader_val = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=16)

    TCP_model = TCP_planner(config, args.lr)

    checkpoint_callback = ModelCheckpoint(save_weights_only=False, mode="min", monitor="val_loss", save_top_k=30, save_last=True,
                                            dirpath=args.logdir, filename="best_{epoch:02d}-{val_loss:.3f}")
    trainer = pl.Trainer(
                        default_root_dir=args.logdir,
                        devices = args.gpus,
                        accelerator='gpu',
                        strategy=DDPStrategy(static_graph=True),
                        sync_batchnorm=True,
                        profiler='simple',
                        benchmark=True,
                        log_every_n_steps=1,
                        callbacks=[checkpoint_callback,
                                    ],
                        check_val_every_n_epoch = args.val_every,
                        max_epochs = args.epochs,
                        )

    trainer.fit(TCP_model, dataloader_train, dataloader_val)
    