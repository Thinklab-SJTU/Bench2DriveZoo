import argparse
import os
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy
from ADMLP.model import ADMLP
from ADMLP.data import CARLA_Data
from ADMLP.config import GlobalConfig

class ADMLP_planner(pl.LightningModule):
	def __init__(self):
		super().__init__()
		self.model = ADMLP()
	
	def forward(self, batch):
		pass

	def training_step(self, batch, batch_idx):
		predict = self.model(batch)
		waypoints = batch['waypoints']
		theta = batch['thetas']

		x_loss =  F.l1_loss(predict[:,:,0], waypoints[:,:,0], reduction='mean')
		y_loss =  F.l1_loss(predict[:,:,1], waypoints[:,:,1], reduction='mean')
		theta  =  F.l1_loss(predict[:,:,2], theta, reduction='mean')

		loss = x_loss + y_loss + theta
		self.log('train_loss', loss.item())
		return loss

	def validation_step(self, batch, batch_idx):
		predict = self.model(batch)
		waypoints = batch['waypoints']
		theta = batch['thetas']

		x_loss =  F.l1_loss(predict[:,:,0], waypoints[:,:,0])
		y_loss =  F.l1_loss(predict[:,:,1], waypoints[:,:,1])
		theta  =  F.l1_loss(predict[:,:,2], theta)
		
		loss = x_loss + y_loss + theta
		self.log('val_loss', loss.item(), sync_dist=True)
	
	def configure_optimizers(self):
		optimizer = optim.AdamW(self.parameters(), lr=4e-6, weight_decay=1e-2)
		lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer,[2,4],gamma=0.2)
		return [optimizer], [lr_scheduler]

if __name__ == "__main__":
	parser = argparse.ArgumentParser()

	parser.add_argument('--id', type=str, default='ADMLP', help='Unique experiment identifier.')
	parser.add_argument('--epochs', type=int, default=60, help='Number of train epochs.')
	parser.add_argument('--val_every', type=int, default=3, help='Validation frequency (epochs).')
	parser.add_argument('--batch_size', type=int, default=3200, help='Batch size')
	parser.add_argument('--logdir', type=str, default='log', help='Directory to log data to.')
	parser.add_argument('--gpus', type=int, default=1, help='number of gpus')

	args = parser.parse_args()
	args.logdir = os.path.join(args.logdir, args.id)

	# Config
	config = GlobalConfig()

	# Data
	train_set = CARLA_Data(data_path=config.train_data)
	print(len(train_set))
	val_set = CARLA_Data(data_path=config.val_data)
	print(len(val_set))

	dataloader_train = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=8)
	dataloader_val = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=8)

	ADMLP_model = ADMLP_planner()

	checkpoint_callback = ModelCheckpoint(save_weights_only=False, mode="min", monitor="val_loss", save_top_k=10, save_last=True,
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

	trainer.fit(ADMLP_model, dataloader_train, dataloader_val)
	