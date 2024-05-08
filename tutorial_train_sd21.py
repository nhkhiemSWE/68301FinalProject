from share import *

import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from tutorial_dataset import MyDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
from pytorch_lightning.callbacks import ModelCheckpoint

# Configs
resume_path = './models/control_sd21_ini.ckpt'
batch_size = 5
logger_freq = 300
learning_rate = 1e-5
sd_locked = True
only_mid_control = False


# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
model = create_model('./models/cldm_v21.yaml').cpu()
model.load_state_dict(load_state_dict(resume_path, location='cpu'))
model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control

#Checkpoint:
# Configure the ModelCheckpoint to save a checkpoint after every epoch
# checkpoint_callback = ModelCheckpoint(
#     dirpath='models/',  # Directory where checkpoints will be saved
#     filename='checkpoint-{epoch:02d}-{val_loss:.2f}',  # Naming convention using epoch and validation loss
#     save_top_k=-1,  # Set to -1 to save all checkpoints; use an integer to limit the number
#     every_n_epochs=10,  # Save checkpoints at every epoch
#     monitor='val_loss',  # Metric to monitor for performance (only necessary if using save_top_k with value other than -1)
#     mode='min',  # Determines if the monitored value should be minimized or maximized ('min' for minimum)
#     verbose=True
# )

# Misc
dataset = MyDataset()
dataloader = DataLoader(dataset, num_workers=0, batch_size=batch_size, shuffle=True)
# # Calculate split sizes
# train_size = int(0.8 * len(dataset))  # 80% of the dataset for training
# val_size = len(dataset) - train_size  # Remaining 20% for validation

# # Split the dataset
# train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# # Create data loaders
# train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
logger = ImageLogger(batch_frequency=logger_freq)
trainer = pl.Trainer(accelerator="gpu", max_epochs=7, precision=32, callbacks=[logger])

# Train the model
trainer.fit(model, dataloader)
