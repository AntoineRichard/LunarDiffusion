import copy

## --------------------  Most frequently changed params here  --------------------

# Resume from last saved checkpoint
resume_training_from_last = True

seed = 1729

max_steps = 150000
batch_size = 2

num_gpus = 1
num_workers_per_gpu = 8

## --------------------  Model --------------------

# --- Denoiser ---
denoiser_model = dict(
    type="SimpleConditionalUnet2D",
    args=dict(
        dim=16,
        init_dim=None,
        out_channels=None,
        block_channels=(16, 64, 256, 256, 64, 16),
        channels=1,
        input_conditioning_dims=None,
        is_self_conditioned=False,
        resnet_block_groups=8,
        learned_variance=False,
        dropout=None,
        is_time_conditioned=True,
        learned_sinusoidal_cond=False,
        random_fourier_features=False,
        learned_sinusoidal_dim=16,
    ),
)
# --- Diffuser ---
model = dict(
    type="DEMDiffuser",
    args=dict(
        model=denoiser_model,
        in_channels=1,
        in_dims=(256, 256),
        diffusion_timesteps=1000,
        diffusion_loss="l2",
        beta_schedule="linear",
        # Using DDIM scheduler allows val loop to run fast in generation mode
        # While training is still with DDPM 1000 steps
        noise_scheduler_type="ddim",
        is_conditioned=True,
        denoising_loss_weight=1,
        variance_type="fixed_small",
        beta_start=5e-5,
        beta_end=5e-2,
    ),
)

## --------------------  Data --------------------

augmentation_config = [dict()]

root_data_dir = "/mnt/irisgpfs/projects/mis-urso/grasp/data/acronym/renders/objects_filtered_grasps_63cat_8k/"

# Train split
train_data = dict(
    type="LunarSLDEMDataset",
    args=dict(
        data_root="data/processed",
        data_split="train",
        prefix="MoonORTO2DEM",
        num_repeat_dataset=2,
    ),
    batch_size=batch_size,
)

# Val split
val_data = copy.deepcopy(train_data)
val_data["args"]["data_split"] = "val"
val_data["args"]["num_repeat_dataset"] = 1
val_data["batch_size"] = 4

data = dict(
    train=train_data,
    val=val_data,
)

## --------------------  Trainer  --------------------

# --- Logger ---
logger = dict(type="WandbLogger", project="lunar_diffusion")
# logger = dict(type="TensorBoardLogger")
# logger = dict(type="CSVLogger")

# --- Optimizer ---
optimizer = dict(
    initial_lr=0.001,
    scheduler=dict(
        type="MultiStepLR",
        args=dict(
            milestones=[
                int(max_steps / 4),
                int(2 * max_steps / 4),
                int(3 * max_steps / 4),
            ],
            gamma=0.1,
        ),
        # args=dict(milestones=list(range(100)), gamma=0.9),
    ),
)

trainer = dict(
    max_steps=max_steps,
    batch_size=batch_size,
    num_workers=num_workers_per_gpu * num_gpus,
    accelerator="gpu",
    devices=num_gpus,
    strategy="ddp",
    logger=logger,
    log_every_n_steps=10,
    optimizer=optimizer,
    resume_training_from_last=resume_training_from_last,
    check_val_every_n_epoch=1,
    # use_compile=True,
    deterministic=True,
)
