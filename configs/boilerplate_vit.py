import copy

## --------------------  Most frequently changed params here  --------------------

# Resume from last saved checkpoint
resume_training_from_last = True

seed = 1729

max_steps = 150000
batch_size = 1

num_gpus = 1
num_workers_per_gpu = 4

img_C = 1
img_H = 512
img_W = 512

## --------------------  Model --------------------

# --- Denoiser ---
denoiser_model = dict(
    type="UViT",
    args=dict(
        dim=16,
        init_dim=None,
        out_dim=None,
        dim_mults=(1, 2, 4, 8),
        downsample_factor=2,
        channels=1,
        vit_depth=6,
        vit_dropout=0.2,
        attn_dim_head=32,
        attn_heads=4,
        ff_mult=4,
        resnet_block_groups=8,
        learned_sinusoidal_dim=16,
        init_img_transform=None,
        final_img_itransform=None,
        patch_size=1,
        dual_patchnorm=False,
    ),
)
# --- Diffuser ---
model = dict(
    type="DEMDiffuser",
    args=dict(
        model=denoiser_model,
        in_channels=1,
        in_dims=(img_H, img_W),
        diffusion_timesteps=1000,
        diffusion_loss="l2",
        beta_schedule="cosine",
        # Using DDIM scheduler allows val loop to run fast in generation mode
        # While training is still with DDPM 1000 steps
        noise_scheduler_type="ddim",
        is_conditioned=False,
        denoising_loss_weight=1,
        variance_type="fixed_small",
        beta_start=0.0001,
        beta_end=0.02,
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
        out_shape=(img_C, img_H, img_W),
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
    deterministic=False,
)
