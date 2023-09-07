import copy

## --------------------  Most frequently changed params here  --------------------

# Resume from last saved checkpoint
resume_training_from_last = True

max_steps = 100000
batch_size = 8

num_gpus = 1
num_workers_per_gpu = 7

## --------------------  Model --------------------

# --- Denoiser ---
denoiser_model = dict(
    dim=16,
    init_dim=None,
    out_dim=None,
    dim_mults=(1, 2, 4, 8),
    channels=1,
    self_condition=False,
    resnet_block_groups=8,
    learned_variance=False,
    learned_sinusoidal_cond=False,
    random_fourier_features=False,
    learned_sinusoidal_dim=16,
    sinusoidal_pos_emb_theta=10000,
    attn_dim_head=32,
    attn_heads=4,
    full_attn=(False, False, False, True),
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
        noise_scheduler_type="ddpm",
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

train_data = dict(
    type="ToDoDataset",
    args=dict(
        data_root_dir=root_data_dir,
    ),
    batch_size=batch_size,
)

val_data = copy.deepcopy(train_data)
val_data["args"]["split"] = "test"
val_data["batch_size"] = 4

data = dict(
    train=train_data,
    val=val_data,
)

## --------------------  Trainer  --------------------

# --- Logger ---
logger = dict(type="WandbLogger", project="debug-project")

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
)