<!-- # Event-T2M: Event-level Conditioning for Complex Text-to-Motion Synthesis -->

# Event-T2M: Event-level Conditioning for Complex Text-to-Motion Synthesis (ICLR 2026)

The official PyTorch implementation of the paper "Event-T2M: Event-level Conditioning for Complex Text-to-Motion Synthesis".

## Setting an environment

<details>

### 1. Create Conda Environment

```bash
conda create -n event-t2m python==3.10.14
conda activate event-t2m

# install pytorch
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121


# install requirements
pip install -r requirements.txt
```

### 2. Download the original datasets

We conduct experiments on the HumanML3D and KIT-ML datasets. For both datasets, you can download them by following the instructions in [here](https://github.com/EricGuo5513/HumanML3D).

### 3. Preprocessing for HumanML3D-E

You can download the completed HumanML3D-E dataset from [here](https://drive.google.com/drive/folders/19mPyYV8j1vnfJ6W9tZX9758JtDUQpYop?usp=sharing).

If you want to preprocess the dataset, follow the steps below:

Since an LLM (Gemini 2.5 flash) was used for HumanML3D-E data preprocessing, an API key is required.
Please enter the issued API key on line 6 of `src/tools/data_decompose.py`.
```bash
GOOGLE_API_KEY = "" # your api key here
```

- For processing,
```bash
python src/tools/data_decompose.py
```

### 4. Setting up the TMR encoder

### 5. Preparing for training

```bash
python src/tools/data_preprocess_decomposed.py --dataset hml3d
python src/tools/data_preprocess_decomposed.py --dataset kit
```

### 6. Download Dependencies
Download and unzip dependencies from [here](https://onedrive.live.com/?id=76593CF7B7FC849C%21180700&resid=76593CF7B7FC849C%21180700&e=345HR5&migratedtospo=true&redeem=aHR0cHM6Ly8xZHJ2Lm1zL3UvcyFBcHlFX0xmM1BGbDJpNE5jRThtZ1ZVTjNvWDluVFE_ZT0zNDVIUjU&cid=76593cf7b7fc849c&v=validatepermission).

</details>

## Training

<details>

- For HumanML3D
```bash
python src/train.py trainer.devices=\"0,1\" logger=wandb data=hml3d_event_final \
    data.batch_size=128 data.repeat_dataset=5 trainer.max_epochs=600 \
    callbacks/model_checkpoint=t2m +model/lr_scheduler=cosine model.guidance_scale=4\
    model.noise_scheduler.prediction_type=sample trainer.precision=bf16-mixed
```

- For KIT-ML
```bash
python src/train.py trainer.devices=\"2,3\" logger=wandb data=kit_event_final \
    data.batch_size=128 data.repeat_dataset=5 trainer.max_epochs=1000 \
    callbacks/model_checkpoint=t2m +model/lr_scheduler=cosine model.guidance_scale=4\
    model.noise_scheduler.prediction_type=sample trainer.precision=bf16-mixed
```
</details>

## Evaluation

<details>

Set `model.metrics.enable_mm_metric` to `True` to evaluate Multimodality. Setting `model.metrics.enable_mm_metric` to `False` an speed up the evaluation.

```bash
python src/eval.py trainer.devices=\"0,\" data=hml3d_event_final data.test_batch_size=128 \
    model=event_final  \
    model.guidance_scale=4 model.noise_scheduler.prediction_type=sample\
    model.denoiser.stage_dim=\"256\*4\" \
    ckpt_path=\"checkpoints/hml3d.ckpt\" model.metrics.enable_mm_metric=true
```

</details>
