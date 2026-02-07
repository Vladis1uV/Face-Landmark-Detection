# Face Landmark Detection

Train and evaluate a 68-point facial landmark detector on the 300W (iBUG) dataset using PyTorch.

## Overview
- Model: ResNet-18 backbone with a regression head for 68 landmark points.
- Input: Cropped face image resized to 224x224, normalized to [-0.5, 0.5].
- Output: 68 (x, y) landmark coordinates in normalized space.
- Evaluation: NME (inter-ocular), RMSE/MAE in pixels, AUC@0.08, failure rate.

## Project Structure
- `src/models/` - model definitions
- `src/datamodules/` - dataset, transforms, training, evaluation
- `data/` - dataset root (not committed)
- `checkpoints/` - saved weights (not committed)
- `runs/` - tensorboard logs and eval visuals (not committed)

## Requirements
Install Python dependencies:
```bash
python3 -m pip install -r requirements.txt
```

## Dataset
This project uses the 300W (iBUG) dataset. Expected layout:
```
data/
└── ibug_300W_large_face_landmark_dataset/
    ├── labels_ibug_300W_train.xml
    ├── images/
    └── ...
```

The dataset is not included in this repository.

## Training
```bash
python3 -m src.datamodules.train
```

Weights are saved to `checkpoints/best.pth`.

## Evaluation
```bash
python3 -m src.datamodules.eval --ckpt checkpoints/best.pth --viz 8
```

This prints metrics and saves visualization to `runs/eval_viz/eval_samples.png`.

## Visual Results
After running evaluation with `--viz`, open:
- `runs/eval_viz/eval_samples.png`

Red points are model predictions and green points are ground-truth landmarks.

## Notes
- If you are on CPU, `pin_memory` warnings are safe to ignore.
- If `numpy` is missing, install requirements as shown above.

## License
See `LICENSE`.
