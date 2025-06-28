# Grasping
# The video for the real-world robotic grasping experiments is available at https://youtu.be/S-QHBtbmLc4.
# The environment setup and user guide are coming soon.
# Pre-trained models can be downloaded at：https://pan.baidu.com/s/1oYZEFiYA0mllFEBfxSy7bQ password: t1fp 
### Table: Key Implementation Details

| Item                    | Description                        |
|-------------------------|------------------------------------|
| Input Image Size        | 224x224                            |
| Data Augmentation       | Random Rotation & Zoom             |
| Epochs                  | 100                                |
| Optimizer               | SGD                                |
| Weight Decay            | 1e-4                               |
| Initial Learning Rate   | 0.001                              |
| Learning Rate Schedules | Cosine Annealing                   |
| PyTorch Version         | 2.0.1                              |
| Batch Sizes             | 16                                 |
| CUDA                    | 11.8                               |
| System                  | Ubuntu 20.04.6                     |
| CPU                     | Intel Xeon Gold 6146               |
| GPU                     | NVIDIA GeForce RTX 4090            |
| Camera                  | Realsense D435i                    |
| Manipulator             | AUBO-i5                            |
