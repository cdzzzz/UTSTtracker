
## Installation
### Requirements

* Install pytorch using conda (optional)

    ```bash
    conda create -n 'yourconda' python=3.7
    conda activate 'yourconda'
    conda install pytorch=1.8.1 torchvision=0.9.1 cudatoolkit=10.2 -c pytorch
    ```

* Other requirements
    ```bash
    pip install -r requirements.txt
    ```

* Build MultiScaleDeformableAttention
    ```bash
    cd ./models/ops
    sh ./make.sh
    ```

## Usage

### Dataset preparation
Please prepare dataset as follows:
```
/data/Dataset
├── yourdata
│   ├── train
│   └── test
```
You need to ensure that the data format is consistent with the MOT format
### Training
You may train your own weights or simply use pretrianed weights of MOTRv2 as following:
```bash 
./tools/train.sh configs/utst.args
```
Our pretrained weights will be released after acceptance.
### Inference
Using the trained weight for inference.
```bash
# run a simple inference on your pretrained weights
./tools/simple_inference.sh ./yourweights.pth
```



