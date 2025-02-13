# uniHAR 🚀

We introduce **uniHAR**, an LLM-assisted universal HAR system designed to achieve "one model to fit them all"—just one model that can adapt to diverse HAR datasets without any dataset-specific operation.
In particular, we propose **Cross-Dataset Neural Network (CDNet)** for the “one model”, which models both the temporal context and spatial relationships of IMU data to capture cross-dataset representations, encompassing differences in device, participant, data collection position and environment, etc.
Additionally, we introduce LLM-driven data synthesis, which enhances the training process by generating virtual IMU data through three carefully designed strategies. Furthermore, LLM-assisted adaptive position processing optimizes the inference process by handling variable number and combination of positional inputs.

<div align="center">
  <img src="https://github.com/user-attachments/assets/c2512ea0-0f11-4114-a2c1-b4432376d20f" width="800"/>
</div>


## 📌 Installation

1. **Clone the repo into a local folder.**

```bash
git clone https://github.com/ZJU-wqx-2025/uniHAR.git
cd uniHAR
```

2. **Create and activate the conda environment.**

```bash
conda create -n uniHAR python=3.12 -y
conda activate uniHAR
```

3. **Install the required packages.**

```bash
pip install torch==2.3.0+cu121 -f https://download.pytorch.org/whl/torch_stable.html
pip install transformers
pip install seaborn==0.10.1
pip install pandas==1.0.5
pip install scipy==1.5.0
pip install matplotlib==3.2.2
pip install numpy==1.18.5
pip install scikit-learn==1.0
```


## 📂 Dataset

The test folder contains the test sets for five widely used datasets in previous HAR studies:

- HHAR
- Pamap2
- RealworldHAR
- USC-HAD
- UCI-HAR

-**Data**: A numpy array of shape `(B * N * T * C)`, where:
- **B** is the batch size
- **N** is the number of sensors (located at different positions)
- **T** is the number of time steps
- **C** is the number of measurement channels (e.g., accelerometer x, y, z axes)

-**Label**: A numpy array of shape `(B * T * 1)`, containing the corresponding labels for the data.

The detailed label information is provided in the `dataset/data_config.json`.

Each dataset folder contains its respective `"data_X_Y.npy"` and `"label_X_Y.npy"`, where X represents the sampling rate and Y represents the sequence length.


## 🛠 Usage

1. **Set environment variables.**
```bash
export HF_ENDPOINT=https://hf-mirror.com
```
2. **Execute test command.**

You can run the `test.py` script with the following command:
```bash
python -u test.py v1 HHAR 50_120 -f hhar
```
For this command, it will first load the pretrained LIMU-BERT from the file "hhar.pt" in the `saved/pretrain_base_hhar_20_120` folder as our IMU encoder, and it will also load the pretrained parameters of 'bert-base-uncased' into our label encoder. Finally, it will run the `test.py` file to perform classification predictions on the five test datasets.


Command-line Arguments:
```bash

usage: xxx.py [-h] [-g GPU] [-f MODEL_FILE] [-t TRAIN_CFG] [-a MASK_CFG]
                   model_version {HHAR} {50_120}

positional arguments:
  model_version         Model config, e.g. v1
  {HHAR}
                        Dataset name
  {50_120}              Dataset version

optional arguments:
  -h, --help            show this help message and exit
  -g GPU, --gpu GPU     Set specific GPU
  -f MODEL_FILE, --model_file MODEL_FILE
                        Pretrain model file, default: None
  -t TRAIN_CFG, --train_cfg TRAIN_CFG
                        Training config json file path
  -a MASK_CFG, --mask_cfg MASK_CFG
                        Mask strategy json file path, default: config/mask.json
```
