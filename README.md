# uniHAR 🚀

uniHAR, an LLM-assisted universal HAR system designed to achieve "one model to fit them all"—just one model that can adapt to diverse HAR datasets without any dataset-specific operation.
In particular, we propose Cross-Dataset neural network (CDNet) for the “one model”, which models both the temporal context and spatial relationships of IMU data to capture cross-dataset representations, encompassing differences in device, participant, data collection position and environment, etc.
Additionally, we introduce LLM-driven data synthesis, which enhances the training process by generating virtual IMU data through three carefully designed strategies. Furthermore, LLM-assisted adaptive position processing optimizes the inference process by handling variable number and combination of positional inputs.

## 📌 Installation

This project uses the following libraries and tools：

- Python: 3.12 
- CUDA: 12.1

Necessary Python libraries can be installed manually：
```bash
pip install seaborn==0.10.1
pip install torch==1.5.1
pip install pandas==1.0.5
pip install scipy==1.5.0
pip install matplotlib==3.2.2
pip install numpy==1.18.5
pip install scikit-learn==1.0
pip install torch==2.3.0+cu121 torchvision==0.14.0+cu121 torchaudio==2.0.0 -f https://download.pytorch.org/whl/torch_stable.html

```

## 🛠 Usage

```bash
# 克隆仓库
git clone https://github.com/ZJU-wqx-2025/uniHAR.git

# 进入目录
cd your-repo

# 运行代码
export HF_ENDPOINT=https://hf-mirror.com
python -u test.py v1 HHAR 50_120 -f hhar
