#!/bin/bash
# Optional fallback for HF accounts that cannot build their own Docker Space.
# Uses the leaderboard's public scorer image, which already has torch 2.8/cu128.
set -e -o pipefail
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq --no-install-recommends build-essential cmake
python -m pip install --no-cache-dir \
    numpy==1.26.4 soundfile==0.13.1 safetensors==0.6.2 \
    huggingface_hub==1.33.0 datasets==3.6.0 pyarrow==20.0.0 pandas==2.2.3 \
    librosa==0.11.0 tqdm==4.67.1 pypinyin==0.55.0 jieba==0.42.1 \
    eng_to_ipa==0.0.2 Unidecode==1.4.0 inflect==7.5.0 \
    pyopenjtalk==0.4.1 einops==0.8.1 scipy==1.16.2
git clone -q https://github.com/ydqmkkx/GibbsTTS.git /opt/GibbsTTS
git -C /opt/GibbsTTS checkout -q 2af61715fe832b8dbef909b545090a56056428a3
export PYTHONPATH=/opt/GibbsTTS${PYTHONPATH:+:${PYTHONPATH}}
python -c 'import torch; from config import ModelConfig; from models import GibbsTTS; assert torch.__version__.split("+")[0] == "2.8.0"; assert torch.cuda.is_available(); print("Bootstrap ready:", torch.__version__, torch.cuda.get_device_name(0))'
