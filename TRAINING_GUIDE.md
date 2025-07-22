# OpenCLIP 훈련 가이드

이 가이드는 `open_clip_train/main.py`를 사용하여 CLIP 모델을 훈련하는 방법을 상세히 설명합니다.

## 목차

1. [환경 설정](#환경-설정)
2. [데이터 준비](#데이터-준비)  
3. [기본 훈련](#기본-훈련)
4. [고급 설정](#고급-설정)
5. [분산 훈련](#분산-훈련)
6. [모니터링 및 로깅](#모니터링-및-로깅)
7. [체크포인트 관리](#체크포인트-관리)
8. [성능 최적화](#성능-최적화)
9. [문제 해결](#문제-해결)

## 환경 설정

### 1. 가상환경 생성

```bash
python3 -m venv .env
source .env/bin/activate
pip install -U pip
```

### 2. 의존성 설치

```bash
# 기본 훈련 의존성 설치
pip install 'open_clip_torch[training]'

# 또는 개발용으로 소스에서 설치
git clone https://github.com/mlfoundations/open_clip.git
cd open_clip
make install
make install-training
```

### 3. 시스템 요구사항

- **GPU**: CUDA 지원 GPU (최소 8GB VRAM 권장)
- **메모리**: 최소 32GB RAM (대용량 데이터셋의 경우)
- **저장공간**: 데이터셋과 체크포인트를 위한 충분한 공간

## 데이터 준비

### 지원되는 데이터셋 형식

1. **CSV 형식**: 이미지 경로와 캡션이 포함된 CSV 파일
2. **WebDataset**: tar 파일들로 구성된 대규모 데이터셋
3. **Synthetic**: 테스트용 합성 데이터

### 1. CSV 데이터셋 준비

```bash
# CSV 파일 구조 예시
# filepath\ttitle
# /path/to/image1.jpg\t"A cat sitting on a chair"
# /path/to/image2.jpg\t"A dog playing in the park"
```

### 2. WebDataset 준비

WebDataset은 대규모 데이터셋에 권장되는 형식입니다:

```bash
# tar 파일 구조
# dataset_000.tar
#   ├── sample_001.jpg
#   ├── sample_001.txt
#   ├── sample_002.jpg
#   └── sample_002.txt
```

## 기본 훈련

### 최소 설정으로 훈련 시작

```bash
python -m open_clip_train.main \
    --train-data="/path/to/train_data.csv" \
    --val-data="/path/to/validation_data.csv" \
    --csv-img-key filepath \
    --csv-caption-key title \
    --model ViT-B-32 \
    --batch-size 64 \
    --lr 1e-3 \
    --epochs 10 \
    --workers 4 \
    --logs ./logs/
```

### 주요 파라미터 설명

| 파라미터 | 설명 | 기본값 |
|----------|------|--------|
| `--model` | 사용할 모델 아키텍처 | RN50 |
| `--batch-size` | GPU당 배치 크기 | 64 |
| `--lr` | 학습률 | 모델별 기본값 |
| `--epochs` | 전체 에포크 수 | 32 |
| `--workers` | 데이터 로더 워커 수 | 4 |
| `--precision` | 정밀도 모드 | amp |

## 고급 설정

### 1. 모델 선택

```bash
# Vision Transformer 모델들
--model ViT-B-32      # ViT-Base/32
--model ViT-B-16      # ViT-Base/16
--model ViT-L-14      # ViT-Large/14

# ResNet 모델들
--model RN50          # ResNet-50
--model RN101         # ResNet-101

# ConvNext 모델들
--model convnext_base # ConvNext-Base
```

### 2. 사전훈련된 모델에서 시작

```bash
python -m open_clip_train.main \
    --model ViT-B-32 \
    --pretrained laion2b_s34b_b79k \
    --train-data="/path/to/data.csv" \
    --lr 1e-4 \
    --epochs 5
```

### 3. 이미지 및 텍스트 타워 고정

```bash
# 이미지 인코더 일부 고정 (마지막 n개 레이어만 학습)
--lock-image \
--lock-image-unlocked-groups 2

# 텍스트 인코더 일부 고정 (마지막 n개 레이어만 학습)  
--lock-text \
--lock-text-unlocked-layers 2
```

### 4. 데이터 증강 설정

```bash
--aug-cfg scale='(0.9, 1.0)' \
--aug-cfg color_jitter='(0.4, 0.4, 0.4, 0.1)' \
--aug-cfg color_jitter_prob=0.8 \
--aug-cfg gray_scale_prob=0.2
```

## 분산 훈련

### 1. 단일 노드 멀티 GPU

```bash
# 4개 GPU 사용
cd src
torchrun --nproc_per_node 4 -m open_clip_train.main \
    --train-data '/path/to/dataset.tar' \
    --dataset-type webdataset \
    --batch-size 320 \
    --precision amp \
    --workers 4 \
    --model ViT-B-32
```

### 2. 멀티 노드 훈련

```bash
# 마스터 노드에서
torchrun --nproc_per_node=8 \
    --nnodes=4 \
    --node_rank=0 \
    --master_addr="192.168.1.1" \
    --master_port=12355 \
    -m open_clip_train.main \
    --train-data '/path/to/dataset.tar' \
    --dataset-type webdataset \
    --batch-size 256 \
    --model ViT-B-32
```

### 3. SLURM 클러스터에서 훈련

```bash
#!/bin/bash
#SBATCH --nodes=8
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=6

eval "$(/path/to/conda/bin/conda shell.bash hook)"
conda activate open_clip

export CUDA_VISIBLE_DEVICES=0,1,2,3
export MASTER_PORT=12802

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr

cd /path/to/open_clip
export PYTHONPATH="$PYTHONPATH:$PWD/src"

srun python -u src/open_clip_train/main.py \
    --save-frequency 1 \
    --train-data="/data/dataset/{00000..99999}.tar" \
    --dataset-type webdataset \
    --warmup 2000 \
    --batch-size=256 \
    --epochs=32 \
    --workers=8 \
    --model ViT-B-32 \
    --precision amp \
    --local-loss \
    --gather-with-grad
```

## 모니터링 및 로깅

### 1. TensorBoard 로깅

```bash
python -m open_clip_train.main \
    --report-to tensorboard \
    --logs ./logs/ \
    --train-data="/path/to/data.csv"

# 별도 터미널에서 TensorBoard 실행
tensorboard --logdir=logs/tensorboard/ --port=7777
```

### 2. Weights & Biases 로깅

```bash
python -m open_clip_train.main \
    --report-to wandb \
    --wandb-project-name "my-clip-training" \
    --wandb-notes "Initial CLIP training experiment" \
    --train-data="/path/to/data.csv"
```

### 3. 둘 다 사용

```bash
--report-to wandb,tensorboard
```

## 체크포인트 관리

### 1. 체크포인트 저장 설정

```bash
python -m open_clip_train.main \
    --save-frequency 2 \         # 2 에포크마다 저장
    --save-most-recent \         # 최신 체크포인트 항상 저장
    --delete-previous-checkpoint \ # 이전 체크포인트 삭제
    --train-data="/path/to/data.csv"
```

### 2. 훈련 재시작

```bash
python -m open_clip_train.main \
    --resume /path/to/checkpoints/epoch_10.pt \
    --train-data="/path/to/data.csv"

# 또는 최신 체크포인트에서 자동 재시작
python -m open_clip_train.main \
    --resume latest \
    --train-data="/path/to/data.csv"
```

### 3. 원격 저장소 동기화

```bash
python -m open_clip_train.main \
    --remote-sync s3://my-bucket/checkpoints \
    --remote-sync-frequency 300 \  # 5분마다 동기화
    --train-data="/path/to/data.csv"
```

## 성능 최적화

### 1. 메모리 최적화

```bash
python -m open_clip_train.main \
    --precision amp \              # 자동 혼합 정밀도
    --grad-checkpointing \         # 그래디언트 체크포인팅
    --local-loss \                 # 로컬 손실 계산
    --gather-with-grad \           # 그래디언트와 함께 수집
    --train-data="/path/to/data.csv"
```

### 2. 대용량 배치 시뮬레이션

```bash
python -m open_clip_train.main \
    --batch-size 64 \
    --accum-freq 4 \              # 4번 누적하여 배치크기 256 효과
    --train-data="/path/to/data.csv"
```

### 3. 패치 드롭아웃 (훈련 가속화)

```bash
python -m open_clip_train.main \
    --force-patch-dropout 0.5 \   # 50% 패치 드롭아웃
    --model ViT-B-16 \
    --train-data="/path/to/data.csv"
```

### 4. 모델 컴파일 (PyTorch 2.0+)

```bash
python -m open_clip_train.main \
    --torchcompile \
    --train-data="/path/to/data.csv"
```

## 평가 및 제로샷 테스트

### 1. ImageNet 제로샷 평가

```bash
python -m open_clip_train.main \
    --imagenet-val /path/to/imagenet/val \
    --zeroshot-frequency 2 \      # 2 에포크마다 평가
    --train-data="/path/to/data.csv"
```

### 2. 훈련 없이 평가만 수행

```bash
python -m open_clip_train.main \
    --imagenet-val /path/to/imagenet/val \
    --model ViT-B-32 \
    --pretrained laion2b_s34b_b79k
```

## 특수 모델 훈련

### 1. CoCa 모델 훈련

```bash
python -m open_clip_train.main \
    --model coca_ViT-B-32 \
    --coca-contrastive-loss-weight 1.0 \
    --coca-caption-loss-weight 2.0 \
    --train-data="/path/to/data.csv"
```

### 2. SigLIP 손실 함수 사용

```bash
python -m open_clip_train.main \
    --siglip \
    --model ViT-B-32 \
    --train-data="/path/to/data.csv"
```

### 3. 지식 증류

```bash
python -m open_clip_train.main \
    --model ViT-B-32 \
    --distill-model ViT-L-14 \
    --distill-pretrained openai \
    --train-data="/path/to/data.csv"
```

## 실전 예제

### 1. 소규모 실험용 설정

```bash
python -m open_clip_train.main \
    --train-data="./data/small_dataset.csv" \
    --val-data="./data/validation.csv" \
    --csv-img-key filepath \
    --csv-caption-key caption \
    --model ViT-B-32 \
    --batch-size 32 \
    --lr 1e-3 \
    --epochs 5 \
    --workers 2 \
    --precision amp \
    --report-to tensorboard \
    --logs ./logs/small_exp
```

### 2. 대규모 데이터셋 훈련

```bash
torchrun --nproc_per_node 8 -m open_clip_train.main \
    --train-data="/data/laion400m/{00000..41455}.tar" \
    --dataset-type webdataset \
    --train-num-samples 400000000 \
    --val-data="/data/validation/{00000..00099}.tar" \
    --val-num-samples 10000 \
    --model ViT-B-16 \
    --batch-size 128 \
    --lr 5e-4 \
    --beta1 0.9 \
    --beta2 0.98 \
    --eps 1e-6 \
    --wd 0.2 \
    --warmup 10000 \
    --epochs 32 \
    --workers 6 \
    --precision amp \
    --grad-checkpointing \
    --local-loss \
    --gather-with-grad \
    --save-frequency 1 \
    --save-most-recent \
    --delete-previous-checkpoint \
    --zeroshot-frequency 1 \
    --imagenet-val /data/imagenet/val \
    --report-to wandb \
    --wandb-project-name "laion400m-vitb16" \
    --remote-sync s3://my-bucket/checkpoints \
    --logs ./logs/laion400m_training
```

### 3. 파인튜닝 예제

```bash
python -m open_clip_train.main \
    --train-data="./data/domain_specific.csv" \
    --model ViT-B-32 \
    --pretrained laion2b_s34b_b79k \
    --lr 1e-5 \                    # 낮은 학습률
    --epochs 3 \                   # 짧은 에포크
    --batch-size 64 \
    --warmup 500 \
    --lock-image \                 # 이미지 인코더 고정
    --lock-image-unlocked-groups 1 \ # 마지막 1개 그룹만 학습
    --precision amp \
    --report-to tensorboard
```

## 문제 해결

### 1. 메모리 부족 오류

```bash
# 배치 크기 줄이기
--batch-size 32

# 그래디언트 체크포인팅 활성화
--grad-checkpointing

# 로컬 손실 사용
--local-loss --gather-with-grad

# 정밀도 변경
--precision fp16
```

### 2. 학습 속도가 느림

```bash
# 워커 수 증가
--workers 8

# 배치 크기 증가 (GPU 메모리가 허용하는 경우)
--batch-size 128

# 패치 드롭아웃 사용 (ViT 모델의 경우)
--force-patch-dropout 0.5

# 모델 컴파일 사용
--torchcompile
```

### 3. 분산 훈련 문제

```bash
# 네트워크 백엔드 변경
--dist-backend nccl

# 포트 변경
export MASTER_PORT=12355

# 정적 그래프 최적화
--ddp-static-graph
```

### 4. 데이터 로딩 오류

```bash
# CSV 구분자 확인
--csv-separator "\t"

# 이미지/캡션 키 확인
--csv-img-key image_path
--csv-caption-key text

# WebDataset 샘플 수 명시적 지정
--train-num-samples 1000000
```

## 유용한 팁

1. **실험 추적**: 각 실험에 고유한 이름을 부여하세요
   ```bash
   --name "experiment_$(date +%Y%m%d_%H%M%S)"
   ```

2. **디버깅**: 작은 데이터셋으로 먼저 테스트하세요
   ```bash
   --debug --batch-size 4 --epochs 1
   ```

3. **체크포인트 검증**: 훈련 후 체크포인트가 제대로 로드되는지 확인하세요
   ```bash
   python -c "import torch; print(torch.load('epoch_10.pt').keys())"
   ```

4. **모니터링**: 훈련 중 GPU 사용률을 모니터링하세요
   ```bash
   nvidia-smi -l 1
   ```

이 가이드를 참고하여 다양한 규모와 요구사항에 맞는 CLIP 모델을 성공적으로 훈련하시기 바랍니다. 