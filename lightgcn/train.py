import os
import argparse

import torch
import wandb

from lightgcn.args import parse_args
from lightgcn.datasets import prepare_dataset
from lightgcn import trainer
from lightgcn.utils import get_logger, set_seeds, logging_conf


logger = get_logger(logging_conf) # 로그 객체 생성


def main(args: argparse.Namespace):
    wandb.login()
    wandb.init(project="dkt", config=vars(args)) # wandb dkt디렉토리에 config.yaml 파일에 저장
    set_seeds(args.seed) # 42 디폴트
    
    use_cuda: bool = torch.cuda.is_available() and args.use_cuda_if_available # gpu 사용 여부
    device = torch.device("cuda" if use_cuda else "cpu") # 사용할 디바이스: gpu/cpu

    logger.info("Preparing data ...") # 데이터셋 준비!
    # gpu로 연산할 수 있는 텐서로 이루어진 딕셔너리형 데이터셋
    train_data, test_data, n_node = prepare_dataset(device=device, data_dir=args.data_dir) 

    logger.info("Building Model ...") # 모델 생성!
    # 모델 생성: 학습해둔 가중치들이 있다면 그 가중치를 적용한 모델을 생성, 아닐 시 그냥 모델 생성
    model = trainer.build(
        n_node=n_node, # 유저와 아이템 전체 합친 개수만큼 노드 생성(학습/테스트 데이터셋에서 등장하는 모든 유저/아이템)
        embedding_dim=args.hidden_dim, # 유저/아이템을 임베딩할 벡터 크기: 64 디폴트
        num_layers=args.n_layers, # 몇 hope를 사용할 것인가
        alpha=args.alpha, # 각 레이어에서 나온 값에 동일하게 곱해지는 하이퍼파라미터(None이면 각 레이어 임베딩마다 1/(k+1)씩 곱해짐)
        beta=args.beta,
        select_beta=args.select_beta
    )
    model = model.to(device) # 모델 gpu에 태우기
    
    logger.info("Start Training ...") # 학습 시작!
    trainer.run(
        model=model,
        train_data=train_data,
        n_epochs=args.n_epochs,
        learning_rate=args.lr,
        model_dir=args.model_dir,
    )


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(name=args.model_dir, exist_ok=True)
    main(args=args)
