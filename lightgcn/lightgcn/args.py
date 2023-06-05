import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", default=42, type=int, help="seed")
    parser.add_argument("--use_cuda_if_available", default=True, type=bool, help="Use GPU")
    
    parser.add_argument("--data_dir", default="/opt/ml/input/data", type=str, help="")
    
    parser.add_argument("--output_dir", default="./outputs/", type=str, help="")
    
    parser.add_argument("--hidden_dim", default=64, type=int, help="")
    parser.add_argument("--n_layers", default=2, type=int, help="")
    parser.add_argument("--alpha", default=None, type=float, help="") # 각 레이어에서 나온 임베딩 합산시에 곱해지는 하이퍼파라미터
    parser.add_argument("--beta", default=0.8, type=float, help="")
    parser.add_argument("--select_beta", default=True, type=bool, help="")

    parser.add_argument("--n_epochs", default=60, type=int, help="")
    parser.add_argument("--lr", default=0.001, type=float, help="")
    parser.add_argument("--model_dir", default="./models/", type=str, help="") # 모델 저장 위치
    parser.add_argument("--model_name", default="best_model.pt", type=str, help="") # 저장되어있는 모델 불러오기
    

    args = parser.parse_args()

    return args
