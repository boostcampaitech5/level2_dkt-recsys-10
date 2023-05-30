from typing import Optional, Union

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Embedding, ModuleList
from torch.nn.modules.loss import _Loss

from torch_geometric.nn.conv import LGConv
from torch_geometric.typing import Adj, OptTensor
from torch_geometric.utils import is_sparse, to_edge_index

'''https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/models/lightgcn.html#LightGCN'''

class LightGCN(torch.nn.Module):

    def __init__(
        self,
        num_nodes: int,
        embedding_dim: int,
        num_layers: int,
        beta: float,
        select_beta: bool,
        alpha: Optional[Union[float, Tensor]] = None, # 알파는 실수형, 텐서, none이 될 수 있음
        **kwargs,
    ):
        super().__init__()

        self.num_nodes = num_nodes # 노드 개수
        self.embedding_dim = embedding_dim # 임베딩 차원 크기
        self.num_layers = num_layers # 레이어 개수

        if select_beta:
            betak = torch.tensor([1. for _ in range(num_layers + 1)])
            betak[0] = beta
            for i in range(1,num_layers + 1):
                betak[i] = betak[i-1]*(1-beta)
                if i==num_layers:
                    betak[i] /= beta
            alpha = betak
        else:
            if alpha is None: 
                #alpha = 1. / (num_layers + 1) # 알파 입력 안해줄 경우 알아서 1/(레이어 개수+1)로 고정됨
                alpha = torch.tensor([1. for _ in range(num_layers + 1)])
                for i in range(num_layers + 1):
                    alpha[i] *= 1. / (i + 1)
        

        if isinstance(alpha, Tensor): # 알파로 텐서를 입력받았다면 
            assert alpha.size(0) == num_layers + 1 # 알파 텐서의 행 개수가 레이어 개수+1이어야 에러가 안남
        else:
            alpha = torch.tensor([alpha] * (num_layers + 1)) # 상수 알파 레이어 개수만큼 복제
        self.register_buffer('alpha', alpha) 

        self.embedding = Embedding(num_nodes, embedding_dim) # 노드 개수만큼 임베딩벡터 만들기
        self.convs = ModuleList([LGConv(**kwargs) for _ in range(num_layers)]) # 레이어마다 컨볼루션이 이뤄짐

        self.reset_parameters() # 학습 파라미터 초기화

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        torch.nn.init.xavier_uniform_(self.embedding.weight)
        for conv in self.convs: # 컨볼루션 레이어도 초기화
            conv.reset_parameters()

    def get_embedding(
        self,
        edge_index: Adj,
        edge_weight: OptTensor = None,
    ) -> Tensor: # 임베딩 행렬 출력
        r"""Returns the embedding of nodes in the graph."""
        x = self.embedding.weight # 임베딩 행렬
        out = x * self.alpha[0] # 임베딩 행렬 전체에 첫번째 알파값을 곱해줌

        for i in range(self.num_layers): # 레이어 개수만큼 반복
            x = self.convs[i](x, edge_index, edge_weight) # i번째 레이어의 임베딩 레이어. (노드 임베딩, 상호작용이 있는 노드의 인덱스 ) 
            out = out + x * self.alpha[i + 1] # 이번 레이어에 해당 알파를 곱하고 이전 레이어의 임베딩을 누적해서 더해줌

        return out # 유저, 아이템의 상호작용이 반영된 임베딩

    def forward(
        self,
        edge_index: Adj, # 인접행렬
        edge_label_index: OptTensor = None, # 랭킹이나 확률을 계산할 노드쌍을 입력받음. none일 경우 인접행렬의 모든 인덱스가 사용됨
        edge_weight: OptTensor = None, # 각 간선의 가중치
    ) -> Tensor:
        
        if edge_label_index is None: # 랭킹이나 확률을 계산할 노드 쌍이 없을 경우 -> 상호작용이 있는 모든 노드를 계산함
            # edge_index(인접 행렬)를 일반적인 텐서로 바꾸기 위한 과정인 듯
            if is_sparse(edge_index): # 인접행렬이 희소행렬 형태로 주어진 경우
                edge_label_index, _ = to_edge_index(edge_index) # 인덱스 정보를 텐서로 바꿈. 인접 행렬이라 그런지 행렬에 채워져있던 값은 그냥 버림
            else: # 희소행렬이 아니면
                edge_label_index = edge_index # 입력받은 그대로 사용
            

        out = self.get_embedding(edge_index, edge_weight) # 최종 임베딩

        out_src = out[edge_label_index[0]] # 최종 임베딩의 행들 중에서 1인 행들
        out_dst = out[edge_label_index[1]] # 최종 임베딩의 열들 중에서 1인 행들
        return (out_src * out_dst).sum(dim=-1) # 상호작용이 있는 유저와 아이템을 내적

    def predict_link(
        self,
        edge_index: Adj,
        edge_label_index: OptTensor = None,
        edge_weight: OptTensor = None,
        prob: bool = False,
    ) -> Tensor:
        
        pred = self(edge_index, edge_label_index, edge_weight).sigmoid() # forward를 통해 나온 결과를 시그모이드에 넣어서 0~1사이 나오게 함
        return pred if prob else pred.round() # 확률을 구하는 경우 그대로 출력, 아닐 경우 반올림하여 출력

    # src_index에서 추천할 k개 노드 인덱스
    def recommend(
        self,
        edge_index: Adj,
        edge_weight: OptTensor = None,
        src_index: OptTensor = None, # 추천이 이뤄질 노드 인덱스. None이면 모든 노드 사용
        dst_index: OptTensor = None, # 가능한 추천 선택지를 표현하는 노드 인덱스. None이면 모든 노드 사용
        k: int = 1, # 추천할 개수
    ) -> Tensor:
        
        out_src = out_dst = self.get_embedding(edge_index, edge_weight) # 임베딩 생성

        if src_index is not None:
            out_src = out_src[src_index]

        if dst_index is not None:
            out_dst = out_dst[dst_index]

        pred = out_src @ out_dst.t() # 그냥 두 임베딩을 내적
        top_index = pred.topk(k, dim=-1).indices

        if dst_index is not None:  # Map local top-indices to original indices.
            top_index = dst_index[top_index.view(-1)].view(*top_index.size())

        return top_index

    # binary cross entropy로 링크 예측의 오차를 계산
    def link_pred_loss(self, f, pred: Tensor, edge_label: Tensor,
                       **kwargs) -> Tensor:
        if f=='bcelog':
            loss_fn = torch.nn.BCEWithLogitsLoss(**kwargs) # BCEWithLogitsLoss loss function 객체 생성
        elif f=='mse':
            loss_fn = torch.nn.MSELoss(**kwargs)
        return loss_fn(pred, edge_label.to(pred.dtype)) # 같은 데이터타입으로 맞춰서 loss계산

    def recommendation_loss(self, pos_edge_rank: Tensor, neg_edge_rank: Tensor,
                            lambda_reg: float = 1e-4, **kwargs) -> Tensor: # 랭킹의 경우에는 BPR을 사용함
        
        loss_fn = BPRLoss(lambda_reg, **kwargs)
        return loss_fn(pos_edge_rank, neg_edge_rank, self.embedding.weight)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.num_nodes}, '
                f'{self.embedding_dim}, num_layers={self.num_layers})')


class BPRLoss(_Loss): # loss function 클래스 생성
    '''
    BPRLoss: 관찰되지 않은 상대보다 관찰된 상대의 예측을 높이는 쌍 loss 
    -> 그럼 예측해야하는 값들을 0으로 예측할 가능성이 높은건가..? 학습데이터셋에 대해서만 그런건가?
    '''
    
    __constants__ = ['lambda_reg']
    lambda_reg: float

    def __init__(self, lambda_reg: float = 0, **kwargs):
        super().__init__(None, None, "sum", **kwargs)
        self.lambda_reg = lambda_reg # 임베딩 L2에 붙는 정규화조절 term

    def forward(self, positives: Tensor, negatives: Tensor,
                parameters: Tensor = None) -> Tensor:
        
        n_pairs = positives.size(0)
        log_prob = F.logsigmoid(positives - negatives).mean()
        regularization = 0

        if self.lambda_reg != 0:
            regularization = self.lambda_reg * parameters.norm(p=2).pow(2)

        return (-log_prob + regularization) / n_pairs # 최소화 되어야하는 loss