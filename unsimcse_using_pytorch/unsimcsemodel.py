### unsupervised simcse 모델 구조 코드입니다.
import torch.nn as nn
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions


class SimCSEmodel(nn.Module):
    def __init__(self, pretrain_model):
        """
        pretrain 모델을 받아와서 hiddens embedding을 가져오는 함수입니다.
        """
        super().__init__()
        # pretrain_model 불러오기
        self.model = pretrain_model
        # pretrain_model에서 hidden state를 가져와서 MLP 모델 통과하기
        self.hidden_size = self.model.config.hidden_size
        # input : hidden size, output : hidden size
        self.dense = nn.Linear(self.hidden_size, self.hidden_size)
        self.activation = nn.Tanh()
            

    def forward(self, input_ids, attention_mask, token_type_ids):
        """
        학습 시 output에서 pooling 된 last hidden state를 가져와 [CLS] 토큰 임베딩을 MLP layer로 통과시키는 코드입니다.
        """


        ## output에서 pooling 된 laste hidden states 얻기위함
        outputs = BaseModelOutputWithPoolingAndCrossAttentions(self.model(input_ids = input_ids, attention_mask = attention_mask, token_type_ids=token_type_ids))
        # [CLS] 토큰을 embedding 한 값 얻기
        # max pooling or mean pooling 사용가능
        embedding_value = outputs.last_hidden_state[:, 0]
        # unsupervised learning에서는 학습과정에 MLP를 넣어야함
        # 학습시 MLP layer 사용하기에 아래와 같이 코드를 작성
        if self.train():
            embedding_value = self.dense(embedding_value)
            embedding_value = self.activation(embedding_value)
        
        return embedding_value
    
