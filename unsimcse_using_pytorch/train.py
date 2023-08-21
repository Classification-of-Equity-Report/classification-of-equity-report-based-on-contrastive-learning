### unsupervised model을 학습하기 위한 파일입니다.
from util import Args, bring_pretrain, csv_to_pandas
import torch
import transformers
from tqdm.auto import tqdm
from transformers import BatchEncoding
import torch.nn.functional as F


# 학습 코드 구현
# batch size = 64, learning rate 3e-5


def train_model(model, optimizer, train_data, val_data, lr_scheduler):
    for epoch in range(Args.epochs):
        model.train()
        for step, batch in enumerate(tqdm(train_data)):
            batch_data = BatchEncoding(batch).to(Args.device)

    
            # forward input을 2번 통과하기 위함
            # 다른 dropout mask가 자동적으로 적용된다.

            emb1 = model.forward(**batch_data)
            emb2 = model.forward(**batch_data)

            # emb1, emb2 pair간의 유사도를 비슷하게 확인하기 위해 코사인 유사도 계산

            emb1 = emb1[1].unsqueeze(1)
            emb2 = emb2[1].unsqueeze(0)

            sim_matrix = F.cosine_similarity(emb1, emb2, dim=-1)
            # temperature을 나누어줌
            sim_matrix = sim_matrix / Args.temperature

            # labels : 대각원소의 인덱스를 알려주는 역할(positive examples끼리 묶는용도)
            labels = torch.arange(Args.batch_size).long().to(Args.device)

            # cross entropy 사용 : softmax 효과를 주고 대각행렬의 유사도를 최대화할 수 있기때문

            loss = F.cross_entropy(sim_matrix, labels)    

            # 기울기 초기화
            optimizer.zero_grad()
            
            loss.backward()

            optimizer.step()
            
            lr_scheduler.step()