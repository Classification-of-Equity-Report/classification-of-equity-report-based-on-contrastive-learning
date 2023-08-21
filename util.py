## 데이터를 불러오고 기본적인 하이퍼파라미터 pretrain_model을 불러오기 위한 코드입니다.

import os
import pickle
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel


class Args:
    '''
    기본적인 파라메터를 설정하는 클래스
    
    batch size(int) : 64 / 데이터 배치사이즈를 설정하는 객체입니다. 모델에선 64가 성능이 좋다고 합니다.
    learning_late(float) : 3e-5 / 기본적인 학습률을 설정하는 객체입니다. 모델에선 3e-5의 성능이 제일 좋다고 합니다.
    data_dir(str) : 데이터가 저장되어 있는 경로를 설정하는 코드입니다.
    sts_eval_dir(str) : STS data을 기반으로 평가한 evaluation을 저장하기 위한 경로입니다.
    output_dir(str) : 결과물을 저장하기 위한 경로입니다. 
    epoch(int) : 1 / unsupervised learning에서는 epoch 1로 설정
    temperature(float) : 0.05 / loss를 구할때 유사도에 나누어질 수치 
    eval_logging_interval(int) : 250 / 논문에서는 250번마다 평가
    seed(int) : 42 /seed 설정하기 위한 수치
    device = "cuda:0" / GPU 사용 설정
    max_length(int) : 토크나이징의 max_length
    truncation(bool) : 토크나이징 truncation
    padding(str) : "max_length"/ 토크나이징의 padding 조건 선택
    '''


    model_name = "snunlp/KR-FinBert"
    dataset_dir = ".\\data\\txt_files_f1\\txt_files_f1\\"
    learning_rate = 3e-5
    batch_size = 4
    sts_eval_dir = ".\\sts"
    output_dir = ".\\output"
    epochs = 1
    temperature = 0.05
    eval_logging_interval = 250
    seed = 42
    num_warmup_steps = 0
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    max_length = 512
    truncation=True
    padding="max_length"


def pickle_to_dict(path):
    """
    피클파일을 수합하여 딕셔너리로 반환하기 위한 코드입니다.
    input:
        path(str):피클파일이 저장되어 있는 경로
    
    return:
        data_dict(dict) : 피클파일들을 전부 수합한 딕셔너리 파일(key('파일명') :value(피클파일의 데이터))
    """
    data_list = os.listdir(path)

    data_dict={}

    for data_name in data_list:
        with open(path + data_name, 'rb') as f:
            data_dict[data_name] = pickle.load(f)
    
    return data_dict


def csv_to_pandas(path):
    """
    csv 파일을 수합하여 데이터프레임 형태로 반환하기 위한 코드입니다.

    input:
        path(str) : csv파일이 저장되어 있는 경로
    
    return:
        data_df(DataFrame): csv 파일을 데이터 프레임형태로 수합한 데이터프레임 
    """

    data_list = os.listdir(path)

    data_df = pd.DataFrame()

    for data_name in data_list:
        data_df = pd.concat([data_df, pd.read_csv(path+data_name)], axis=0)

    data_df.rename(columns={"Unnamed: 0" : '주가번호'}, inplace=True)
    data_df.reset_index(drop=True, inplace=True)
    data_df.rename(columns={'0':'report'}, inplace=True)
    data_df.drop(columns=['주가번호'], axis=1, inplace=True)
    return data_df


def bring_pretrain(model_name):
    """
    pretrain된 모델을 불러오기 위한 함수입니다.
    input:
        model_name(str) : 허깅페이스에 저장된 모델과 토크나이즈를 불러오기위한 코드입니다.

    return:
        tokenizer : pretrain된 토크나이저
        model : pretrain된 모델
    """
    tokenizer = AutoTokenizer.from_pretrained(Args.model_name)
    model = AutoModel.from_pretrained(Args.model_name)
    
    return tokenizer, model



     



