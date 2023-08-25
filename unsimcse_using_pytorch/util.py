## 데이터를 불러오고 기본적인 하이퍼파라미터 pretrain_model을 불러오기 위한 코드입니다.

import os
import pickle
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np


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
    csv_dataset_dir = ".\\data\\txt_files_f1\\txt_files_f1\\"
    pickle_dataset_dir = ".\\data\\txt_pkl_v3\\txt_pkl_v3\\"
    save_new_dataset_dir = ".\\data\\pickle_180_days\\"
    kospi_dataset_dir = '.\\data\\'
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


def excel_to_pandas(path, file_name):
    """
    엑셀 파일을 판다스로 바꾸는 코드입니다.
   
   input:
        path(str) : 엑셀파일이 저장되어 있는 경로
        file_name(str) : 파일 이름
    
    return:
        data_df(DataFrame): 엑셀 파일을 데이터 프레임형태로 변환
    """
    data_df = pd.read_excel(path + file_name)

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


def make_day_return(data_dict):
    """
    레포트마다 누적수익률이 저장된 피클파일 데이터셋에서 컬럼을 180일까지 늘리는 함수입니다.

    input:
        data_dict(Dict): 피클 파일 제목을 키값, 데이터는 DataFrame형태로 되어있는 딕셔너리 파일

    output:
        data_dict(Dict): 컬럼 길이를 180일까지 늘려둔 Dict 데이터
    """

    for pkl_file in data_dict.keys():
        while data_dict[pkl_file].shape[1] <= 180:
            data_dict[pkl_file].loc[:, data_dict[pkl_file].columns[-1] +1] = np.NaN
    
    return data_dict


def _check_make_dir(dir):
    """
    새로운 데이터를 저장하기 위해 저장할 dir을 확인하는 코드입니다.

    input:
        dir(str) : 생성 및 확인하고자하는 폴더 경로
    output:
        해당되는 경로에 폴더 생성 
    """
    try:
        if not os.path.exists(dir):
            os.makedirs(dir)
        print('complete make dir')
    except OSError:
        print('fail to make dir')


def save_new_data_to_pickle(dir, data_dict):
    """
    새로운 데이터를 저장하기 위한 dir를 확인하고 데이터를 pickle파일로 저장하는 코드입니다.

    input:
        dir(str) : 파일을 저장하기 위한 경로
        data_dict(dict) : 데이터가 저장되어 있는 dict

    output:
         pickle_file : 해당 경로에 pickle file로 저장됩니다.
    """
    
    _check_make_dir(dir)

    for file_name in data_dict.keys():
        with open(dir + file_name, 'wb') as f:
            pickle.dump(data_dict[file_name], f)
            print(file_name +' save complete')
