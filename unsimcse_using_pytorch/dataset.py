
import os
os.chdir('C:/Users/user/Desktop/박준영/공부/nlp_study/simcse/')
import torch
from util import Args, bring_pretrain, csv_to_pandas
from torch.utils.data import Dataset, DataLoader, random_split
from datasets import Dataset


def _make_dataset(data):
    """
    데이터로더에 싣기위한 코드입니다.
    
    input:
        Data(pandas): 판다스 형태의 데이터 파일

    return:
        dataset : 데이터 로더에 싣기위한형태의 파일
    """
    dataset = Dataset.from_pandas(data)
    
    return dataset


def _tokenizing_dataset(dataset, tokenizer, Args):
    """
    데이터셋을 토크나이징 하기 위한 함수입니다.
    
    input:
        dataset : 데이터셋 형태화 된 데이터
        tokenizer : pretrain tokenizer
        *kwargs : util의 Args class

    return:
        dataset : 토크나이징 된 데이터셋
    """

    dataset = dataset.map(lambda x: tokenizer(x['report'], max_length = Args.max_length, truncation=Args.truncation, padding=Args.padding), batched=True, remove_columns=['report']).with_format("torch", columns = ['input_ids', 'token_type_ids', 'attention_mask'])
    
    return dataset


def _split_data(dataset):
    """
    데이터셋을 train, valid, test로 나누기위한 코드입니다.

    input:
        dataset : 토크나이징 된 데이터셋
    
    return:
        train_dataset
        valid_dataset
        test_dataset
    """
    train_size = 0.8
    validate_size = 0.1
    test_size = 1- train_size-validate_size

    train_dataset, validate_dataset, test_dataset = random_split(dataset, [train_size, validate_size, test_size])

    return train_dataset, validate_dataset, test_dataset


def _make_dataloader(dataset, Args):
    """
    데이터로더를 만들기 위한 함수입니다.

    input:
        dataset : 토크나이징 된 데이터셋
    
    return:
        dataloader : 데이터로드화 데이터셋
    """

    dataloader = DataLoader(dataset, Args.batch_size, shuffle=True)
    return dataloader


def bring_dataset(data, tokenizer, Args):
    """
    CSV 파일을 데이터로더화 하기 위한 코드입니다.
    input:
        data(dataframe): 데이터프레임화 된 데이터
        tokenizer : pretrain tokenizer
        Args : util의 Args class

    output:
        train_dataloader
        valid_dataloader
        test_dataloader
    """
    dataset = _make_dataset(data)
    dataset = _tokenizing_dataset(dataset, tokenizer, Args)
    train_dataset, validate_dataset, test_dataset = _split_data(dataset)
    train_dataloader = _make_dataloader(train_dataset, Args)
    valid_dataloader = _make_dataloader(validate_dataset, Args)
    test_dataloader = _make_dataloader(test_dataset, Args)

    return train_dataloader, valid_dataloader, test_dataloader

if __name__ == '__main__':
    Args=Args()
    data = csv_to_pandas(Args.dataset_dir)
    tokenizer, model = bring_pretrain(Args.model_name)
    bring_dataset(data, tokenizer, Args)



