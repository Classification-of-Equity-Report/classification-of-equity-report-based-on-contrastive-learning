import numpy as np
from scipy import stats
import pandas as pd


def change_str(pandas_df):
    """
    코스피 데이터의 object를 int로 바꾸기 위한 함수입니다.
    
    """
    pandas_df = pandas_df.replace(',', '')
    pandas_df = float(pandas_df)
    return pandas_df


def change_str_to_datatime(pandas_df):
    """
    코스피 데이터의 object를 date_time 으로 바꾸기 위한 함수입니다.
    """
    return pd.to_datetime(pandas_df)


def _calculate_index(pandas_df, col1, col2):
    """
    코스피 현재 지수 대비 수익률을 구하기 위한 함수입니다.

    input:
        pandas_df(df) : 현재 지수에서 일차별 지수 데이터 프레임
        col1 : 일차별 지수 데이터
        col2 : 현재 지수
    
    output:
        y(float) : 일차별 코스피 누적 수익률 데이터
    """
    
    y = (pandas_df[col1] - pandas_df[col2]) / pandas_df[col2]

    return y


def make_cumulative_return(pandas_df, col_name):
    """
    코스피 현재지수 대비 수익률을 구하기 위한 함수입니다.

    input:
        pandas_df(df): 코스피 데이터 프레임
        col_name(str) : 지수 수익률을 구하기 위한 컬럼 이름

    output:
        pandas_df(df) : 현재 지수 대비 누적수익률을 구하기 위한 데이터프레임(일차별 누적 수익률)
    """
    for day in range(180):
        pandas_df[day] = pandas_df[col_name].shift(periods=-(day+1))

    # 판다스의 일차별 수익률 컬럼
    colnames = pandas_df.loc[:, 0:].columns

    for col in colnames:
        pandas_df[col] = _calculate_index(pandas_df, col, col_name)

    return pandas_df


def change_date_to_str(pandas_df):
    """
    '/' 형태의 '일자'데이터를 리포트별 일자 데이터형태로 일치시키기 위한 코드입니다.
 
    """
    pandas_df = pandas_df.replace('/', '')
    return pandas_df



def _make_file_name(x):
    return x.split('.')[0]


def _split_filename(data_dict):
    """
    코스피 수익률과 비교하기 위한 str 타입을 찾기위한 코드입니다.
    
    input:
        data_dict(dict) : 리포트 데이터 딕셔너리

    output:
        str_date_key(list) : list 데이트 키값
    """
    data_dict_keys = list(data_dict.keys())

    str_date_key = list(map(_make_file_name, data_dict_keys))

    return str_date_key


def compare_kospi_stock_return(kospi_data, data_dict):
    """
    일자별 stock 누적 수익률 - kospi data를 빼는 함수입니다.

    input:
        kospi_data(df) : 코스피 데이터
        data_dict(dict) : 피클파일이 모여있는 딕셔너리

    output:
        data_dict(dict) : 코스피 수익률을 뺀 딕셔너리 데이터
        empty_kospi_li(list) : 코스피 데이터가 없는 데이터
    """

    str_date_keys = _split_filename(data_dict)
    empty_kospi_li = []

    for str_date_key in str_date_keys:
        try:
            # 코스피 데이터의 해당 일자에 대한 값을 가져오는 코드입니다.
            key_kospi_index = kospi_data[kospi_data['일자'] == str_date_key]

            # 해당 일자에 해당하는 리포트를 가져오는 코드입니다.
            data_dict[str_date_key+'.pkl'].reset_index(inplace=True)

            # 해당 일자에 해당하는 리포트에서 코스피 수익률을 빼는 코드입니다.
            data_dict[str_date_key+'.pkl'].loc[:, 0:] = data_dict[str_date_key+'.pkl'].loc[:, 0:] - key_kospi_index.loc[:, 0:].values

            data_dict[str_date_key+'.pkl'].set_index('주가번호', inplace=True, drop = True)

        except ValueError as e:
            print(e)
            print('no kospi index in', str_date_key)
            empty_kospi_li.append(str_date_key)
        
    return data_dict, empty_kospi_li


def delete_and_make_error_data(empty_kospi_li, data_dict):
    """
    코스피 데이터가 없는 리포트 딕셔너리를 만듭니다.

    input:
        empty_kospi_li(list): 코스피 데이터가 없는 파일 이름 리스트
        data_dict(dict) : 코스피 대비 수익률이 계산된 리포트 파일

    output : 
        empty_dict(dict) : 코스피 데이터가 없는 리포트들의 딕셔너리
        data_dict(dict) : empty_kosip를 제거한 딕셔너리 파일
    """
    empty_dict = {}
    empty_kospi_li = list(map(lambda x : x + '.pkl', empty_kospi_li))
    
    for empty_kospi in empty_kospi_li:
        empty_dict[empty_kospi] = data_dict.pop(empty_kospi)
    
    return empty_dict, data_dict