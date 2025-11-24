## Main_far.py
# LGBM 모델 학습 및 예측을 위한 메인 스크립트
# - MongoDB 또는 Postgres에서 데이터 로드

import os
import yaml
import numpy as np
import pandas as pd
import pickle
import glob
import psycopg2
from dbfread import DBF
import time

from Utils.logger import get_logger
from Utils.get_focus_timepoints import get_focus_timepoints
from Utils.Preprocess_data import (
    process_day_to_linkwise, 
    build_tensor,
    build_y_tensor
)
from Utils.Preprocess_fast import (
    for_LGBM_build_tensors
)
from Script.Dataloader_DB import (
    load_from_mongo,
    load_from_postgres
    )
from Script.Tester import (
    Predictor,
    Make_Test_results
    )
from Utils.share import Make_Postgres_form_far
from Script import Trainer
from Core.LGBM_model import (
    training_LGBM,
    predict_LGBM

)
from Utils.Node2Embedding import generate_node2vec_embeddings
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def load_link_ids(model_version: str, cluster_id: str) -> list:
    """
    모델 버전 및 클러스터 ID에 따라 LINK_ID 리스트를 불러옴
    예: Model/v3.2.1/cluster_001/link_ids.txt
    """
    file_path = os.path.join("Model", model_version, f"cluster_{cluster_id:03}", "link_ids.txt")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"❌ link_ids 파일이 존재하지 않음: {file_path}")
    return np.loadtxt(file_path, dtype=str).tolist()

def clear_parquet_files(processed_dir):
    parquet_files = glob.glob(os.path.join(processed_dir, "*"))
    for file_path in parquet_files:
        if os.path.exists( file_path) :
            os.remove(file_path)


def main(mode="train",version='v1.0.0',
         config_name='Livinglab_test.yaml',
         course=[0,0,0,0],db_name='Mongo_db'):
    # 0. link list 불러오고, mongo DB에서 데이터 불러오기
    Model_version = 'v'+str(version[1])  # 'v3'에서 숫자만 추출 → 3
    db_yaml = db_name+'.yaml'  # MongoDB 접속 정보 파일
    Model_yaml = 'LGBM_config.yaml'

    # 1. 로깅 및 출력 경로 설정
    logger = get_logger()
    output_dir_raw = os.path.join("Data", "Raw")
    output_dir_processed = os.path.join("Data", "Processed")
    output_dir_model = os.path.join("Model", Model_version)
    Config_path = os.path.join("Config" ,Model_yaml)

    # 2. Config 로드
    db_config = load_config(os.path.join("Config" ,db_yaml))
    task_config = load_config(os.path.join('Config',config_name))
    path_config = load_config(os.path.join('Config','Data_path_config.yaml'))
    dates,time_points=get_focus_timepoints(task_config,logger)
    target = task_config['target']  # 'Korea' 또는 'Livinglab'
    link_ids = np.loadtxt(os.path.join('Data',target,'Link.txt'), dtype=str)
    cluster_path = os.path.join('Data',target ,'expanded_clusters.pkl')
    with open(cluster_path, "rb") as f:
        clusters = pickle.load(f) #cluster information
    sorted_clusters = dict(sorted(clusters.items(), key=lambda x: x[0]))  # key 기준 오름차순 정렬
    D= len(dates)
    L = len(link_ids)
    # 4. MongoDB에서 데이터 로드 및 저장
    if course[0] == 0 :
        if db_name == 'Mongo_db':
            projection=task_config['projection']
            load_from_mongo(link_ids, dates,time_points, projection, db_config, output_dir_raw,target, logger)
        elif db_name == 'Postgres_db':
            load_from_postgres(link_ids, dates, time_points, task_config, db_config, output_dir_raw, target, logger)
        logger.info(f"✅ Data load 완료: 총 {len(dates)}개 파일 저장됨")
    else: 
        logger.info(f"✅ Data load skip: 총 {len(dates)}개 파일 저장됨")
    if course[1] == 0 :
        ITS_Link_path = path_config['data']['ITS_info']
        df = DBF(ITS_Link_path, encoding='cp949')
        Linkdata = pd.DataFrame(iter(df))
        selectedLink = Linkdata[['LINK_ID','F_NODE','T_NODE','LANES', 'MAX_SPD']]
        generate_node2vec_embeddings(output_dir_model,selectedLink,logger, link_ids, embed_dim=32)

    if course[2] == 0 :
        for_LGBM_build_tensors(
            output_dir_raw, link_ids, path_config,
            task_config, output_dir_model, mode, logger
        )
        logger.info(f"✅ 데이터 전처리(x-y) 완료: {output_dir_model}에 저장됨")
    else:
        logger.info(f"✅ 데이터 전처리(x-y) skip: {output_dir_model}에 저장됨")
    #     build_tensor(link_ids, output_dir_raw, path_config,
    #                      task_config, output_dir_model, logger,mode=mode) #cluster_id
    #     logger.info(f"✅ 데이터 전처리(x) 완료: {output_dir_model}에 저장됨")
    # else: 
    #     logger.info(f"✅ 데이터 전처리(x) skip: {output_dir_model}에 저장됨")
    # # split to train, val, test    
    # # make map for MTGNN
    # if course[3] == 0 :
    #     build_y_tensor(link_ids, output_dir_raw, path_config,
    #                      task_config, output_dir_model, logger,mode=mode) #cluster_id
    #     logger.info(f"✅ 데이터 전처리(y) 완료: {output_dir_model}에 저장됨")
    # else:
    #     logger.info(f"✅ 데이터 전처리(y) skip: {output_dir_model}에 저장됨")
    if mode == "train":

        model_names=training_LGBM(link_ids,Config_path,output_dir_model,logger,emb_path='node2vec_embeddings.npy')
        model_names= np.load(os.path.join(output_dir_model, "lgbm_model_names.npy"))
        predict_LGBM(output_dir_model,model_names,D,L,logger,emb_path='node2vec_embeddings.npy',mode=mode)

    elif mode == "predict":
        #이때 train이 아니라면, 각 데이터셋을 test용도로만 정리
        model_names= np.load(os.path.join(output_dir_model, "lgbm_model_names.npy"))
        Predict_result=predict_LGBM(output_dir_model,model_names,D,L,logger,emb_path='node2vec_embeddings.npy',mode=mode)
        conn = psycopg2.connect(**task_config['Postgres_info'])
        Make_Postgres_form_far(Predict_result,link_ids,dates,conn,path_config,task_config,logger)


if __name__ == "__main__":
    #main(mode="train",version='v1.0.0')
    """
    config_name = "Korea_train.yaml"
    target = 'Korea'
    course= [1, 1, 1, 1, 1 ]
    main(mode="train",version='v1.0.0',config_name=config_name,target=target,course=course) #v0 위치에 모델 저장
    # config_name = "Livinglab_test.yaml"
    # target= 'Livinglab'
    # main(mode="train",version='v0.0.0',config_name=config_name,target=target) #v0 위치에 모델 저장
    """
    config_name = "Korea_train_far.yaml"
    #config_name = "Korea_far_test.yaml"
    #config_name = "Livinglab_test.yaml"
    db_name = 'Postgres_db'
    course= [1, 1, 1, 0 ] # 1 = skip
    main(mode="train",version='v3.0.0',config_name=config_name,course=course,db_name=db_name) #v0 위치에 모델 저장
    