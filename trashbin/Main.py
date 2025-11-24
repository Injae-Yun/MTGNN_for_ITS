import os
import yaml
import numpy as np
import pandas as pd
import pickle
import glob
import psycopg2
import time
import shutil

from Utils.logger import get_logger
from Utils.get_focus_timepoints import get_focus_timepoints
from Utils.Preprocess_data import (
    process_day_to_linkwise, 
    build_cluster_tensor,
    build_cluster_y_tensor,
    build_cluster_y_tensor_optimized,
    process_raw_to_cluster_tensor,
    process_raw_to_cluster_tensor_h5
)
from Utils.Preprocess_fast import (
    process_raw_to_cluster_tensor_pivot,
    process_predict_to_cluster_tensor_pivot,
)
from Script.Dataloader_DB import load_from_mongo
from Script.Tester import (
    Predictor,
    Make_Test_results,
)
from Utils.share import Make_Postgres_form
from Script import Trainer
from Core.Train_MTGNN import training_MTGNN
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
        if os.path.isfile(file_path):
            os.remove(file_path)
        elif os.path.isdir(file_path):
            # 디렉터리가 비어 있으면
            try:
                os.rmdir(file_path)
            except OSError:
                # 비어 있지 않다면 하위 포함 전부 삭제
                shutil.rmtree(file_path)


def main(mode="train",version='v1.0.0',
         config_name='Livinglab_test.yaml',
         course=[0,0,0,0],db_name='Mongo_db'):
    # 0. link list 불러오고, mongo DB에서 데이터 불러오기
    Model_version = 'v'+str(version[1])  # 'v3'에서 숫자만 추출 → 3
    db_yaml = db_name+'.yaml'  # MongoDB 접속 정보 파일
    Model_yaml = 'MTGNN_config.yaml'
    # 1. 로깅 및 출력 경로 설정
    logger = get_logger()
    output_dir_raw = os.path.join("Data", "Raw")
    output_dir_processed = os.path.join("Data", "Processed")
    output_dir_model = os.path.join("Model", Model_version)
    Config_path = os.path.join("Config" ,Model_yaml)

    # 2. Config 로드
    mongo_config = load_config(os.path.join("Config" ,db_yaml))
    task_config = load_config(os.path.join('Config',config_name))
    path_config = load_config(os.path.join('Config','Data_path_config.yaml'))
    dates,time_points=get_focus_timepoints(task_config,logger)
    target = task_config['target']  # 'Korea' 또는 'Livinglab'


    link_ids = np.loadtxt(os.path.join('Data',target,'Link.txt'), dtype=str)
    cluster_path = os.path.join('Data',target ,'expanded_clusters.pkl')
    with open(cluster_path, "rb") as f:
        clusters = pickle.load(f) #cluster information
    sorted_clusters = dict(sorted(clusters.items(), key=lambda x: x[0]))  # key 기준 오름차순 정렬

    # 4. MongoDB에서 데이터 로드 및 저장
    if course[0] == 0 :
        if db_name == 'Mongo_db':
            projection=task_config['projection']
            load_from_mongo(link_ids, dates,time_points,projection, mongo_config, output_dir_raw,target, logger)
            logger.info(f"✅ Data load 완료: 총 {len(dates)}개 파일 저장됨")
        else:
            logger.info(f"Error: other DB is not supported")
            exit()
    else: 
        logger.info(f"✅ Data load skip: 총 {len(dates)}개 파일 저장됨")

# 데이터 전처리 수행
    # Make processed data 
    if course[1] == 0:
        logger.info(f"🔹 pivot 방법: Raw → 클러스터 텐서 (고속·저메모리)")
        if mode == "predict":
                    # 1) 빠른 피벗 기반 예측 입력 생성 (predict_data.npy + timestamp)
            timestamp = process_predict_to_cluster_tensor_pivot(
                output_dir_raw, sorted_clusters, path_config, task_config, output_dir_model, logger
            )
            logger.info(f"✅ Raw 데이터에서 predict 클러스터 텐서 생성 완료: {output_dir_model}에 저장됨")
        else: 
            # H5 방법: Raw에서 바로 클러스터 텐서로 (메모리 효율적)
            timestamp = process_raw_to_cluster_tensor_pivot(
                output_dir_raw, sorted_clusters, path_config,
                task_config, output_dir_model, logger, generate_y=True
            )
            logger.info(f"✅ Raw 데이터에서 train/val/test 클러스터 텐서 생성 완료: {output_dir_model}에 저장됨")


    else: 
        timestamp = np.load(os.path.join(output_dir_processed, 'timestamp.npy'))
        logger.info(f"✅ 데이터 전처리 skip: {output_dir_model}에 저장됨")
    
    if mode == "train":
        # make map for MTGNN

        if course[2] == 0 :
            Trainer.genmap(link_ids,sorted_clusters,target,path_config,logger)#cluster_id
        else:
            logger.info(f"✅ map 형성 skip")
        training_MTGNN(Config_path, output_dir_model, sorted_clusters,target, logger, cluster_id=None )

    elif mode == "test":
        #이때 train이 아니라면, 각 데이터셋을 test용도로만 정리
        tic=time.time()
        Predict_result=Make_Test_results(output_dir_model,sorted_clusters,logger) #cluster_id=None
        toc=time.time()
        print(f"Time taken: {toc-tic} seconds")

    elif mode == "predict":
        #이때 train이 아니라면, 각 데이터셋을 test용도로만 정리
        tic=time.time()
        # 2) 예측 실행
        Predict_result=Predictor(output_dir_model,sorted_clusters,task_config,logger) #cluster_id=None
        toc=time.time()
        print(f"Time taken: {toc-tic} seconds")
        conn = psycopg2.connect(**task_config['Postgres_info'])
        Make_Postgres_form(timestamp,Predict_result,conn,path_config, task_config,logger)


if __name__ == "__main__":
    #main(mode="train",version='v1.0.0')
    """
    config_name = "Korea_train.yaml"
    target = 'Korea'
    course= [1, 1, 1, 1, 1 ]
    main(mode="train",version='v0.0.0',config_name=config_name,target=target,course=course) #v0 위치에 모델 저장
    # config_name = "Livinglab_test.yaml"
    # target= 'Livinglab'
    # main(mode="train",version='v0.0.0',config_name=config_name,target=target) #v0 위치에 모델 저장
    """
    config_name = "Livinglab_test.yaml"
    db_name = 'Mongo_db'
    course= [1, 1, 1] # 1 = skip, course[1]=0, course[2]=0으로 설정하여 최적화된 방법 테스트
    tic=time.time()
    main(mode="predict",version='v0.0.0',config_name=config_name,course=course,db_name=db_name) #v0 위치에 모델 저장
    toc=time.time()
    print(f"Time taken: {toc-tic} seconds")
    