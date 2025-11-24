# MTGNN을 이용한 교통속도 예측 프로젝트
# config, Data, cash, model 상세 제외. 각 데이터 형식에 맞춰서 진행할 것

## 1. 프로젝트 개요

본 프로젝트는 과거의 교통 데이터를 기반으로 미래의 교통 상황(예: 속도, 교통량)을 예측하는 것을 목표로 합니다. 이를 위해 교통 네트워크의 공간적, 시간적 종속성을 효과적으로 포착하는 **MTGNN(Multi-component-based Graph Neural Network)** 모델을 활용합니다.

프로젝트는 "Korea", "Livinglab" 등 다양한 데이터셋을 처리할 수 있도록 구조화되어 있으며, 각 데이터셋에 맞는 메인 스크립트와 설정 파일을 사용합니다.

## 2. 주요 디렉토리 구조

- **/Main_*.py**: 학습, 평가, 테스트를 실행하기 위한 메인 스크립트 파일들.
- **/Config**: 실험을 위한 YAML 설정 파일들.
- **/Core**: MTGNN 네트워크 아키텍처(`net.py`) 및 학습 로직(`trainer.py`) 등 모델의 핵심 구성 요소.
- **/Data**: 원본 및 가공된 데이터셋 저장 디렉토리.
- **/Results**: 모델의 출력 및 예측 결과가 저장되는 디렉토리.
- **/Logs**: 모델 실행 중 생성되는 로그 파일 저장 디렉토리.

## 3. 실행 방법

모델 학습 및 평가 프로세스는 메인 스크립트 중 하나를 해당 설정 파일과 함께 실행하여 시작합니다.

### 3.1. Korea 데이터셋의 경우

`Main_MTGNN_Korea.py`를 사용하여 Korea 데이터셋에 대한 실험을 실행합니다. `--config` 인자를 사용하여 설정 파일을 지정해야 합니다.

**예시: 모델 학습**
```bash
python Main_MTGNN_Korea.py --config Config/Korea_train.yaml
```

**예시: 모델 테스트**
```bash
python Main_MTGNN_Korea.py --config Config/Korea_far_test.yaml
```

### 3.2. Livinglab 데이터셋의 경우

유사하게, Livinglab 데이터셋에 대해서는 `Main_MTGNN_livinglab.py`를 사용합니다.

**예시: 실험 실행**
```bash
python Main_MTGNN_livinglab.py --config Config/Livinglab_test.yaml
```

## 4. 코어 모델

이 프로젝트의 핵심은 `Core/net.py`에 정의된 MTGNN 모델입니다. 이 모델은 그래프 컨볼루션과 시간적 컨볼루션을 활용하여 그래프 구조의 시계열 데이터로부터 학습합니다. 학습 과정은 `Core/trainer.py`의 `trainer` 클래스에 의해 관리됩니다.

더 자세한 설명은 `MTGNN_Description.md` 문서를 참고하십시오.

## 5. 전체 폴더 구조

```
/home/kayden/Analysis/Traffic_prediction/Project_AUTO/
├───Main_LGBM.py
├───Main_MTGNN_Korea.py
├───Main_MTGNN_livinglab.py
├───Main_MTGNN.py
├───README.txt
├───sample_output.csv
├───valid_link_id_read.py
├───__pycache__/
│   ├───layer.cpython-312.pyc
│   ├───Main_MTGNN.cpython-312.pyc
│   └───net.cpython-312.pyc
├───.vscode/
│   ├───launch.json
│   └───settings.json
├───Cache/
│   ├───0/
│   └───1/
├───Config/
│   ├───Auto2_config.yaml
│   ├───Data_path_config.yaml
│   ├───HSprediction.yaml
│   ├───Korea_far_test.yaml
│   ├───Korea_train_far.yaml
│   ├───Korea_train.yaml
│   ├───LGBM_config.yaml
│   ├───Livinglab_test.yaml
│   ├───Mongo_db.yaml
│   ├───MTGNN_config.yaml
│   ├───optimization_example.yaml
│   ├───Postgres_db.yaml
│   └───Test_db.yaml
├───Core/
│   ├───layer.py
│   ├───LGBM_model.py
│   ├───net.py
│   ├───Train_MTGNN.py
│   ├───trainer.py
│   └───__pycache__/
├───Data/
│   ├───Korea/
│   ├───Livinglab/
│   ├───Processed/
│   ├───Raw/
│   └───Results/
├───db/
│   ├───postgres_manager.py
│   └───__pycache__/
├───logs/
│   └───profiler/
├───Logs/
│   ├───log_2025-07-04 08:53:25.log
│   └───...
├───Model/
├───Results/
├───Script/
├───trashbin/
└───Utils/
```
