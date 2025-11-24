import os
import numpy as np
import pandas as pd
import joblib
import glob
from dbfread import DBF
import psycopg2
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from Utils.share import Scaler_tool
import h5py
import gc
from sklearn.model_selection import train_test_split
import torch

def prepare_feature_target_h5(
    ds: h5py.Dataset,
    emb_vals: np.ndarray,
    step: int,
    predict_steps: int,
    kind: str = 'x',
) -> np.ndarray:
    """
    ds:   HDF5 dataset (shape=(N, C, L) or (N, 1, L) for y)
    emb_vals: (L, E) node2vec embeddings
    step:     0 <= step < predict_steps
    predict_steps: e.g. 288
    kind: 'x' or 'y'
    """
    # 1) step 단위로 슬라이스 → (D, C, L)
    arr = ds[step::predict_steps, ...]  
    D, C, L = arr.shape

    # 2) (D, C, L) → (D, L, C) → (D*L, C)
    X_flat = arr.transpose(0,2,1).reshape(D * L, C)

    if kind == 'y':
        # y는 (D*L,) 스칼라값
        return X_flat.ravel()

    # 3) emb_vals tile → (D*L, E)
    E = emb_vals.shape[1]
    static_tile = np.tile(emb_vals, (D, 1))  # (D*L, E)

    # 4) concatenate → (D*L, C+E)
    return np.hstack([X_flat, static_tile])
    
def log_to_logger(logger, period=1):
    """LightGBM 학습 로그를 Python logger로 보내는 콜백"""
    def _callback(env):
        if period > 0 and env.iteration % period == 0:
            result_str = f"[{env.iteration}] " + \
                         " ".join(f"{name}'s {metric}: {value:.5f}"
                                  for name, metric, value, _ in env.evaluation_result_list)
            logger.info(result_str)
    _callback.order = 10
    return _callback

def training_LGBM(
    link_ids,
    Config_path,
    output_dir_model,
    logger,
    emb_path: str = 'node2vec_embeddings.npy'
):
    predict_steps = 288  # 하루 5분 간격

    # 1) 임베딩 로드
    emb_vals = np.load(
        os.path.join(output_dir_model, emb_path),
        allow_pickle=True
    )  # shape = (L, E)

    # 2) HDF5로 저장된 train/val/test/X, y 열기
    hf_x = h5py.File(os.path.join(output_dir_model, 'tensors.h5'), 'r')
    hf_y = h5py.File(os.path.join(output_dir_model, 'y_tensors.h5'),   'r')
    dsX = hf_x['train']
    dsY = hf_y['train']

    models = []
    model_names = []

    # 3) step별로 학습
    for step in range(predict_steps):
        fname = os.path.join(output_dir_model, f"lgbm_step_{step:03d}.pkl")
        if os.path.exists(fname):
            logger.info(f"✅ Step {step} model already exists, skipping...")
            model_names.append(os.path.basename(fname))
            continue
        logger.info(f"▶ Training step {step+1}/{predict_steps}")

        # 3.1) 이 스텝의 X, y 준비
        X = prepare_feature_target_h5(dsX, emb_vals, step, predict_steps, kind='x')
        y = prepare_feature_target_h5(dsY, emb_vals, step, predict_steps, kind='y')
        # 3.2) NaN 마스킹
        mask = ~np.isnan(y)
        X_clean = X[mask,:]
        y_clean= y[mask]
        X_train_sub, X_val, y_train_sub, y_val = train_test_split(X_clean, y_clean, test_size=0.2, random_state=42)
        # 3.3) LightGBM 학습
        model = lgb.LGBMRegressor(
            objective='regression',
            # 하이퍼파라미터 설정
            num_leaves=63,
            max_depth=10,
            learning_rate=0.02,
            n_estimators=2500,
            # 과적합 방지 규제 파람
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha= 0.1,
            reg_lambda= 0.1,
            # 기타 설정
            force_col_wise=True,
            random_state=42,
            verbose=1
        )
        model.fit(
            X_train_sub,
            y_train_sub,
            eval_set=[(X_val, y_val)],  # 검증 데이터셋 지정
            eval_names=['valid'],            # 이름 명시
            eval_metric='rmse',         # 평가 지표
            callbacks=[
                lgb.early_stopping(stopping_rounds=50),
                log_to_logger(logger, period=100)  # 50 라운드 마다 logger로 출력
            ]
        )
        # ✅ 학습 후 별도로 평가
        train_preds = model.predict(X_train_sub)
        train_rmse = mean_squared_error(y_train_sub, train_preds, squared=False)
        valid_rmse = model.best_score_['valid']['rmse']

        logger.info(
            f"🏆 Step {step} Best iteration: {model.best_iteration_}, "
            f"Train RMSE: {train_rmse:.5f}, Valid RMSE: {valid_rmse:.5f}"
        )
        # 3.4) 저장
        joblib.dump(model, fname)
        logger.info(f"✔ Saved model step {step} → {fname}")

        #models.append(model)
        model_names.append(os.path.basename(fname))

        # 메모리 해제
        del X, y, model
        gc.collect()

    # 4) 마무리
    hf_x.close()
    hf_y.close()

    np.save(
        os.path.join(output_dir_model, "lgbm_model_names.npy"),
        np.array(model_names, dtype='<U32')
    )
    logger.info(f"✅ Training complete: {len(model_names)} models saved")
    return model_names

# 추론 단계

def predict_LGBM(
    models_dir: str,
    model_names: list[str],
    D: int,
    L: int,
    logger,
    emb_path: str = 'node2vec_embeddings.npy',
    mode = "train"
) -> np.ndarray:
    predict_steps = 288
    # 1) 임베딩 & 스케일러 로드
    emb_vals = np.load(os.path.join(models_dir, emb_path), allow_pickle=True)
    scaler = Scaler_tool()
    #scaler = joblib.load(os.path.join(models_dir, "scaler.npy"))  # 수정: scaler도 joblib로 저장했다고 가정

    # 2) HDF5 테스트셋 열기

    if mode =='train':
        hf_x = h5py.File(os.path.join(models_dir, 'tensors.h5'), 'r')
        hf_y = h5py.File(os.path.join(models_dir, 'y_tensors.h5'),   'r')
        dsX_test = hf_x['test']
        dsY_test = hf_y['test']
        D, L = dsX_test.shape[0] // predict_steps, dsX_test.shape[2]
        all_pred_1d = np.empty((predict_steps, D*L), dtype=np.float32)
        all_true_1d = np.empty((predict_steps, D*L), dtype=np.float32)
            # 3) step별 예측 & 스텝별 지표
        for step, model_fname in enumerate(model_names):
            model = joblib.load(os.path.join(models_dir, model_fname))

            # 3.1) 스트리밍으로 X, y 불러오기
            X = prepare_feature_target_h5(dsX_test, emb_vals, step, predict_steps, kind='x')
            y= prepare_feature_target_h5(dsY_test, emb_vals, step, predict_steps, kind='y')

            # 3.2) 예측
            y_pred = model.predict(X)
            # 3.3) NaN 마스크
            mask = ~np.isnan(y)
            y_clean_1d = y[mask]
            y_pred_1d = y_pred[mask]
            all_pred_1d[step, :] = y_pred_1d
            all_true_1d[step, :] = y_clean_1d
            # 3.5) 스텝별 지표 계산
            mse  = mean_squared_error(y_clean_1d, y_pred_1d)
            rmse = np.sqrt(mse)
            mae  = mean_absolute_error(y_clean_1d, y_pred_1d)
            r2   = r2_score(y_clean_1d, y_pred_1d)
            y_pred_1d = scaler.inverse_transform(y_pred_1d)
            y_clean_1d = scaler.inverse_transform(y_clean_1d)

            y_clean_1d [y_clean_1d< 5] = 5  # 최소값 설정 #divided by 0 방지
            mape = mean_absolute_percentage_error(y_clean_1d, y_pred_1d)

            logger.info(
                f"Step {step:03d} → RMSE={rmse:.3f}, MAE={mae:.3f}, "
                f"MAPE={mape:.4f}, R2={r2:.3f}"
            )
            gc.collect()

        hf_x.close()
        hf_y.close()
        all_true_1d = all_true_1d.reshape(-1)
        all_pred_1d = all_pred_1d.reshape(-1)
        mask = ~np.isnan(all_true_1d)
        all_true_1d = all_true_1d[mask]
        all_pred_1d = all_pred_1d[mask]
        # 4) 전체 지표
        mse  = mean_squared_error(all_true_1d, all_pred_1d)
        tot_rmse = np.sqrt(mse)
        tot_mae  = mean_absolute_error(all_true_1d, all_pred_1d)
        tot_r2   = r2_score(all_true_1d, all_pred_1d)
        # 전체 MAPE 계산
        all_true_1d = scaler.inverse_transform(all_true_1d)
        all_pred_1d = scaler.inverse_transform(all_pred_1d)
        tot_mape = mean_absolute_percentage_error(all_true_1d, all_pred_1d)

        logger.info(
            f"Overall → RMSE={tot_rmse:.3f}, MAE={tot_mae:.3f}, "
            f"MAPE={tot_mape:.4f}, R2={tot_r2:.3f}"
        )
    else: 
        hf_x = h5py.File(os.path.join(models_dir, 'predict_tensors.h5'), 'r')
        dsX_test = hf_x['predict']
        D, L = dsX_test.shape[0] // predict_steps, dsX_test.shape[2]
        all_pred_2d = np.empty((predict_steps, D, L), dtype=np.float32)
        # 3) step별 예측 & 스텝별 지표
        for step, model_fname in enumerate(model_names):
            model = joblib.load(os.path.join(models_dir, model_fname))
            # 3.1) 스트리밍으로 X, y 불러오기
            X = prepare_feature_target_h5(dsX_test, emb_vals, step, predict_steps, kind='x')
            y_pred = model.predict(X)
            # 3.2) 역스케일링
            y_pred = scaler.inverse_transform(y_pred)
            # 3.4) 1d -> 2d 재변환
            all_pred_2d[step, :, :] = y_pred.reshape(D,L)

        hf_x.close()
        logger.info('Predicting complete, returning predictions')
        
    # 5) 마지막에 (N_test, 288) 형태로 반환
    # all_preds 리스트는 [step0_preds, step1_preds, …], 각 길이가 N_test_days*L
    return all_pred_2d# (steps_288, N_test_days*L) 형태로 반환

