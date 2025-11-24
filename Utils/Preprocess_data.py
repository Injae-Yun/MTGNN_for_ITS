import pandas as pd
import os
import numpy as np
from datetime import timedelta, datetime
import holidays
from multiprocessing import Pool, cpu_count
from Utils.share import Scaler_tool
from dbfread import DBF

from datetime import datetime, timedelta
import os
import pandas as pd
import gc
import pyarrow as pa
import pyarrow.parquet as pq
import h5py

import os, gc
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

import multiprocessing as mp
from functools import partial

def process_day_to_linkwise(df, date, processed_dir, full_time_index,logger):
    df["insert_time"] = pd.to_datetime(df["insert_time"], format="%H:%M").dt.time
    grouped = df.groupby("link_id")
    timestamp = logger.timestamp  # 로거에서 타임스탬프 가져오기
    for link_id, group in grouped:
        group = group[["insert_time", "speed"]]
        group.set_index("insert_time", inplace=True)
        group = group.reindex(full_time_index)  # NaN 채움
        group["speed"] = group["speed"].astype("float32")
        # date_only = timestamp[:10]
        # save_path = os.path.join(processed_dir,date_only, f"{link_id}.parquet")
        save_path = os.path.join(processed_dir, f"{link_id}.parquet")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # 날짜 컬럼으로 구분 후 append
        group = group.rename_axis("time").reset_index()
        group["date"] = date

        if os.path.exists(save_path):
            old_df = pd.read_parquet(save_path) #2d format 최적
            merged = pd.concat([old_df, group], ignore_index=True)
        else:
            merged = group

        merged.to_parquet(save_path, index=False)
    logger.info(f"✅ 날짜 {date}: 정상 변환되었습니다..")
    gc.collect()  # 메모리 정리

    
def load_tensor_for_link(
    link_id,
    base_start: pd.Timestamp,
    offsets: list[int],    # 분 단위 오프셋 리스트
    max_offset: int,       # 분 단위 최대 오프셋
    selectedLink: pd.DataFrame,
    processed_data_dir: str,
    type: str,
    Processmode:str,
    logger
):
    # parquet 경로
    file_path = os.path.join(processed_data_dir, f"{link_id}.parquet")

    # max_speed 조회
    max_speed = float(
        selectedLink.loc[
            selectedLink['LINK_ID']==link_id,
            'MAX_SPD'
        ].iat[0]
    )

    # 전체 가능한 base_time 생성 (파일이 있든 없든 동일)
    # end_time = 하루 뒤 00:00 직전에서 max_offset 분 뺀 시점
    if Processmode == 'date_range':
        end_time = base_start + timedelta(days=1) - timedelta(minutes=max_offset)
    elif Processmode == 'time_reference':
        end_time = base_start + timedelta(minutes=max_offset)
    # offsets 간격: offsets 가 [0,5,10,...] 식이어야 함
    if len(offsets) > 1:
        step = offsets[1] - offsets[0]
    else:
        step = max_offset  # fallback
    # 00:00 부터 end_time 까지 step 분 간격
    all_base_times = pd.date_range(
        start=base_start, end=end_time, freq=f"{step}T"
    )

    # 파일 없으면 NaN 채운 배열 반환
    if not os.path.exists(file_path):
        if logger:
            logger.warning(f"파일 없음, NaN tensor로 대체: {file_path}")
        # (T, len(offsets))
        nan_tensor = np.full(
            (len(all_base_times), len(offsets)),
            np.nan,
            dtype=float
        )
        return nan_tensor, link_id, all_base_times

    # 실제 파일이 있으면 원래 로직
    df = pd.read_parquet(file_path, engine="pyarrow")
    df["datetime"] = pd.to_datetime(
        df["date"].astype(str) + " " + df["time"].astype(str)
    )
    df = df.set_index("datetime")["speed"].sort_index()

    # 유효한 base_times 필터
    valid = (all_base_times >= base_start) & (all_base_times <= df.index[-1] - timedelta(minutes=max_offset))
    base_times = all_base_times[valid]

    link_tensor = []
    for bt in base_times:
        time_points = [bt + timedelta(minutes=o) for o in offsets]
        row = df.reindex(time_points).values  # shape = (len(offsets),)
        if type == 'x':
            # NaN → max_speed
            row = np.where(np.isnan(row), max_speed, row)
        link_tensor.append(row)

    if len(link_tensor) == 0:
        # 유효한 시간이 하나도 없으면, NaN tensor로
        nan_tensor = np.full(
            (len(all_base_times), len(offsets)),
            np.nan,
            dtype=float
        )
        return nan_tensor, link_id, all_base_times

    return np.stack(link_tensor), link_id, base_times

def build_cluster_tensor(clusters, processed_data_dir, path_config, task_config, output_dir, logger,mode='train',cluster_id=None):
    """
    클러스터 단위로 (T, 1, N_links, 36) 형태의 텐서를 생성하여 저장

    Args:
        cluster_path (str): LivingLab_clusters.pkl 경로
        processed_data_dir (str): 링크별 parquet 파일 저장 폴더
        offsets (list[int]): 관심 시간 오프셋 (예: [-60, -55, ..., -5])
        output_dir (str): 텐서 저장 경로
        logger (Logger, optional): 로깅 객체
    """
    os.makedirs(output_dir, exist_ok=True)
    ITS_Link_path = path_config['data']['ITS_info']
    df = DBF(ITS_Link_path, encoding='cp949')
    Linkdata = pd.DataFrame(iter(df))
    selectedLink = Linkdata[['LINK_ID', 'MAX_SPD']]
    timestamp = logger.timestamp  # 로거에서 타임스탬프 가져오기
    date_only = timestamp[:10]  # '2025-06-27' 형식으로 날짜만 추출
    interval=task_config['interval']  # 시간 간격 (분 단위)
    interest = task_config['interest']
    interest_y = task_config.get('interest_y',0)
    train_ratio=task_config['train_ratio']
    Processmode =task_config['mode']
    offsets = []
    for base in interest:
        offsets.extend([interval * i for i in range(base, base + 12)])  # 12개씩
    offsets = sorted(offsets)  # 시간 순 정렬

    # 가능한 base time (가장 빠른 offset 기준 확보된 시간 범위)
    max_offset = interval *interest_y
    if Processmode == 'date_range':
        start_date = datetime.strptime(task_config['date_range']['start'], "%Y-%m-%d")
        base_start = start_date
    elif Processmode == 'time_reference':
        time_ref = task_config["time_reference"]
        base_date = task_config["time_reference"]["date"]  # "2025-06-27"
        base_time = time_ref.get("time") or "00:00"
        base_dt = datetime.strptime(f"{base_date} {base_time}", "%Y-%m-%d %H:%M")        
        base_start = base_dt - timedelta(minutes= 5) 

    for cid, link_ids in clusters.items():
        if cluster_id is not None and cid != cluster_id: # 특정 클러스터 ID only 처리
            continue
        cluster_path = os.path.join(output_dir, str(cid))
        os.makedirs(cluster_path, exist_ok=True)
        with Pool(processes=cpu_count() - 2) as pool:
            results = pool.starmap(
                load_tensor_for_link,
                [(lid, base_start, offsets, max_offset, selectedLink, processed_data_dir,'x', Processmode,logger) for lid in link_ids]
            )
        # 유효 결과 필터링
        tensors = []
        valid_link_ids = []
        for tensor, lid,_ in results:
            if tensor is not None:
                tensors.append(tensor)
                valid_link_ids.append(lid)
        
        # 저장
        if tensors:
            T = min(t.shape[0] for t in tensors)
            tensors = [t[:T] for t in tensors]  # 동일한 T로 자르기
            cluster_tensor = np.stack(tensors, axis=2)  # shape: (T, 36, N) → (T, 36, N_links)
            cluster_tensor = cluster_tensor[:, np.newaxis, :, :]  # (T, 1, N_links, 36)
            #cluster_tensor = np.transpose(cluster_tensor, (0, 1, 2, 3))  # 명시적

            cluster_path = os.path.join(output_dir,str(cid))
            os.makedirs(os.path.dirname(cluster_path), exist_ok=True)
            if mode == 'train':
                mean = np.nanmean(cluster_tensor)  # NaN 고려
                std = np.nanstd(cluster_tensor)
                Scaler = Scaler_tool(mean,std)  # 고정 스케일러 사용
                tensors = Scaler.transform(cluster_tensor)
                scaler ={"mean": float(mean), "std": float(std)}
                np.save(os.path.join(cluster_path, f"scaler.npy"),scaler)
                
                for option in ["train", "val", "test"]:
                    if option == "train":
                        indices = np.arange(0, int(T * train_ratio[0]))
                    elif option == "val":
                        indices = np.arange(int(T * train_ratio[0]), int(T * (train_ratio[0] + train_ratio[1])))
                    elif option == "test":
                        indices = np.arange(int(T * (train_ratio[0] + train_ratio[1])), T)
                    split_tensor = tensors[indices]
                    save_path = os.path.join(cluster_path, f"{option}_data.npz")
                    np.savez_compressed(save_path, split_tensor)
#                save_path = os.path.join(cluster_path, f"valid_link_id.npy")
#                np.save(save_path, valid_link_ids)
                logger.info(f"✅ {option} 데이터셋 저장 완료: {save_path} | shape={len(valid_link_ids)}")
            elif mode == 'test':  # test 모드
                scaler=np.load(os.path.join(cluster_path, f"scaler.npy"),allow_pickle=True).item()

                mean = scaler['mean']
                std = scaler['std']
                Scaler = Scaler_tool(mean,std)  # 고정 스케일러 사용
                tensors = Scaler.transform(cluster_tensor)
                save_path = os.path.join(cluster_path, f"predict_data.npy")
                np.save(save_path, tensors)
                logger.info(f"✅ predict_data 저장 완료: {save_path} | shape={tensors.shape}")
        else:
            logger.warning(f"⚠️ 클러스터 {cid}에 유효한 데이터 없음")
        if cid ==0 :# timestamp 저장            
            save_path = os.path.join(processed_data_dir, f"timestamp.npy")
            np.save(save_path, results[0][2])

    return results[0][2] # timestamp

def build_cluster_y_tensor(clusters, processed_data_dir, path_config, task_config, 
                           output_dir, logger,cluster_id=None):
    """
    클러스터 단위로 (T, 1, N_links, 36) 형태의 텐서를 생성하여 저장

    Args:
        cluster_path (str): LivingLab_clusters.pkl 경로
        processed_data_dir (str): 링크별 parquet 파일 저장 폴더
        offsets (list[int]): 관심 시간 오프셋 (예: [-60, -55, ..., -5])
        output_dir (str): 텐서 저장 경로
        logger (Logger, optional): 로깅 객체
    """
    os.makedirs(output_dir, exist_ok=True)
    ITS_Link_path = path_config['data']['ITS_info']
    df = DBF(ITS_Link_path, encoding='cp949')
    Linkdata = pd.DataFrame(iter(df))
    selectedLink = Linkdata[['LINK_ID', 'MAX_SPD']]

    timestamp = logger.timestamp  # 로거에서 타임스탬프 가져오기
    date_only = timestamp[:10]  # '2025-06-27' 형식으로 날짜만 추출
    interval=task_config['interval']  # 시간 간격 (분 단위)
    interest_y = task_config['interest_y']
    train_ratio = task_config['train_ratio']

    offsets = [interval * i for i in range(interest_y)]

    # 가능한 base time (가장 빠른 offset 기준 확보된 시간 범위)
    min_offset = abs(min(offsets))
    max_offset = interval *interest_y
    start_date = datetime.strptime(task_config['date_range']['start'], "%Y-%m-%d")
    base_start = start_date
    for cid, link_ids in clusters.items():
        if cluster_id is not None and cid != cluster_id: # 특정 클러스터 ID only 처리
            continue
        cluster_path = os.path.join(output_dir, str(cid))
        os.makedirs(cluster_path, exist_ok=True)

        with Pool(processes=cpu_count() // 4) as pool:
            results = pool.starmap(
                load_tensor_for_link,
                [(lid, base_start, offsets, max_offset, selectedLink, processed_data_dir,'y', logger) for lid in link_ids]
            )
        # 유효 결과 필터링
        tensors = []
        valid_link_ids = []
        for tensor, lid,_ in results:
            if tensor is not None:
                tensors.append(tensor)
                valid_link_ids.append(lid)
        
        # 저장
        if tensors:
            T = min(t.shape[0] for t in tensors)
            tensors = [t[:T] for t in tensors]  # 동일한 T로 자르기
            cluster_tensor = np.stack(tensors, axis=2)  # shape: (T, 12, N) → (T, 12, N_links)
            cluster_tensor = cluster_tensor[:, np.newaxis, :, :]  # (T, 1, N_links, 12)
            #cluster_tensor = np.transpose(cluster_tensor, (0, 1, 2, 3))  # 명시적

            cluster_path = os.path.join(output_dir,str(cid))
            os.makedirs(os.path.dirname(cluster_path), exist_ok=True)
            tensor =cluster_tensor
            
            for option in ["train", "val", "test"]:
                if option == "train":
                    indices = np.arange(0, int(T * train_ratio[0]))
                elif option == "val":
                    indices = np.arange(int(T * train_ratio[0]), int(T * (train_ratio[0] + train_ratio[1])))
                elif option == "test":
                    indices = np.arange(int(T * (train_ratio[0] + train_ratio[1])), T)
                split_tensor = tensor[indices]
                save_path = os.path.join(cluster_path, f"{option}_ydata.npz")
                np.savez_compressed(save_path, split_tensor)
                logger.info(f"✅ {option} 데이터셋 저장 완료: {save_path} | shape={split_tensor.shape}")

        else:
            logger.warning(f"⚠️ 클러스터 {cid}에 유효한 데이터 없음")
def build_tensor(
    link_ids,
    output_dir_raw,
    path_config,
    task_config,
    output_dir,
    logger,
    mode='test'
):
## for far prediction method

    os.makedirs(output_dir, exist_ok=True)

    # 0) 상수들 계산
    start = datetime.strptime(task_config["date_range"]["start"], "%Y-%m-%d")
    end   = datetime.strptime(task_config["date_range"]["end"],   "%Y-%m-%d")
    base_dates = [start + timedelta(days=i)
                  for i in range((end - start).days + 1)]
    interest = task_config['far_time_process']["interest"]   # e.g. [-21,...,0]
    offsets  = interest[:]
    interval = task_config['interval']  # in minutes, e.g. 5
    train_frac, val_frac, test_frac = task_config['train_ratio']

    L = len(link_ids)
    C = len(offsets) + 1
    T = 1440 // interval
    D = len(base_dates)
    N = D * T

    # 1) HDF5 파일 & 데이터셋 미리 생성
    if mode == 'train':
        h5_path = os.path.join(output_dir, "tensors.h5")
    elif mode == 'predict':
        h5_path = os.path.join(output_dir, "predict_tensors.h5")
    s_mode = 'r+' if os.path.exists(h5_path) else 'w'
    hf = h5py.File(h5_path, s_mode)

    if mode == 'train':
        n_train = int(D * train_frac)*T
        n_val   = int(D * val_frac)*T
        n_test  = N- n_train - n_val
        
        if n_train > 0:
            ds_train = hf.create_dataset(
                "train", (n_train, C, L),
                dtype="float16", chunks=(T, C, L), compression="gzip"
            )
        if n_val > 0:
            ds_val   = hf.create_dataset(
                "val",   (n_val,   C, L),
                dtype="float16", chunks=(T, C, L), compression="gzip"
            )
        if n_test > 0:
            ds_test  = hf.create_dataset(
                "test",  (n_test,  C, L),
                dtype="float16", chunks=(T, C, L), compression="gzip"
            )
    else:  # test
        if 'predict' in hf:
            del hf['predict']
        ds_pred = hf.create_dataset(
            "predict", (N, C, L),
            dtype="float16", chunks=(T, C, L), compression="gzip"
        )

    # 2) max_speed dict
    ITS_Link_path = path_config['data']['ITS_info']
    Linkdata = pd.DataFrame(iter(DBF(ITS_Link_path, encoding='cp949')))
    max_speed_dict = dict(zip(Linkdata['LINK_ID'], Linkdata['MAX_SPD']))

    global_idx = 0
    # 3) 날짜별로 하루치만 메모리에 올려서 처리
    for di, base_date in enumerate(base_dates):
        date_key = base_date.strftime("%Y-%m-%d")
        logger.info(f"[{date_key}] 처리 시작 ({di+1}/{D})")

        # a) 시간 인덱스
        times = [
            (base_date + timedelta(minutes=interval*i))
                .time().strftime("%H:%M")
            for i in range(T)
        ]

        # b) speed 채널 배열 (T, len(offsets), L)
        speeds = np.empty((T, len(offsets), L), dtype=np.float32)
        speeds.fill(np.nan)
        Scaler = Scaler_tool()
        if di == 0:
            np.save(os.path.join(output_dir, "scaler.npy"), Scaler)
        start_idx = global_idx
        end_idx   = start_idx + T
        # 각 offset 채널을 바로 쓰기
        for ci, off in enumerate(offsets):
            d = base_date + timedelta(days=off)
            fn = os.path.join(output_dir_raw, d.strftime("%Y%m%d") + ".parquet")
            if os.path.exists(fn):
                df = pd.read_parquet(fn,
                            columns=["link_id","insert_time","speed"])
                tmp = (
                    df.pivot(index="link_id", columns="insert_time", values="speed")
                    .reindex(index=link_ids, columns=times)
                    .apply(lambda row: row.fillna(max_speed_dict[row.name]), axis=1)
                )
                channel = tmp.values.T.astype(np.float16)  # shape = (T, L)
                del df, tmp
            else:
                channel = np.full((T, L), np.nan, dtype=np.float16)

            # 스케일링도 채널별로
            channel = Scaler.transform(channel)  # Scaler_tool 에 채널 단위 변환 메서드 가정

            # HDF5 write (train/val/test 혹은 predict)


            if mode == 'train':
                # train
                if 'ds_train' in locals():
                    a0, a1 = start_idx, min(end_idx, n_train)
                    if a1 > a0:
                        ds_train[a0:a1, ci, :] = channel[0:(a1-a0)]
                # val
                if 'ds_val' in locals():
                    b0, b1 = max(start_idx, n_train), min(end_idx, n_train+n_val)
                    if b1 > b0:
                        ofs = b0 - start_idx
                        ds_val[b0-n_train:b1-n_train, ci, :] = channel[ofs:ofs+(b1-b0)]
                # test
                if 'ds_test' in locals():
                    c0, c1 = max(start_idx, n_train+n_val), end_idx
                    if c1 > c0:
                        ofs = c0 - start_idx
                        ds_test[c0-(n_train+n_val):c1-(n_train+n_val), ci, :] = channel[ofs:ofs+(c1-c0)]
            else:
                ds_pred[start_idx:end_idx, ci, :] = channel

            del channel
        # c) holiday 채널 (T, L)
        kr = holidays.KR(years={base_date.year})
        flag = float(base_date in kr)
        hch = np.full((T, L), flag, dtype=np.float32)

        # d) 채널 결합 → (T, C, L)

        if mode == 'train':
            # train
            if 'ds_train' in locals():
                a0, a1 = start_idx, min(end_idx, n_train)
                if a1 > a0:
                    ds_train[a0:a1,-1,:] = hch
            # val
            if 'ds_val' in locals():
                b0, b1 = max(start_idx, n_train), min(end_idx, n_train+n_val)
                if b1 > b0:
                    ofs = b0 - start_idx
                    ds_val[b0-n_train:b1-n_train,-1,:] = hch
            # test
            if 'ds_test' in locals():
                c0, c1 = max(start_idx, n_train+n_val), end_idx
                if c1 > c0:
                    ofs = c0 - start_idx
                    ds_test[c0-(n_train+n_val):c1-(n_train+n_val)-1,:] = hch
        else:
            ds_pred[start_idx:end_idx,-1,:] = hch

        global_idx = end_idx
        # 메모리 해제
        del speeds, hch, 
        gc.collect()
        hf.flush()
    hf.close()
    logger.info(f"✅ HDF5 생성 완료: {h5_path}")


## for far prediction method
def build_y_tensor(
    link_ids,
    output_dir_raw,
    path_config,
    task_config,
    output_dir,
    logger,
    mode='test'
):
    os.makedirs(output_dir, exist_ok=True)

    # 0) 설정 불러오기
    start = datetime.strptime(task_config["date_range"]["start"], "%Y-%m-%d")
    end   = datetime.strptime(task_config["date_range"]["end"],   "%Y-%m-%d")
    base_dates = [start + timedelta(days=i)
                  for i in range((end - start).days + 1)]
    interval    = task_config['interval']             # in minutes, e.g. 5
    train_frac, val_frac, test_frac = task_config['train_ratio']

    L = len(link_ids)
    T = 1440 // interval     # 하루 스텝 수, e.g. 288
    D = len(base_dates)
    N = D * T

    # 1) HDF5 열기 & y용 데이터셋 미리 생성
    h5_path = os.path.join(output_dir, "y_tensors.h5")
    hf = h5py.File(h5_path, "w")
    Scaler = Scaler_tool()

    if mode == 'train':
        n_train = int(D * train_frac)*T
        n_val   = int(D * val_frac)*T
        n_test  = N - n_train - n_val

        if n_train > 0:
            ds_train = hf.create_dataset(
                "train", (n_train, 1, L),
                dtype="float16", chunks=(T,1,L), compression="gzip"
            )
        if n_val > 0:
            ds_val   = hf.create_dataset(
                "val", (n_val, 1, L),
                dtype="float16", chunks=(T,1,L), compression="gzip"
            )
        if n_test > 0:
            ds_test  = hf.create_dataset(
                "test", (n_test, 1, L),
                dtype="float16", chunks=(T,1,L), compression="gzip"
            )
    else:  # test 모드
        ds_pred = hf.create_dataset(
            "predict", (N, 1, L),
            dtype="float16", chunks=(T,1,L), compression="gzip"
        )

    # 2) max_speed dict 준비
    ITS_Link_path = path_config['data']['ITS_info']
    Linkdata = pd.DataFrame(iter(DBF(ITS_Link_path, encoding='cp949')))
    max_speed_dict = dict(zip(Linkdata['LINK_ID'], Linkdata['MAX_SPD']))

    global_idx = 0
    # 3) 날짜별 스트리밍 처리
    for di, base_date in enumerate(base_dates):
        date_key = base_date.strftime("%Y-%m-%d")
        logger.info(f"[{date_key}] Y tensor 처리 ({di+1}/{D})")

        # 3-1) time index
        times = [
            (base_date + timedelta(minutes=interval*i))
                .time().strftime("%H:%M")
            for i in range(T)
        ]

        # 3-2) 하루치 speed DataFrame
        fn = os.path.join(output_dir_raw, base_date.strftime("%Y%m%d") + ".parquet")
        if os.path.exists(fn):
            df = pd.read_parquet(fn, columns=["link_id","insert_time","speed"])
        else:
            df = pd.DataFrame(columns=["link_id","insert_time","speed"])
            logger.warning(f"  - {date_key}.parquet 없음 → 전부 NaN 처리")

        # 3-3) pivot + reindex
        tmp = (
            df
            .pivot(index="link_id", columns="insert_time", values="speed")
            .reindex(index=link_ids, columns=times)
        )
        # 3-4) NaN → max_speed
        tmp = tmp.apply(lambda row: row.fillna(max_speed_dict[row.name]), axis=1)

        # 3-5) NumPy `(T, L)` → reshape `(T,1,L)`
        day_y = tmp.values.T.astype(np.float32)
        day_y = day_y.reshape((T,1,L))
        day_y = Scaler.transform(day_y)  # Scaler_tool 에 채널 단위 변환 메서드 가정
        # 4) HDF5에 바로 쓰기
        start_idx = global_idx
        end_idx   = start_idx + T

        if mode == 'train':
            # train
            if 'ds_train' in locals():
                a0, a1 = start_idx, min(end_idx, n_train)
                if a1 > a0:
                    ds_train[a0:a1] = day_y[0:(a1-a0)]
            # val
            if 'ds_val' in locals():
                b0, b1 = max(start_idx, n_train), min(end_idx, n_train+n_val)
                if b1 > b0:
                    ofs = b0 - start_idx
                    ds_val[b0-n_train:b1-n_train] = day_y[ofs:ofs+(b1-b0)]
            # test
            if 'ds_test' in locals():
                c0, c1 = max(start_idx, n_train+n_val), end_idx
                if c1 > c0:
                    ofs = c0 - start_idx
                    ds_test[c0-(n_train+n_val):c1-(n_train+n_val)] = day_y[ofs:ofs+(c1-c0)]
        else:
            ds_pred[start_idx:end_idx] = day_y

        global_idx = end_idx
        # 메모리 해제
        del df, tmp, day_y
        gc.collect()
        hf.flush()

    hf.close()
    logger.info(f"✅ Y HDF5 생성 완료: {h5_path}")


def process_raw_to_cluster_tensor(raw_data_dir, clusters, path_config, task_config, output_dir, logger, mode='train', cluster_id=None, dates=None, exclude_missing_links=False, generate_y=False):
    """
    Raw 데이터에서 바로 클러스터 텐서를 생성하는 최적화된 함수
    - Raw parquet 파일들을 읽어서 링크별로 정리하고 클러스터 텐서로 변환
    - 중간 parquet 파일 생성 없이 메모리에서 직접 처리
    - dates 파라미터로 특정 날짜들만 처리
    - generate_y=True일 때 y 텐서도 함께 생성
    
    Args:
        exclude_missing_links: True면 비존재하는 링크를 제외, False면 max_speed로 채움
        generate_y: True면 y 텐서도 함께 생성
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 설정 로드
    interval = task_config['interval']
    interest = task_config['interest']
    interest_y = task_config.get('interest_y', 0)
    train_ratio = task_config['train_ratio']
    
    # x offsets 계산
    offsets = []
    for base in interest:
        offsets.extend([interval * i for i in range(base, base + 12)])
    offsets = sorted(offsets)
    
    # y offsets 계산 (generate_y가 True일 때만)
    if generate_y:
        y_offsets = [interval * i for i in range(interest_y)]
    
    # base_start 계산
    max_offset = interval * interest_y
    if task_config['mode'] == 'date_range':
        start_date = datetime.strptime(task_config['date_range']['start'], "%Y-%m-%d")
        base_start = start_date
    elif task_config['mode'] == 'time_reference':
        time_ref = task_config["time_reference"]
        base_date = time_ref["date"]
        base_time = time_ref.get("time") or "00:00"
        base_dt = datetime.strptime(f"{base_date} {base_time}", "%Y-%m-%d %H:%M")
        base_start = base_dt - timedelta(minutes=5)
    
    # MAX_SPD 정보 로드
    ITS_Link_path = path_config['data']['ITS_info']
    df = DBF(ITS_Link_path, encoding='cp949')
    Linkdata = pd.DataFrame(iter(df))
    selectedLink = Linkdata[['LINK_ID', 'MAX_SPD']]
    max_speed_dict = dict(zip(selectedLink['LINK_ID'], selectedLink['MAX_SPD']))
    
    # exclude_missing_links 옵션 가져오기
    exclude_missing_links = task_config.get('exclude_missing_links', True)
    
    # 처리할 날짜들 결정
    if dates is None:
        # dates가 없으면 모든 parquet 파일 처리
        raw_files = sorted([f for f in os.listdir(raw_data_dir) if f.endswith('.parquet')])
        dates_to_process = [f.replace('.parquet', '') for f in raw_files]
    else:
        # dates에서 정의된 날짜들만 처리
        dates_to_process = dates
        raw_files = [f"{date}.parquet" for date in dates]
    
    logger.info(f"🔹 Raw 파일 {len(raw_files)}개 처리 시작 (날짜: {dates_to_process[0]} ~ {dates_to_process[-1]})")
    if generate_y:
        logger.info(f"🔹 y 텐서 생성 옵션 활성화")
    
    # 전체 시간 인덱스 생성
    full_time_index = pd.Index(pd.date_range(start="00:00", end="23:59", freq="5min").time)
    
    # 링크별 데이터를 저장할 딕셔너리
    link_data_dict = {}
    
    # Raw 파일들을 순차적으로 처리
    for date in dates_to_process:
        # datetime 객체를 YYYYMMDD 문자열로 변환
        if isinstance(date, datetime):
            date_str = date.strftime("%Y%m%d")
        else:
            # 이미 문자열인 경우 그대로 사용
            date_str = str(date)
        
        file_path = os.path.join(raw_data_dir, f"{date_str}.parquet")
        
        try:
            # Raw 파일 읽기
            df = pd.read_parquet(file_path)
            if df.empty:
                logger.warning(f"⚠️ 날짜 {date}: 읽어온 데이터가 없습니다.")
                continue
                
            logger.info(f"🔹 날짜 {date_str} 처리 중...")
            
            # 링크별로 데이터 정리
            df["insert_time"] = pd.to_datetime(df["insert_time"], format="%H:%M").dt.time
            grouped = df.groupby("link_id")
            
            for link_id, group in grouped:
                if link_id not in link_data_dict:
                    link_data_dict[link_id] = []
                
                # 시간별 데이터 정리
                group = group[["insert_time", "speed"]]
                group.set_index("insert_time", inplace=True)
                group = group.reindex(full_time_index)
                group["speed"] = group["speed"].astype("float32")
                
                # 날짜 정보 추가
                group = group.rename_axis("time").reset_index()
                group["date"] = date_str
                
                link_data_dict[link_id].append(group)
                
        except Exception as e:
            logger.error(f"❌ 날짜 {date_str} 처리 중 오류 발생: {e}")
            continue
    
    logger.info(f"✅ Raw 데이터 처리 완료: {len(link_data_dict)}개 링크")
    
    # 클러스터별 텐서 생성
    for cid, cluster_link_ids in clusters.items():
        if cluster_id is not None and cid != cluster_id:
            continue
            
        cluster_path = os.path.join(output_dir, str(cid))
        os.makedirs(cluster_path, exist_ok=True)
        
        logger.info(f"🔹 클러스터 {cid} 텐서 생성 중... (링크 수: {len(cluster_link_ids)})")
        
        # 클러스터에 속한 링크들의 x 텐서 생성
        x_tensors = []
        y_tensors = []
        valid_link_ids = []
        
        for lid in cluster_link_ids:
            if lid in link_data_dict:
                try:
                    # 링크 데이터를 하나의 DataFrame으로 합치기
                    link_df = pd.concat(link_data_dict[lid], ignore_index=True)
                    
                    # x 텐서 생성
                    x_tensor = create_tensor_from_link_data_fast(
                        link_df, lid, base_start, offsets, max_offset, 
                        max_speed_dict, 'x', task_config['mode'], exclude_missing_links
                    )
                    
                    if x_tensor is not None:
                        x_tensors.append(x_tensor)
                        
                        # y 텐서도 생성 (generate_y가 True일 때만)
                        if generate_y:
                            y_tensor = create_tensor_from_link_data_fast(
                                link_df, lid, base_start, y_offsets, max_offset, 
                                max_speed_dict, 'y', task_config['mode'], exclude_missing_links
                            )
                            if y_tensor is not None:
                                y_tensors.append(y_tensor)
                            else:
                                # y 텐서가 None이면 x 텐서도 제거
                                x_tensors.pop()
                                continue
                        
                        valid_link_ids.append(lid)
                        
                except Exception as e:
                    logger.warning(f"링크 {lid} 텐서 생성 중 오류: {e}")
                    continue
        
        # x 텐서 저장 로직
        if x_tensors:
            T = min(t.shape[0] for t in x_tensors)
            x_tensors = [t[:T] for t in x_tensors]
            cluster_x_tensor = np.stack(x_tensors, axis=2)
            cluster_x_tensor = cluster_x_tensor[:, np.newaxis, :, :]
            
            if mode == 'train':
                mean = np.nanmean(cluster_x_tensor)
                std = np.nanstd(cluster_x_tensor)
                Scaler = Scaler_tool(mean, std)
                x_tensors_scaled = Scaler.transform(cluster_x_tensor)
                scaler = {"mean": float(mean), "std": float(std)}
                np.save(os.path.join(cluster_path, f"scaler.npy"), scaler)
                
                for option in ["train", "val", "test"]:
                    if option == "train":
                        indices = np.arange(0, int(T * train_ratio[0]))
                    elif option == "val":
                        indices = np.arange(int(T * train_ratio[0]), int(T * (train_ratio[0] + train_ratio[1])))
                    elif option == "test":
                        indices = np.arange(int(T * (train_ratio[0] + train_ratio[1])), T)
                    split_tensor = x_tensors_scaled[indices]
                    save_path = os.path.join(cluster_path, f"{option}_data.npz")
                    np.savez_compressed(save_path, split_tensor)
                    
                    # y 텐서도 함께 저장
                    if generate_y and y_tensors:
                        y_tensors_trimmed = [t[:T] for t in y_tensors]
                        cluster_y_tensor = np.stack(y_tensors_trimmed, axis=2)
                        cluster_y_tensor = cluster_y_tensor[:, np.newaxis, :, :]
                        split_y_tensor = cluster_y_tensor[indices]
                        save_y_path = os.path.join(cluster_path, f"{option}_ydata.npz")
                        np.savez_compressed(save_y_path, split_y_tensor)
                        logger.info(f"✅ {option} x,y 데이터셋 저장 완료: x={split_tensor.shape}, y={split_y_tensor.shape}")
                    else:
                        logger.info(f"✅ {option} x 데이터셋 저장 완료: shape={split_tensor.shape}")
                        
            elif mode == 'test':
                scaler = np.load(os.path.join(cluster_path, f"scaler.npy"), allow_pickle=True).item()
                mean = scaler['mean']
                std = scaler['std']
                Scaler = Scaler_tool(mean, std)
                x_tensors_scaled = Scaler.transform(cluster_x_tensor)
                save_path = os.path.join(cluster_path, f"predict_data.npy")
                np.save(save_path, x_tensors_scaled)
                logger.info(f"✅ predict_data 저장 완료: {save_path} | shape={x_tensors_scaled.shape}")
        else:
            logger.warning(f"⚠️ 클러스터 {cid}에 유효한 데이터 없음")
    
    # timestamp 생성 및 저장
    timestamp = logger.timestamp
    save_path = os.path.join(output_dir, f"timestamp.npy")
    np.save(save_path, timestamp)
    return timestamp

def create_tensor_from_link_data(link_df, link_id, base_start, offsets, max_offset, selectedLink, type, Processmode, logger):
    """
    링크 데이터에서 텐서를 생성하는 헬퍼 함수
    """
    # max_speed 조회
    max_speed = float(
        selectedLink.loc[
            selectedLink['LINK_ID']==link_id,
            'MAX_SPD'
        ].iat[0]
    )
    
    # 전체 가능한 base_time 생성
    if Processmode == 'date_range':
        end_time = base_start + timedelta(days=1) - timedelta(minutes=max_offset)
    elif Processmode == 'time_reference':
        end_time = base_start + timedelta(minutes=max_offset)
    
    if len(offsets) > 1:
        step = offsets[1] - offsets[0]
    else:
        step = max_offset
    
    all_base_times = pd.date_range(
        start=base_start, end=end_time, freq=f"{step}T"
    )
    
    # 데이터프레임을 datetime 인덱스로 변환
    link_df["datetime"] = pd.to_datetime(
        link_df["date"].astype(str) + " " + link_df["time"].astype(str)
    )
    link_df = link_df.set_index("datetime")["speed"].sort_index()
    
    # 유효한 base_times 필터
    valid = (all_base_times >= base_start) & (all_base_times <= link_df.index[-1] - timedelta(minutes=max_offset))
    base_times = all_base_times[valid]
    
    link_tensor = []
    for bt in base_times:
        time_points = [bt + timedelta(minutes=o) for o in offsets]
        row = link_df.reindex(time_points).values
        if type == 'x':
            # NaN → max_speed
            row = np.where(np.isnan(row), max_speed, row)
        link_tensor.append(row)
    
    if len(link_tensor) == 0:
        # 유효한 시간이 하나도 없으면, NaN tensor로
        nan_tensor = np.full(
            (len(all_base_times), len(offsets)),
            np.nan,
            dtype=float
        )
        return nan_tensor
    
    return np.stack(link_tensor)

def create_tensor_from_link_data_fast(link_df, link_id, base_start, offsets, max_offset, max_speed_dict, type, Processmode, exclude_missing_links=False):
    """
    최적화된 링크 데이터에서 텐서 생성
    
    Args:
        exclude_missing_links: True면 비존재하는 링크를 제외, False면 max_speed로 채움
    """
    # max_speed 조회 (딕셔너리에서 직접)
    max_speed = max_speed_dict.get(link_id, 60.0)
    
    # 전체 가능한 base_time 생성
    if Processmode == 'date_range':
        end_time = base_start + timedelta(days=1) - timedelta(minutes=max_offset)
    elif Processmode == 'time_reference':
        end_time = base_start + timedelta(minutes=max_offset)
    
    if len(offsets) > 1:
        step = offsets[1] - offsets[0]
    else:
        step = max_offset
    
    all_base_times = pd.date_range(
        start=base_start, end=end_time, freq=f"{step}T"
    )
    
    # 데이터프레임을 datetime 인덱스로 변환 (최적화)
    link_df["datetime"] = pd.to_datetime(
        link_df["date"].astype(str) + " " + link_df["time"].astype(str)
    )
    link_df = link_df.set_index("datetime")["speed"].sort_index()
    
    # 유효한 base_times 필터
    valid = (all_base_times >= base_start) & (all_base_times <= link_df.index[-1] - timedelta(minutes=max_offset))
    base_times = all_base_times[valid]
    
    # exclude_missing_links가 true일 때는 데이터가 없으면 None 반환
    if exclude_missing_links and len(base_times) == 0:
        return None
    
    # 벡터화된 연산으로 텐서 생성
    link_tensor = []
    for bt in base_times:
        time_points = [bt + timedelta(minutes=o) for o in offsets]
        row = link_df.reindex(time_points).values
        if type == 'x':
            if exclude_missing_links:
                # 기존 방식: NaN 유지
                pass
            else:
                # 새로운 방식: NaN → max_speed
                row = np.where(np.isnan(row), max_speed, row)
        link_tensor.append(row)
    
    if len(link_tensor) == 0:
        # 유효한 시간이 하나도 없으면, exclude_missing_links에 따라 처리
        if exclude_missing_links:
            return None  # 기존 방식: None 반환
        else:
            # 새로운 방식: NaN tensor 반환
            nan_tensor = np.full(
                (len(all_base_times), len(offsets)),
                np.nan,
                dtype=np.float32
            )
            return nan_tensor
    
    # np.stack을 밖에서 처리
    try:
        tensor = np.stack(link_tensor)
        
        # exclude_missing_links가 true일 때는 NaN 비율이 높으면 None 반환
        if exclude_missing_links:
            nan_ratio = np.isnan(tensor).sum() / tensor.size
            if nan_ratio > 0.8:  # 80% 이상이 NaN이면 제외
                return None
        
        return tensor
    except Exception as e:
        # np.stack 실패 시 None 반환
        return None

def build_cluster_y_tensor_optimized(raw_data_dir, clusters, path_config, task_config, output_dir, logger, cluster_id=None, dates=None):
    """
    Raw 데이터에서 직접 클러스터 y 텐서 생성 (최적화된 방식)
    - Raw parquet 파일들을 읽어서 클러스터별로 직접 y 텐서 생성
    - 중간 parquet 파일 생성 없이 메모리에서 직접 처리
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 설정 로드
    interval = task_config['interval']
    interest_y = task_config['interest_y']
    train_ratio = task_config['train_ratio']
    
    # y offsets 계산 (예측할 미래 시간)
    offsets = [interval * i for i in range(interest_y)]
    
    # base_start 계산
    start_date = datetime.strptime(task_config['date_range']['start'], "%Y-%m-%d")
    base_start = start_date
    
    # MAX_SPD 정보 로드
    ITS_Link_path = path_config['data']['ITS_info']
    df = DBF(ITS_Link_path, encoding='cp949')
    Linkdata = pd.DataFrame(iter(df))
    selectedLink = Linkdata[['LINK_ID', 'MAX_SPD']]
    max_speed_dict = dict(zip(selectedLink['LINK_ID'], selectedLink['MAX_SPD']))
    
    # 처리할 날짜들 결정
    if dates is None:
        raw_files = sorted([f for f in os.listdir(raw_data_dir) if f.endswith('.parquet')])
        dates_to_process = [f.replace('.parquet', '') for f in raw_files]
    else:
        dates_to_process = dates
    
    logger.info(f"🔹 Raw 파일 {len(dates_to_process)}개에서 y 텐서 생성 시작")
    
    # 전체 시간 인덱스 생성
    full_time_index = pd.Index(pd.date_range(start="00:00", end="23:59", freq="5min").time)
    
    # 링크별 데이터를 저장할 딕셔너리
    link_data_dict = {}
    
    # Raw 파일들을 순차적으로 처리
    for date in dates_to_process:
        file_path = os.path.join(raw_data_dir, f"{date}.parquet")
        
        try:
            # Raw 파일 읽기
            df = pd.read_parquet(file_path)
            if df.empty:
                logger.warning(f"⚠️ 날짜 {date}: 읽어온 데이터가 없습니다.")
                continue
                
            logger.info(f"🔹 날짜 {date} y 텐서용 데이터 처리 중...")
            
            # 링크별로 데이터 정리
            df["insert_time"] = pd.to_datetime(df["insert_time"], format="%H:%M").dt.time
            grouped = df.groupby("link_id")
            
            for link_id, group in grouped:
                if link_id not in link_data_dict:
                    link_data_dict[link_id] = []
                
                # 시간별 데이터 정리
                group = group[["insert_time", "speed"]]
                group.set_index("insert_time", inplace=True)
                group = group.reindex(full_time_index)
                group["speed"] = group["speed"].astype("float32")
                
                # 날짜 정보 추가
                group = group.rename_axis("time").reset_index()
                group["date"] = date
                
                link_data_dict[link_id].append(group)
                
        except Exception as e:
            logger.error(f"❌ 날짜 {date} 처리 중 오류 발생: {e}")
            continue
    
    logger.info(f"✅ Raw 데이터 처리 완료: {len(link_data_dict)}개 링크")
    
    # 클러스터별 y 텐서 생성
    for cid, cluster_link_ids in clusters.items():
        if cluster_id is not None and cid != cluster_id:
            continue
            
        cluster_path = os.path.join(output_dir, str(cid))
        os.makedirs(cluster_path, exist_ok=True)
        
        logger.info(f"🔹 클러스터 {cid} y 텐서 생성 중... (링크 수: {len(cluster_link_ids)})")
        
        # 클러스터에 속한 링크들의 y 텐서 생성
        tensors = []
        valid_link_ids = []
        
        for lid in cluster_link_ids:
            if lid in link_data_dict:
                try:
                    # 링크 데이터를 하나의 DataFrame으로 합치기
                    link_df = pd.concat(link_data_dict[lid], ignore_index=True)
                    
                    # y 텐서 생성 (미래 예측용)
                    tensor = create_tensor_from_link_data_fast(
                        link_df, lid, base_start, offsets, max_offset, 
                        max_speed_dict, 'y', 'date_range'
                    )
                    
                    if tensor is not None:
                        tensors.append(tensor)
                        valid_link_ids.append(lid)
                        
                except Exception as e:
                    logger.warning(f"링크 {lid} y 텐서 생성 중 오류: {e}")
                    continue
        
        # y 텐서 저장 로직
        if tensors:
            T = min(t.shape[0] for t in tensors)
            tensors = [t[:T] for t in tensors]
            cluster_tensor = np.stack(tensors, axis=2)
            cluster_tensor = cluster_tensor[:, np.newaxis, :, :]  # (T, 1, N_links, interest_y)
            
            # train/val/test 분할 저장
            for option in ["train", "val", "test"]:
                if option == "train":
                    indices = np.arange(0, int(T * train_ratio[0]))
                elif option == "val":
                    indices = np.arange(int(T * train_ratio[0]), int(T * (train_ratio[0] + train_ratio[1])))
                elif option == "test":
                    indices = np.arange(int(T * (train_ratio[0] + train_ratio[1])), T)
                split_tensor = cluster_tensor[indices]
                save_path = os.path.join(cluster_path, f"{option}_ydata.npz")
                np.savez_compressed(save_path, split_tensor)
                logger.info(f"✅ {option} y 데이터셋 저장 완료: {save_path} | shape={split_tensor.shape}")
        else:
            logger.warning(f"⚠️ 클러스터 {cid}에 유효한 y 데이터 없음")
    
    logger.info(f"✅ 클러스터 y 텐서 생성 완료: {output_dir}에 저장됨")


# 모듈 레벨 함수들 (multiprocessing을 위해)
def process_cluster_batch_mp(args):
    """multiprocessing을 위한 클러스터 배치 처리 함수"""
    cid, link_ids, timestamps_with_offsets, date_data_cache, max_speed_dict, x_offsets, y_offsets, generate_y = args
    
    x_batch_data = []
    y_batch_data = []
    
    for timestamp_data in timestamps_with_offsets:
        current_timestamp, x_offset_data, y_offset_data = timestamp_data
        
        # x 데이터 처리
        x_row = process_links_vectorized_mp(link_ids, current_timestamp, x_offsets, date_data_cache, max_speed_dict)
        x_batch_data.append(x_row)
        
        # y 데이터 처리
        if generate_y and y_offsets:
            y_row = process_links_vectorized_mp(link_ids, current_timestamp, y_offsets, date_data_cache, max_speed_dict)
            y_batch_data.append(y_row)
    
    return cid, x_batch_data, y_batch_data

def process_links_vectorized_mp(link_ids, current_timestamp, offsets, date_data_cache, max_speed_dict):
    """multiprocessing 호환 vectorized 링크 처리"""
    n_links = len(link_ids)
    n_features = len(offsets)
    result = np.zeros((n_links, n_features), dtype=np.float32)
    
    # 오프셋별로 필요한 날짜와 시간 미리 계산
    offset_data = {}
    for feat_idx, offset in enumerate(offsets):
        target_dt = current_timestamp + timedelta(minutes=offset)
        target_date_str = target_dt.strftime("%Y%m%d")
        target_time = target_dt.time()
        
        if target_date_str not in offset_data:
            offset_data[target_date_str] = {}
        if target_time not in offset_data[target_date_str]:
            offset_data[target_date_str][target_time] = []
        offset_data[target_date_str][target_time].append(feat_idx)
    
    # 날짜별, 시간별로 일괄 조회
    for date_str, time_dict in offset_data.items():
        if date_str in date_data_cache and date_data_cache[date_str] is not None:
            raw_data = date_data_cache[date_str]
            
            for target_time, feat_indices in time_dict.items():
                # 해당 시간의 모든 링크 데이터를 한 번에 조회
                time_data = raw_data[raw_data['insert_time'] == target_time]
                
                if not time_data.empty:
                    # pandas의 merge를 사용하여 빠른 조회
                    link_df = pd.DataFrame({'link_id': link_ids})
                    merged = link_df.merge(time_data[['link_id', 'speed']], on='link_id', how='left')
                    
                    for link_idx, (link_id, speed) in enumerate(zip(merged['link_id'], merged['speed'])):
                        speed_value = speed if not pd.isna(speed) else max_speed_dict.get(link_id, 60.0)
                        for feat_idx in feat_indices:
                            result[link_idx, feat_idx] = speed_value
                else:
                    # 시간 데이터가 없으면 모든 링크에 대해 max_speed 사용
                    for link_idx, link_id in enumerate(link_ids):
                        speed_value = max_speed_dict.get(link_id, 60.0)
                        for feat_idx in feat_indices:
                            result[link_idx, feat_idx] = speed_value
        else:
            # 날짜 데이터가 없으면 모든 링크에 대해 max_speed 사용
            for target_time, feat_indices in time_dict.items():
                for link_idx, link_id in enumerate(link_ids):
                    speed_value = max_speed_dict.get(link_id, 60.0)
                    for feat_idx in feat_indices:
                        result[link_idx, feat_idx] = speed_value
    
    return result

def process_raw_to_cluster_tensor_h5(raw_data_dir, clusters, path_config, task_config, output_dir, logger, mode='train', cluster_id=None, dates=None, generate_y=False):
    """
    최고성능 H5 기반 텐서 생성 - multiprocessing + vectorized
    - multiprocessing으로 진짜 병렬 처리 (GIL 우회)
    - vectorized pandas 연산으로 성능 최적화
    - 모든 클러스터 H5 파일을 동시에 열어서 처리
    """
    import h5py
    import multiprocessing as mp
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 설정 로드
    interval = task_config['interval']
    interest = task_config['interest']
    interest_y = task_config.get('interest_y', 0)
    train_ratio = task_config['train_ratio']
    
    # x offsets 계산
    x_offsets = []
    for base in interest:
        x_offsets.extend([interval * i for i in range(base, base + 12)])
    x_offsets = sorted(x_offsets)
    
    # y offsets 계산
    y_offsets = []
    if generate_y and interest_y > 0:
        y_offsets = [interval * i for i in range(interest_y)]
    elif generate_y:
        y_offsets = [interval * i for i in range(12)]  # 기본값
    
    # 기간 설정
    start_date = datetime.strptime(task_config['date_range']['start'], "%Y-%m-%d")
    end_date = datetime.strptime(task_config['date_range']['end'], "%Y-%m-%d")
    
    # 전체 타임스탬프 생성
    total_minutes = int((end_date - start_date).total_seconds() / 60)
    total_timestamps = total_minutes // interval
    
    # train/val/test 분할
    train_end = int(total_timestamps * train_ratio[0])
    val_end = int(total_timestamps * (train_ratio[0] + train_ratio[1]))
    
    split_info = {
        'train': (0, train_end),
        'val': (train_end, val_end),
        'test': (val_end, total_timestamps)
    }
    
    logger.info(f"🎯 전체 타임스탬프: {total_timestamps}개")
    logger.info(f"📊 데이터 분할: train={train_end}, val={val_end-train_end}, test={total_timestamps-val_end}")
    
    # MAX_SPD 정보 로드
    ITS_Link_path = path_config['data']['ITS_info']
    df = DBF(ITS_Link_path, encoding='cp949')
    Linkdata = pd.DataFrame(iter(df))
    selectedLink = Linkdata[['LINK_ID', 'MAX_SPD']]
    max_speed_dict = dict(zip(selectedLink['LINK_ID'], selectedLink['MAX_SPD']))
    
    # 클러스터 필터링
    if cluster_id is not None:
        filtered_clusters = {cluster_id: clusters[cluster_id]}
    else:
        filtered_clusters = clusters  # 모든 클러스터 처리
    
    logger.info(f"🔹 {len(filtered_clusters)}개 클러스터 최고성능 H5 처리 중...")
    
    # 각 클러스터별 정보 및 H5 파일 경로 준비
    cluster_info = {}
    for cid, cluster_link_ids in filtered_clusters.items():
        cluster_path = os.path.join(output_dir, str(cid))
        os.makedirs(cluster_path, exist_ok=True)
        
        N_links = len(cluster_link_ids)
        x_features = len(x_offsets)
        y_features = len(y_offsets) if generate_y and y_offsets else 0
        
        cluster_info[cid] = {
            'link_ids': cluster_link_ids,
            'N_links': N_links,
            'x_features': x_features,
            'y_features': y_features,
            'cluster_path': cluster_path
        }
        
        logger.info(f"  📋 클러스터 {cid}: {N_links}개 링크")
    
    # 각 split별로 모든 클러스터 H5 파일 동시 처리
    for split_name, (start_idx, end_idx) in split_info.items():
        if start_idx >= end_idx:
            continue
            
        T = end_idx - start_idx
        logger.info(f"🔹 {split_name} 세트 처리: {T}개 타임스탬프")
        
        # 모든 클러스터의 H5 파일들 동시에 열기
        h5_files = {}
        x_datasets = {}
        y_datasets = {}
        
        for cid, info in cluster_info.items():
            # H5 파일 경로
            if mode == 'train':
                h5_path = os.path.join(info['cluster_path'], f"{split_name}_data.h5")
            else:
                h5_path = os.path.join(info['cluster_path'], f"predict_data.h5")
            
            # H5 파일 열기
            h5_files[cid] = h5py.File(h5_path, 'w')
            
            # 데이터셋 생성
            x_datasets[cid] = h5_files[cid].create_dataset('x_data', (T, 1, info['N_links'], info['x_features']), dtype=np.float32)
            
            if generate_y and info['y_features'] > 0:
                y_datasets[cid] = h5_files[cid].create_dataset('y_data', (T, 1, info['N_links'], info['y_features']), dtype=np.float32)
            
            logger.info(f"    📁 클러스터 {cid} H5: {h5_path}")
            logger.info(f"      📏 x: {x_datasets[cid].shape}")
            if cid in y_datasets:
                logger.info(f"      📏 y: {y_datasets[cid].shape}")
        
        # 배치별 처리 (하루씩)
        batch_size = 288  # 하루치 타임스탬프
        n_batches = (T + batch_size - 1) // batch_size
        
        for batch_idx in range(n_batches):
            start_t = batch_idx * batch_size
            end_t = min(start_t + batch_size, T)
            
            logger.info(f"  🔄 배치 {batch_idx+1}/{n_batches}: {start_t}~{end_t-1} 타임스탬프")
            
            # 해당 배치에서 필요한 날짜들 미리 확인
            batch_dates = set()
            for t_idx in range(start_t, end_t):
                global_t_idx = start_idx + t_idx
                current_timestamp = start_date + timedelta(minutes=global_t_idx * interval)
                batch_dates.add(current_timestamp.strftime("%Y%m%d"))
            
            # 필요한 날짜별 데이터 미리 로드 (한 번만!)
            date_data_cache = {}
            for date_str in batch_dates:
                file_path = os.path.join(raw_data_dir, f"{date_str}.parquet")
                try:
                    raw_data = pd.read_parquet(file_path)
                    if not raw_data.empty:
                        raw_data["insert_time"] = pd.to_datetime(raw_data["insert_time"], format="%H:%M").dt.time
                        raw_data = raw_data[["link_id", "insert_time", "speed"]]
                        raw_data["speed"] = raw_data["speed"].astype("float32")
                        date_data_cache[date_str] = raw_data
                        logger.info(f"    📅 {date_str} 로드 완료")
                    else:
                        date_data_cache[date_str] = None
                except Exception as e:
                    logger.warning(f"    ⚠️ {date_str} 로드 실패: {e}")
                    date_data_cache[date_str] = None
            
            # 배치 내 타임스탬프별 처리 (단순 순차 방식)
            logger.info(f"    🔄 순차 처리 시작...")
            
            for t_idx in range(start_t, end_t):
                global_t_idx = start_idx + t_idx
                current_timestamp = start_date + timedelta(minutes=global_t_idx * interval)
                
                # 모든 클러스터 동시 처리 (vectorized)
                for cid, info in cluster_info.items():
                    # x 데이터를 vectorized로 처리
                    x_row = process_links_vectorized_mp(info['link_ids'], current_timestamp, x_offsets, date_data_cache, max_speed_dict)
                    x_datasets[cid][t_idx, 0, :, :] = x_row
                    
                    # y 데이터를 vectorized로 처리
                    if generate_y and info['y_features'] > 0:
                        y_row = process_links_vectorized_mp(info['link_ids'], current_timestamp, y_offsets, date_data_cache, max_speed_dict)
                        y_datasets[cid][t_idx, 0, :, :] = y_row
                
                # 진행 상황 로깅 (10%마다)
                if (t_idx - start_t + 1) % max(1, (end_t - start_t) // 10) == 0:
                    progress = (t_idx - start_t + 1) / (end_t - start_t) * 100
                    logger.info(f"      📈 진행: {progress:.1f}% ({t_idx - start_t + 1}/{end_t - start_t})")
            
            # 진행 상황 로깅
            progress = (batch_idx + 1) / n_batches * 100
            logger.info(f"    ✅ 순차 배치 {batch_idx+1} 완료 ({progress:.1f}%), 메모리 정리됨")
            
            # 배치 완료 후 캐시 정리
            del date_data_cache
        
        # 모든 H5 파일 닫기
        for cid in h5_files:
            h5_files[cid].close()
            logger.info(f"  ✅ 클러스터 {cid} {split_name} H5 파일 완료")
    
    # timestamp 생성 및 저장
    timestamp = logger.timestamp
    np.save(os.path.join(output_dir, "timestamp.npy"), timestamp)
    
    logger.info(f"🎉 최고성능 multiprocessing H5 기반 텐서 생성 완료!")
    return timestamp


def get_speed_value_batch(target_dt, link_id, date_data_cache, max_speed_dict):
    """배치 처리용 속도 값 조회"""
    target_date_str = target_dt.strftime("%Y%m%d")
    target_time = target_dt.time()
    
    # 캐시된 데이터에서 조회
    if target_date_str in date_data_cache and date_data_cache[target_date_str] is not None:
        raw_data = date_data_cache[target_date_str]
        link_data = raw_data[(raw_data['link_id'] == link_id) & (raw_data['insert_time'] == target_time)]
        
        if len(link_data) > 0:
            speed = link_data['speed'].iloc[0]
            return speed if not pd.isna(speed) else max_speed_dict.get(link_id, 60.0)
    
    # 기본값 반환
    return max_speed_dict.get(link_id, 60.0)

