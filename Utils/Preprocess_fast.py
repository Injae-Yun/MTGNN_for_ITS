import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dbfread import DBF
from collections import defaultdict
import h5py
import gc
from Utils.share import Scaler_tool
import holidays
__all__ = [
    "process_raw_to_cluster_tensor_pivot",
    "process_predict_to_cluster_tensor_pivot",
]

def _build_pivot_for_date(
    date_str: str,
    raw_data_dir: str,
    all_links: np.ndarray,
    link_to_idx: dict[int,int],
    max_speed_dict: dict[int,float]
) -> np.ndarray:
    """주어진 날짜(YYYYMMDD)의 raw parquet → (links, 288) pivot 배열 반환"""

    # 1) 모든 링크별 기본값 배열(288칸) 생성 (max_speed_dict 기반)
    default_speeds = np.array(
        [max_speed_dict.get(lid, 60.0) for lid in all_links],
        dtype=np.float32
    )
    # 각 링크마다 288개 슬롯으로 복제
    pivot = default_speeds[:, None].repeat(288, axis=1)

    # 2) 파일 없음 혹은 빈 파일일 때 기본 반환
    file_path = os.path.join(raw_data_dir, f"{date_str}.parquet")
    if not os.path.exists(file_path):
        return pivot

    df = pd.read_parquet(file_path)
    df["link_id"] = df["link_id"].astype(int)
    if df.empty:
        return pivot

    # 3) slot 계산 (0~287)
    t = pd.to_datetime(df["insert_time"], format="%H:%M")
    slots = (t.dt.hour * 12 + t.dt.minute // 5)
    mask_bad = slots.isna()
    # 4) link_id → pivot row index 매핑, 매핑 실패한 행은 필터링
    mapped = df["link_id"].map(link_to_idx)
    valid = mapped.notna()
    mask = (~mask_bad) & valid
    rows   = mapped[mask].astype(np.int32).to_numpy()
    slots  = slots[mask]
    slots = slots.to_numpy(dtype=np.int32)
    speeds = df.loc[mask, "speed"].to_numpy(dtype=np.float32)
     
    del df, t, mapped, valid, mask  # 메모리 절약을 위해 불필요한 변수 삭제
    # 5) 벡터 인덱싱 한 번에 값 덮어쓰기
    pivot[rows, slots] = speeds

    return pivot

def for_LGBM_build_tensors(
    raw_data_dir: str,
    link_ids: list[int],
    path_config: dict,
    task_config: dict,
    output_dir: str,
    mode = 'train',
    logger=None
    ):
    """X(tensors.h5)와 Y(y_tensors.h5)를 통합된 로직으로 생성하는 함수"""
    os.makedirs(output_dir, exist_ok=True)

    # --- 1) Task 설정 불러오기 ---
    logger.info(f"1. 설정 불러오기 (mode: {mode})")
    start_date = datetime.strptime(task_config["date_range"]["start"], "%Y-%m-%d")
    end_date = datetime.strptime(task_config["date_range"]["end"], "%Y-%m-%d")
    interval = task_config["interval"]  # e.g., 5

    # X 피처 설정 (build_tensor 로직)
    x_offsets = task_config['far_time_process']["interest"]  # e.g., [-21, -14, -7, -1, 0]
    
    # Y 피처 설정 (build_y_tensor 로직)
    # y는 예측 시점(base_dt)의 값이므로 offset이 0인 것과 같음

    all_links = np.array(link_ids, dtype=int)
    link_to_idx = {lid: i for i, lid in enumerate(all_links)}
    
    ITS_path = path_config["data"]["ITS_info"]
    link_df = pd.DataFrame(iter(DBF(ITS_path, encoding="cp949")))
    link_df["LINK_ID"] = link_df["LINK_ID"].astype(int)
    max_speed_dict = dict(zip(link_df["LINK_ID"], link_df["MAX_SPD"]))
    kr_holidays = holidays.KR()

    # --- 3) 데이터셋 크기 및 분할 계획 ---
    logger.info("2. 데이터셋 크기 및 파일 경로 설정")
    total_days = (end_date - start_date).days + 1
    T = 1440 // interval  # 하루 당 타임스탬프 수
    L = len(all_links)    # 링크 수

    if mode == 'train':
        C = len(x_offsets) + 1 # X 채널 수 (speed 채널 + holiday 채널)
    elif mode == 'predict':
        C = len(x_offsets)  # 예측 시에는 holiday 채널이 필요 없음

    if mode == 'train':
        N = total_days * T # 전체 유효 샘플 수
        train_frac, val_frac, _ = task_config['train_ratio']
        n_train = int(total_days * train_frac) * T
        n_val = int(total_days * val_frac) * T
        n_test = N - n_train - n_val

        splits = {"train": (0, n_train), "val": (n_train, n_train + n_val), "test": (n_train + n_val, N)}
        logger.info(f"학습 데이터셋 생성. 전체 샘플 수: {N} (Train: {n_train}, Val: {n_val}, Test: {n_test})")
        h5_x_path = os.path.join(output_dir, "tensors.h5")
        h5_y_path = os.path.join(output_dir, "y_tensors.h5")
    elif mode == 'predict':
        N = total_days * T
        logger.info(f"추론 데이터셋 생성. 전체 샘플 수: {N}")
        h5_x_path = os.path.join(output_dir, "predict_tensors.h5")
        h5_y_path = None  # 추론 시에는 Y 텐서 불필요
    else:
        raise ValueError("mode는 'train' 또는 'predict' 여야 합니다.")
    hf_y = h5py.File(h5_y_path, "w") if h5_y_path else None
    with h5py.File(h5_x_path, "w") as hf_x:
        # 데이터셋 생성
        if mode == 'train':
            ds_x = {s: hf_x.create_dataset(s, (size, C, L), dtype="float16") for s, (_, size) in {"train": (0, n_train), "val": (0, n_val), "test": (0, n_test)}.items() if size > 0}
            ds_y = {s: hf_y.create_dataset(s, (size, 1, L), dtype="float16") for s, (_, size) in {"train": (0, n_train), "val": (0, n_val), "test": (0, n_test)}.items() if size > 0}
        else: # predict 모드
            ds_x = {"predict": hf_x.create_dataset("predict", (N, C, L), dtype="float16")}
            ds_y = {} # Y 데이터셋 없음
        # --- 5) 날짜별 작업 계획 수립 (핵심) ---
        logger.info("3. 작업 계획 수립")
        # key: 'YYYYMMDD', value: list of (target_split, target_idx, base_dt)
        date_tasks = defaultdict(list)
        if mode == 'train':
            for i in range(total_days):
                base_date = start_date + timedelta(days=i)
                for t_idx in range(T):
                    current_sample_idx = i * T + t_idx
                    current_split, split_rel_idx = "", 0
                    if current_sample_idx < splits["train"][1]:
                        current_split, split_rel_idx = "train", current_sample_idx
                    elif current_sample_idx < splits["val"][1]:
                        current_split, split_rel_idx = "val", current_sample_idx - splits["val"][0]
                    else:
                        current_split, split_rel_idx = "test", current_sample_idx - splits["test"][0]
                    
                    date_tasks[base_date.strftime("%Y%m%d")].append(("Y", current_split, split_rel_idx, t_idx))
                    for chan_idx, offset in enumerate(x_offsets):
                        fetch_date = base_date + timedelta(days=offset)
                        date_tasks[fetch_date.strftime("%Y%m%d")].append(("X", current_split, split_rel_idx, t_idx, chan_idx))
        else: # predict
            for i in range(total_days):
                base_date = start_date + timedelta(days=i)
                for t_idx in range(T):
                    current_sample_idx = i * T + t_idx
                    for chan_idx, offset in enumerate(x_offsets):
                        fetch_date = base_date + timedelta(days=offset)
                        date_tasks[fetch_date.strftime("%Y%m%d")].append(("X", "predict", current_sample_idx, t_idx, chan_idx))

        # --- 4) 데이터 처리 및 저장 ---
        logger.info("4. 데이터 처리 및 저장 시작")
        Scaler = Scaler_tool()
        
        # 날짜순으로 정렬하여 pivot 로드
        for d_str, tasks in sorted(date_tasks.items()):
            logger.info(f"  - Pivot 로딩: {d_str}")
            pivot = _build_pivot_for_date(d_str, raw_data_dir, all_links, link_to_idx, max_speed_dict)
            
            # Y와 X 모두 동일한 스케일러로 변환
            pivot_scaled = Scaler.transform(pivot).astype(np.float16)
            
            for task_type, split, s_idx, t_idx, *chan_info in tasks:
                if task_type == "Y":
                    ds_y[split][s_idx, 0, :] = pivot_scaled[:, t_idx]
                elif task_type == "X":
                    chan_idx = chan_info[0]
                    ds_x[split][s_idx, chan_idx, :] = pivot_scaled[:, t_idx]
            gc.collect()
        # Holiday 피처 추가 (모든 split에 대해)
        if mode == 'train':
            logger.info("5. Holiday 피처 추가 (주말 포함)")
            for i in range(total_days):
                base_date = start_date + timedelta(days=i)
                is_holiday = float(base_date.weekday() >= 5 or base_date in kr_holidays)
                hch_day = np.full((T, L), is_holiday, dtype=np.float16)
                
                day_start_idx, day_end_idx = i * T, (i + 1) * T
                for split_name, (s_start, s_end) in splits.items():
                    overlap_start, overlap_end = max(day_start_idx, s_start), min(day_end_idx, s_end)
                    if overlap_start < overlap_end:
                        write_start, write_end = overlap_start - s_start, overlap_end - s_end
                        read_start, read_end = overlap_start - day_start_idx, overlap_end - day_start_idx
                        ds_x[split_name][write_start:write_end, -1, :] = hch_day[read_start:read_end]
    

        # 주기적으로 flush 하는 로직을 추가하면 더 안정적입니다.
        hf_x.flush()
        if hf_y:
            hf_y.flush()

    logger.info(f"✅ X HDF5 생성 완료: {h5_x_path}")
    if h5_y_path:
        logger.info(f"✅ Y HDF5 생성 완료: {h5_y_path}")


def process_raw_to_cluster_tensor_pivot(
    raw_data_dir: str,
    clusters: dict[int, list[int]],
    path_config: dict,
    task_config: dict,
    output_dir: str,
    logger,
    generate_y: bool = True
):
    """pivot 기반 고속 tensor 생성 (train/val/test 전부 처리)"""
    os.makedirs(output_dir, exist_ok=True)

    # --- 1) Task 설정 불러오기 ---
    interval     = task_config["interval"]         # 예: 5 (분)
    interest     = task_config["interest"]         # 예: [-12, -11, ..., 0]
    interest_y   = task_config.get("interest_y", 12)
    train_ratio  = task_config["train_ratio"]      # 예: (0.6,0.2,0.2)
    scaler = Scaler_tool()
    # x/y offsets (분 단위)
    x_offsets = np.array(
        [m for base in interest for m in range(base*interval, (base+12)*interval, interval)],
        dtype=int
    )
    x_features = len(x_offsets)
    y_offsets = np.array([m*interval for m in range(interest_y)], dtype=int)
    y_features = len(y_offsets)

    # --- 2) 링크 목록 & 매핑 ---
    ITS_path   = path_config["data"]["ITS_info"]
    link_df    = pd.DataFrame(iter(DBF(ITS_path, encoding="cp949")))
    all_links  = link_df["LINK_ID"].to_numpy(dtype=int)
    link_df["LINK_ID"] = link_df["LINK_ID"].astype(int)
    link_to_idx= {lid: i for i, lid in enumerate(all_links)}
    max_speed_dict = dict(zip(link_df["LINK_ID"], link_df["MAX_SPD"]))

    # --- 3) 클러스터별 링크 인덱스 ---
    cluster_rows = {
        cid: np.array([link_to_idx[int(l)] for l in links if int(l) in link_to_idx], dtype=int)
        for cid, links in clusters.items()
    }

    # --- 4) 날짜 범위 & split 계산 ---
    start_date = datetime.strptime(task_config["date_range"]["start"], "%Y-%m-%d")
    end_date   = datetime.strptime(task_config["date_range"]["end"],   "%Y-%m-%d")
    total_minutes = int((end_date - start_date).total_seconds() / 60)
    total_ts      = total_minutes // interval
    train_end = int(total_ts * train_ratio[0])
    val_end   = int(total_ts * (train_ratio[0] + train_ratio[1]))
    splits = {
        "train": (0, train_end),
        "val":   (train_end, val_end),
        "test":  (val_end, total_ts),
    }
    logger.info(f"🎯 pivot 방식: 전체 타임스탬프 {total_ts} (interval={interval}분)")

    # --- 5) 날짜별 작업 매핑 함수 ---
    def _date_str(dt: datetime) -> str:
        return dt.strftime("%Y%m%d")
    def _slot_of(dt: datetime) -> int:
        return dt.hour * 12 + dt.minute // 5

    def build_date_tasks(split_s: int, split_e: int) -> dict[str, dict[str, list[tuple[int,int,int]]]]:
        """
        { 'YYYYMMDD': { 'x': [(rel_t, slot, k), ...], 'y': [...] }, ... }
        """
        tasks = defaultdict(lambda: {"x": [], "y": []})
        for rel_t in range(split_s, split_e):
            base_dt = start_date + timedelta(minutes=rel_t * interval)
            idx = rel_t - split_s  # dataset index 0..T-1
            # x
            for k, off in enumerate(x_offsets):
                dt_off = base_dt + timedelta(minutes=int(off))
                d = _date_str(dt_off)
                tasks[d]["x"].append((idx, _slot_of(dt_off), k))
            # y
            if generate_y:
                for k, off in enumerate(y_offsets):
                    dt_off = base_dt + timedelta(minutes=int(off))
                    d = _date_str(dt_off)
                    tasks[d]["y"].append((idx, _slot_of(dt_off), k))
        return tasks

    # --- 6) split별 처리 ---
    for split_name, (split_s, split_e) in splits.items():
        if split_s >= split_e:
            continue
        T = split_e - split_s
        logger.info(f"🔹 {split_name} 세트 처리: {T} ts")

        # 6-1) H5 파일 & 데이터셋 생성
        h5_files, x_dsets, y_dsets = {}, {}, {}
        for cid, rows in cluster_rows.items():
            cdir = os.path.join(output_dir, str(cid))
            os.makedirs(cdir, exist_ok=True)
            h5_path = os.path.join(cdir, f"{split_name}_data.h5")
            h5_files[cid]   = h5py.File(h5_path, "w")
            x_dsets[cid]    = h5_files[cid].create_dataset("x", (T,1,len(rows),x_features), 
                                dtype="float16",    compression="lzf" ,    chunks=(1, 1, len(rows), x_features))
            if generate_y:
                y_dsets[cid] = h5_files[cid].create_dataset("y", (T,1,len(rows),y_features), 
                                dtype="float16",    compression="lzf" ,    chunks=(1, 1, len(rows), y_features))
            

        # 6-2) 날짜→작업 매핑
        date_tasks = build_date_tasks(split_s, split_e)

        # 6-3) 날짜별 피벗 로드 & 반영
        for dstr, task in sorted(date_tasks.items()):
            pivot = _build_pivot_for_date(
                dstr, raw_data_dir, all_links, link_to_idx, max_speed_dict
            )
            if pivot is None or not isinstance(pivot, np.ndarray):
                logger.error(f"[ERROR] Pivot failed at date {dstr}")
                continue
            pivot_sc = scaler.transform(pivot)
            del pivot
            # x process
            for rel_idx, slot, k in task["x"]:
                for cid, rows in cluster_rows.items():
                    if cid>10: continue
                    x_dsets[cid][rel_idx,0,:,k] = pivot_sc[rows, slot]
                    

            # y process
            if generate_y:
                for rel_idx, slot, k in task["y"]:
                    for cid, rows in cluster_rows.items():
                        if cid>10: continue
                        y_dsets[cid][rel_idx,0,:,k] = pivot_sc[rows, slot]

            # flush process
            for cid in cluster_rows.keys():
                if cid>10: continue
                x_dsets[cid].flush()
                if generate_y:
                    y_dsets[cid].flush()


            del pivot_sc
            gc.collect()  # 메모리 정리
            logger.info(f"  ▶ {split_name} {dstr} 반영 완료")

        # 6-4) H5 닫기
        for h5 in h5_files.values():
            h5.close()
        logger.info(f"✅ {split_name} 세트 완료")
        

    logger.info("🎉 pivot 기반 tensor 생성 완료")


def process_predict_to_cluster_tensor_pivot(
    raw_data_dir: str,
    clusters: dict[int, list[int]],
    path_config: dict,
    task_config: dict,
    output_dir: str,
    logger,
):
    """time_reference 기반 고속 predict 입력 생성

    - 각 클러스터 폴더에 `predict_data.npy`를 생성 (shape: [T, 1, N_links, 36])
    - 스케일은 학습 시 저장된 `scaler.npy`(클러스터별 mean/std)를 사용
    - 반환값: 예측 기준 타임스탬프 리스트(list[str])
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1) 설정 로드
    interval = task_config["interval"]
    interest = task_config["interest"]
    # x 오프셋: 분 단위
    x_offsets = np.array(
        [m for base in interest for m in range(base * interval, (base + 12) * interval, interval)],
        dtype=int,
    )
    x_features = len(x_offsets)

    # time_reference: dict 또는 list(dict) 모두 지원
    time_ref_cfg = task_config.get("time_reference")
    if isinstance(time_ref_cfg, dict):
        time_refs = [time_ref_cfg]
    elif isinstance(time_ref_cfg, list):
        time_refs = time_ref_cfg
    else:
        raise ValueError("time_reference 설정이 잘못되었습니다. dict 또는 list 형태여야 합니다.")

    # 2) 링크 목록 & 매핑
    ITS_path = path_config["data"]["ITS_info"]
    link_df = pd.DataFrame(iter(DBF(ITS_path, encoding="cp949")))
    all_links = link_df["LINK_ID"].to_numpy(dtype=int)
    link_df["LINK_ID"] = link_df["LINK_ID"].astype(int)
    link_to_idx = {lid: i for i, lid in enumerate(all_links)}
    max_speed_dict = dict(zip(link_df["LINK_ID"], link_df["MAX_SPD"]))

    # 클러스터별 행 인덱스 준비
    cluster_rows = {
        cid: np.array([link_to_idx[int(l)] for l in links if int(l) in link_to_idx], dtype=int)
        for cid, links in clusters.items()
    }

    # 3) 도우미: 날짜 문자열과 슬롯 계산
    def _date_str(dt: datetime) -> str:
        return dt.strftime("%Y%m%d")

    def _slot_of(dt: datetime) -> int:
        return dt.hour * 12 + dt.minute // 5

    # 4) 샘플별 작업 구성: { 'YYYYMMDD': [('sample_idx', 'slot', 'k'), ...] }
    date_tasks = {}
    timestamps = []
    for s_idx, tr in enumerate(time_refs):
        base_date = tr.get("date")
        base_time = tr.get("time") or "00:00"
        base_dt = datetime.strptime(f"{base_date} {base_time}", "%Y-%m-%d %H:%M")
        timestamps.append(base_dt.strftime("%Y-%m-%d %H:%M"))

        for k, off in enumerate(x_offsets):
            dt_off = base_dt + timedelta(minutes=int(off))
            d = _date_str(dt_off)
            date_tasks.setdefault(d, []).append((s_idx, _slot_of(dt_off), k))

    # 5) 날짜별 피벗 미리 계산 (raw → pivot)
    pivot_cache = {}
    for dstr in sorted(date_tasks.keys()):
        pivot_cache[dstr] = _build_pivot_for_date(
            dstr, raw_data_dir, all_links, link_to_idx, max_speed_dict
        )

    # 6) 클러스터별 predict_data.npy 생성
    S = len(time_refs)
    for cid, rows in cluster_rows.items():
        cdir = os.path.join(output_dir, str(cid))
        os.makedirs(cdir, exist_ok=True)

        # 스케일러 로드 (학습 시 저장됨)
        scaler_path = os.path.join(cdir, "scaler.npy")
        if os.path.exists(scaler_path):
            scaler_obj = np.load(scaler_path, allow_pickle=True).item()
            mean = scaler_obj.get("mean", 45.0)
            std = scaler_obj.get("std", 25.0)
        else:
            logger.warning(f"⚠️ 클러스터 {cid}: scaler.npy가 없어 기본 스케일 사용(45/25)")
            mean, std = 45.0, 25.0
        scaler = Scaler_tool(mean, std)

        x_pred = np.zeros((S, 1, len(rows), x_features), dtype=np.float32)

        for dstr, task_list in date_tasks.items():
            pivot_raw = pivot_cache[dstr]
            if pivot_raw is None:
                logger.error(f"[ERROR] Pivot 생성 실패: {dstr}")
                continue
            for s_idx, slot, k in task_list:
                values = pivot_raw[rows, slot]
                x_pred[s_idx, 0, :, k] = scaler.transform(values)

        save_path = os.path.join(cdir, "predict_data.npy")
        np.save(save_path, x_pred)
        logger.info(f"✅ 클러스터 {cid} predict_data 저장: {save_path} | shape={x_pred.shape}")

    logger.info("🎉 pivot 기반 predict 입력 생성 완료")
    return timestamps