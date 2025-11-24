import torch
import numpy as np
import os
import pickle

from sklearn.metrics import mean_absolute_percentage_error
from Core.net import gtnet
from collections import defaultdict
import matplotlib.pyplot as plt
from Utils.share import Scaler_tool
from Utils.share import load_dataset

def load_link_ids(filepath):
    with open(filepath, 'r') as f:
        return [int(line.strip()) for line in f.readlines()]

    
def default_link_entry():
    return {"true": defaultdict(list), "pred": defaultdict(list)}


def compute_individual_mape(model_dir, cluster_list, logger, batch_size=256, cache_dir='Cache',cash_use=True, cluster_id=None):
    os.makedirs(cache_dir, exist_ok=True)
    results = {}
    link_dict = defaultdict(default_link_entry)
    for cid, link_list  in cluster_list.items():
        if cluster_id is not None and cid not in cluster_id: # 특정 클러스터 ID only 처리
            continue
        model_path = os.path.join(model_dir, f'{cid}_best.pth')
        data_path = os.path.join(model_dir, f'{cid}')
        os.makedirs(os.path.join(cache_dir,f"{cid}"), exist_ok=True)        
        results_path = os.path.join(cache_dir,f"{cid}", "_results.pkl")
        linkdict_path = os.path.join(cache_dir,f"{cid}", "_link_dict.pkl")

        # 🔄 캐시 로딩
        if os.path.exists(results_path) and os.path.exists(linkdict_path) and cash_use==True:
            print("📦 Loading results and link_dict from cache...")
            with open(results_path, "rb") as f:
                results[cid] = pickle.load(f)
            with open(linkdict_path, "rb") as f:
                cached_link_dict = pickle.load(f)
                # 캐시된 값 병합
                for link_id, data in cached_link_dict.items():
                    link_dict[link_id]["true"].update(data["true"])
                    link_dict[link_id]["pred"].update(data["pred"])
            logger.info(f"✅ MAPE for model {cid} loaded from cache: {results[cid]:.4f}")
            continue

        # Load model
        torch.serialization.add_safe_globals({'net.gtnet': gtnet})
        model = torch.load(model_path, map_location='cuda',weights_only=False)
        model.eval()

        # Load data
        dataloader = load_dataset(data_path, batch_size, None, logger, None)
        x_raw = dataloader['test_loader'].xs
        y_raw = dataloader['test_loader'].ys
        total = x_raw.shape[0]
        if batch_size > total:
            batch_size = total
        
        Scaler=dataloader['scaler']
        mean, std = Scaler.mean, Scaler.std
        scaler = Scaler_tool(mean,std)
        #x_normalized = scaler.transform(x_raw)         # 정규화
        all_trues, all_preds=[],[]

        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            b_size= end - start
            x_batch = torch.tensor(x_raw[start:end], dtype=torch.float32).to('cuda')
            # (B, 1, N, 36)
            y_batch = torch.tensor(y_raw[start:end], dtype=torch.float32).to('cuda')  
            # (B, 1, N, 12)
            #if x_batch.shape[2] != 36: # 데이터 로더가 이미 transpose를 했다면 건너뛰기
            #     x_batch = x_batch.transpose(2, 3)
            shape = x_batch.shape
            differ = x_batch[..., 1:] - x_batch[..., :-1]
            differ = torch.cat([torch.zeros_like(x_batch[..., :1]), differ], dim=-1)

            with torch.no_grad():
                y_pred = model(x_batch, differ, shape)[:, :, :, 0].cpu().numpy()  # (B, 12, N)
                y_true = y_batch[:, 0, :, :].cpu().numpy()  # (B, N, 12)
                y_true = y_true.transpose(0, 2, 1)  # (B, N, 12) -> (B, 12, N)  
                all_preds.append(y_pred)
                all_trues.append(y_true)  # shape: (B, N)

            for idx, link_id in enumerate(link_list):
                yt = y_true[:, :, idx]
                yp = y_pred[:, :, idx]
                link_dict[link_id]["true"][cid].append(yt)
                link_dict[link_id]["pred"][cid].append(yp)  # 2D로 만들어 누적

            # GPU 메모리 정리
            del x_batch, y_batch, y_pred, y_true, differ
            torch.cuda.empty_cache()

        y_pred_np = np.concatenate(all_preds, axis=0).flatten()
        y_true_np = np.concatenate(all_trues, axis=0).flatten()

        # NaN mask
        mask = ~np.isnan(y_true_np)
        y_true_clean = y_true_np[mask]
        y_pred_clean = y_pred_np[mask]
        y_pred_rvscale = scaler.inverse_transform(y_pred_clean)
        y_true_rvscale = scaler.inverse_transform(y_true_clean)
        y_true_rvscale [y_true_rvscale < 5] = 5  # 최소값 설정
        # MAPE 계산
        mape = mean_absolute_percentage_error(y_true_rvscale, y_pred_rvscale)
        results[cid] = mape
        logger.info(f"✅ MAPE for model {cid}: {mape:.4f}")
        del model
        torch.cuda.empty_cache()
        # 💾 캐시 저장
        with open(results_path, "wb") as f:
            pickle.dump(mape, f)
        with open(linkdict_path, "wb") as f:
            pickle.dump({k: v for k, v in link_dict.items() if any(cid in v["true"] for cid in [cid])}, f)
    return results, link_dict


def compute_aggregated_mape(link_dict):
    true_all = []
    pred_all = []
    for data in link_dict.values():
        # 1. 비어있지 않은 배열만 필터링
        true_list = [np.concatenate(arr) for arr in data['true'].values() if len(arr) > 0]
        pred_list = [np.concatenate(arr) for arr in data['pred'].values() if len(arr) > 0]
        if len(true_list) == 0 or len(pred_list) == 0:
            continue
        elif len(true_list) == 1:
            # only one model's data, flatten it
            true_array = true_list[0]
            pred_array = pred_list[0]
        else: 
            # 2. shape 맞춰서 평균 (T 시점 기준)
            true_array = np.stack([arr for arr in true_list])
            pred_array = np.stack([arr for arr in pred_list])
            # 모델 평균 (T,)
            true_array = np.nanmean(true_array, axis=0)
            pred_array = np.nanmean(pred_array, axis=0)
            
        if np.all(np.isnan(true_array)) or np.all(np.isnan(pred_array)):
            continue
        true_all.extend(true_array)
        pred_all.extend(pred_array)

    # 역정규화
    scaler = Scaler_tool()
    mask = ~np.isnan(true_all)
    y_true_clean = np.array(true_all)[mask]
    y_pred_clean = np.array(pred_all)[mask]
    flat_pred_all= scaler.inverse_transform(y_pred_clean)
    y_true_clean = scaler.inverse_transform(y_true_clean)

    mape = mean_absolute_percentage_error(y_true_clean, flat_pred_all)
    print(f"✅ Aggregated MAPE (시간 단위 기준): {mape:.4f}")
    return mape,flat_pred_all


def compute_merged_mape(pred_file, gt_file):
    pred_data = np.load(pred_file)
    gt_data = np.load(gt_file)

    y_true = []
    y_pred = []
    for key in pred_data:
        if key in gt_data:
            y_true.append(gt_data[key])
            y_pred.append(pred_data[key])

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    return mean_absolute_percentage_error(y_true, y_pred)

def plot_results(individual_results,aggregated_mape):
    # Aggregated 값을 dict에 추가
    plot_results = individual_results.copy()
    plot_results["Total"] = aggregated_mape
    # 막대 x 좌표는 숫자 리스트로 생성
    model_ids = list(individual_results.keys())
    model_vals = list(individual_results.values())

    # x 인덱스는 0 ~ len
    x_pos = np.arange(len(model_ids)+1)
    y_vals = model_vals + [aggregated_mape[0]]
    labels = [str(k) for k in model_ids] + ['Total']

    plt.figure(figsize=(14, 6))
    plt.bar(x_pos, y_vals, color='skyblue')
    plt.xticks(x_pos, labels, rotation=45)

    plt.xlabel("Cluster Index")
    plt.ylabel("MAPE")
    plt.title("MAPE per Cluster and Total MAPE")
    # aggregated 강조 표시
    plt.axhline(aggregated_mape[0], color='red', linestyle='--', label=f"Total MAPE = {aggregated_mape[0]:.4f}")
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.savefig('Results/Model_mape_barplot.png')

##predictor
def predict_individual_cluster(model_dir, cluster_list,logger,batch_size=256):
    results = {}
    link_dict = defaultdict(default_link_entry)
    for cid, link_list  in cluster_list.items():
        model_path = os.path.join(model_dir, f'{cid}_best.pth')
        data_path = os.path.join(model_dir, f'{cid}', 'predict_data.npy')
        # Load model
        torch.serialization.add_safe_globals({'net.gtnet': gtnet})
        model = torch.load(model_path, map_location='cuda',weights_only=False)
        model.eval()

        # Load data
        x_raw = np.load(data_path)
        
        # Scaler=os.path.join(model_dir, str(cid), f"scaler.npy")
        # scaler= np.load(Scaler, allow_pickle=True).item()
        # mean = scaler['mean']
        # std = scaler['std']
        
        scaler = Scaler_tool()
        
        total = x_raw.shape[0]
        if batch_size > total:
            batch_size = total

        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            x_batch = torch.tensor(x_raw[start:end], dtype=torch.float32).to('cuda')
            if x_batch.shape[2] != 36: # 데이터 로더가 이미 transpose를 했다면 건너뛰기
                 x_batch = x_batch.transpose(2, 3)            
            shape = x_batch.shape
            differ = x_batch[..., 1:] - x_batch[..., :-1]
            differ = torch.cat([torch.zeros_like(x_batch[..., :1]), differ], dim=-1)

            with torch.no_grad():
                y_pred = model(x_batch, differ, shape)[:, :, :, 0].cpu().numpy()  # (B, 12, N)

            for idx, link_id in enumerate(link_list):
                yp = y_pred[:, :, idx] # (B, 12) #1 ,12
                link_dict[link_id]["pred"][cid].append(yp)  # 2D로 만들어 누적

            # GPU 메모리 정리
            del x_batch,  y_pred,  differ
            torch.cuda.empty_cache()
    logger.info('Done: predict individual cluster results')
    return link_dict

def predict_aggregated_cluster(link_dict):
    pred_all = {}
    scaler = Scaler_tool()
    for link_id, data in link_dict.items():
        # 1. 비어있지 않은 배열만 필터링
        preds = data.get('pred', [])
        all_arrs = [arr for arr_list in preds.values() for arr in arr_list]
        # 2. shape 맞춰서 평균 (T 시점 기준)
        arr_stack = np.stack(all_arrs, axis=0)
        # 모델 평균 (T,)
        pred_array = np.nanmean(arr_stack, axis=0)
        pred_array= scaler.inverse_transform(np.array(pred_array))
        pred_all[link_id] = pred_array
    # 역정규화

    return pred_all

def Make_Test_results(model_path, cluster_list,logger, cluster_id=None):

    individual_results, link_dict = compute_individual_mape(model_path, cluster_list,logger,cluster_id=cluster_id)
    aggregated_mape = compute_aggregated_mape(link_dict)

    print("Individual MAPE results:", individual_results)
    print("Aggregated MAPE:", aggregated_mape)

    plot_results(individual_results, aggregated_mape)

    #merged_mape = compute_merged_mape('results/merged_preds.npz', 'data/ITS/ground_truth.npz')

def Predictor(model_path, cluster_list, task_config, logger, cluster_id=None):
    """
    Test the model and compute MAPE for each cluster.
    
    Args:
        model_path (str): Path to the model directory.
        clusters (dict): Dictionary containing cluster information.
        task_config (dict): Task configuration dictionary.
        logger (Logger): Logger object for logging.
        cluster_id (int, optional): Specific cluster ID to test. Defaults to None.
    """
    link_dict = predict_individual_cluster(model_path, cluster_list,logger)
    flat_pred_all = predict_aggregated_cluster(link_dict)

    return flat_pred_all
