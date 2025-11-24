import pickle
import numpy as np
import os
import scipy.sparse as sp
import torch
import pandas as pd
import psycopg2
import json
from tqdm import tqdm
import h5py

from psycopg2.extras import execute_values

from scipy.sparse import linalg
from torch.autograd import Variable
from datetime import timedelta
from dbfread import DBF
from datetime import datetime, timedelta
from psycopg2 import sql

class Scaler_tool:
    def __init__(self, mean=45.0, std=25.0):
        self.mean = mean
        self.std = std
    def transform(self, data):
        return (data - self.mean) / self.std+ 1e-8  # 작은 값 추가로 0으로 나누는 오류 방지
    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def normal_std(x):
    return x.std() * np.sqrt((len(x) - 1.)/(len(x)))

class DataLoaderS(object):
    # train and valid is the ratio of training set and validation set. test = 1 - train - valid
    def __init__(self, file_name, train, valid, device, horizon, window, normalize=2):
        self.P = window
        self.h = horizon
        fin = open(file_name)
        self.rawdat = np.loadtxt(fin, delimiter=',')
        self.dat = np.zeros(self.rawdat.shape)
        self.n, self.m = self.dat.shape
        self.normalize = 2
        self.scale = np.ones(self.m)
        self._normalized(normalize)
        self._split(int(train * self.n), int((train + valid) * self.n), self.n)

        self.scale = torch.from_numpy(self.scale).float()
        tmp = self.test[1] * self.scale.expand(self.test[1].size(0), self.m)

        self.scale = self.scale.to(device)
        self.scale = Variable(self.scale)

        self.rse = normal_std(tmp)
        self.rae = torch.mean(torch.abs(tmp - torch.mean(tmp)))

        self.device = device

    def _normalized(self, normalize):
        # normalized by the maximum value of entire matrix.

        if (normalize == 0):
            self.dat = self.rawdat

        if (normalize == 1):
            self.dat = self.rawdat / np.max(self.rawdat)

        # normlized by the maximum value of each row(sensor).
        if (normalize == 2):
            for i in range(self.m):
                self.scale[i] = np.max(np.abs(self.rawdat[:, i]))
                self.dat[:, i] = self.rawdat[:, i] / np.max(np.abs(self.rawdat[:, i]))

    def _split(self, train, valid, test):

        train_set = range(self.P + self.h - 1, train)
        valid_set = range(train, valid)
        test_set = range(valid, self.n)
        self.train = self._batchify(train_set, self.h)
        self.valid = self._batchify(valid_set, self.h)
        self.test = self._batchify(test_set, self.h)

    def _batchify(self, idx_set, horizon):
        n = len(idx_set)
        X = torch.zeros((n, self.P, self.m))
        Y = torch.zeros((n, self.m))
        for i in range(n):
            end = idx_set[i] - self.h + 1
            start = end - self.P
            X[i, :, :] = torch.from_numpy(self.dat[start:end, :])
            Y[i, :] = torch.from_numpy(self.dat[idx_set[i], :])
        return [X, Y]

    def get_batches(self, inputs, targets, batch_size, shuffle=True):
        length = len(inputs)
        if shuffle:
            index = torch.randperm(length)
        else:
            index = torch.LongTensor(range(length))
        start_idx = 0
        while (start_idx < length):
            end_idx = min(length, start_idx + batch_size)
            excerpt = index[start_idx:end_idx]
            X = inputs[excerpt]
            Y = targets[excerpt]
            X = X.to(self.device)
            Y = Y.to(self.device)
            yield Variable(X), Variable(Y)
            start_idx += batch_size

class DataLoaderM(object):
    def __init__(self, xs, ys, batch_size, pad_with_last_sample=False):
        """
        :param xs:
        :param ys:
        :param batch_size:
        :param pad_with_last_sample: pad with the last sample to make number of samples divisible to batch_size.
        """
        self.batch_size = batch_size
        self.current_ind = 0
        if pad_with_last_sample:
            num_padding = (batch_size - (len(xs) % batch_size)) % batch_size
            x_padding = np.repeat(xs[-1:], num_padding, axis=0)
            y_padding = np.repeat(ys[-1:], num_padding, axis=0)
         
            xs = np.concatenate([xs, x_padding], axis=0)
            ys = np.concatenate([ys, y_padding], axis=0)

        self.size = len(xs)
        self.num_batch = int(self.size // self.batch_size)
        self.xs = xs
        self.ys = ys

    def shuffle(self):
        permutation = np.random.permutation(self.size)
        xs, ys = self.xs[permutation], self.ys[permutation]
        self.xs = xs
        self.ys = ys

    def get_iterator(self):
        self.current_ind = 0
        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1))
                x_i = self.xs[start_ind: end_ind, ...]
                y_i = self.ys[start_ind: end_ind, ...]
                
                yield (x_i, y_i)
                self.current_ind += 1
        return _wrapper()  # ✅ 제너레이터 반환

def sym_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).astype(np.float32).todense()

def asym_adj(adj):
    """Asymmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat= sp.diags(d_inv)
    return d_mat.dot(adj).astype(np.float32).todense()

def calculate_normalized_laplacian(adj):
    """
    # L = D^-1/2 (D-A) D^-1/2 = I - D^-1/2 A D^-1/2
    # D = diag(A 1)
    :param adj:
    :return:
    """
    adj = sp.coo_matrix(adj)
    d = np.array(adj.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_laplacian = sp.eye(adj.shape[0]) - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    return normalized_laplacian

def calculate_scaled_laplacian(adj_mx, lambda_max=2, undirected=True):
    if undirected:
        adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])
    L = calculate_normalized_laplacian(adj_mx)
    if lambda_max is None:
        lambda_max, _ = linalg.eigsh(L, 1, which='LM')
        lambda_max = lambda_max[0]
    L = sp.csr_matrix(L)
    M, _ = L.shape
    I = sp.identity(M, format='csr', dtype=L.dtype)
    L = (2 / lambda_max * L) - I
    return L.astype(np.float32).todense()


def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data

def load_adj(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npz":
        data = np.load(path, allow_pickle=True)
        try:
            # .files 속성이 있으면 NpzFile
            if hasattr(data, "files"):
                # 저장된 배열이 하나뿐이라면 첫 번째 key 사용
                key = data.files[0]
                adj = data[key]
            else:
                # files 속성이 없으면 ndarray가 직접 반환된 경우
                adj = data
        finally:
            # close() 가 있으면 닫아 줍니다
            if hasattr(data, "close"):
                data.close()
    elif ext == ".npy":
        adj = np.load(path,allow_pickle=True)
    return adj


def load_dataset(dataset_dir, batch_size,Scaler,logger, valid_batch_size= None, test_batch_size=None, differ = None):
    if valid_batch_size is None:
        valid_batch_size = batch_size
    if test_batch_size is None:
        test_batch_size = batch_size
    Data = {}
    if Scaler is None or not os.path.exists(Scaler):
        mean=45
        std=25
    else:
        scaler= np.load(Scaler, allow_pickle=True).item()
        mean = scaler['mean']
        std = scaler['std']
    scaler = Scaler_tool(mean=mean, std=std) 
    if differ is None:
        datax={}
        datay={}
        for category in ['train', 'val', 'test']:
            npy_path = os.path.join(dataset_dir, f"{category}_data.npy")
            npz_path = os.path.join(dataset_dir, f"{category}_data.npz")
            h5_path = os.path.join(dataset_dir, f"{category}_data.h5")

            if os.path.exists(npy_path):
                logger.info(f"📂 Loading .npy file: {npy_path}")
                datax[category]= np.load(npy_path)['arr_0']
            elif os.path.exists(npz_path):
                logger.info(f"📂 Loading .npz file: {npz_path}")
                datax[category]= np.load(npz_path)['arr_0']
            elif os.path.exists(h5_path):
                logger.info(f"📂 Loading .h5 file: {h5_path}")
                with h5py.File(h5_path, 'r') as f:
                    datax[category] = f['x'][:]
            else:
                raise FileNotFoundError(f"❌ No data(x) file found for category '{category}' in {dataset_dir}")
            
            npy_path = os.path.join(dataset_dir, f"{category}_ydata.npy")
            npz_path = os.path.join(dataset_dir, f"{category}_ydata.npz")
            
            if os.path.exists(npy_path):
                logger.info(f"📂 Loading .npy file: {npy_path}")
                datay[category]= np.load(npy_path)['arr_0']
            elif os.path.exists(npz_path):
                logger.info(f"📂 Loading .npz file: {npz_path}")
                datay[category]= np.load(npz_path)['arr_0']
            elif os.path.exists(h5_path):
                logger.info(f"📂 Loading .h5 file: {h5_path}")
                with h5py.File(h5_path, 'r') as f:
                    datay[category] = f['y'][:]
            else:
                raise FileNotFoundError(f"❌ No data(x) file found for category '{category}' in {dataset_dir}")

        # scaler = StandardScaler(mean=data['x_train'][..., 0].mean(), std=data['x_train'][..., 0].std())
        # Data format
        # for category in ['train', 'val', 'test']:
        #     data['x_' + category][..., 0] = scaler.transform(data['x_' + category][..., 0])

        # data['train_loader'] = DataLoaderM(data['x_train'], data['y_train'], batch_size)
        # data['val_loader'] = DataLoaderM(data['x_val'], data['y_val'], valid_batch_size)
        # data['test_loader'] = DataLoaderM(data['x_test'], data['y_test'], test_batch_size)
        # data['scaler'] = scaler
        # return data
    
    else: # 불용

        differVal = np.expand_dims(differ.values, axis = -1)
        num_train = round(len(differVal) * 0.6)
        num_val = round(len(differVal) * 0.2) + num_train

        # differVal = np.nan_to_num(differVal, nan=0.0, posinf=0.0, neginf=0.0)
        x_train = differVal[:num_train]
        x_val = differVal[num_train:num_val]
        x_test = differVal[num_val:]

        # def getWeight(inputVal):
        #     output = []

        #     for input in inputVal:
        #         if(input >= 0):
        #             output.append(np.exp(-1 * input))
        #         else:
        #             output.append(-1 * np.exp(input))

        #     return output

        # x_train_differ = getWeight(x_train)
        # x_val_differ = getWeight(x_val)
        # x_test_differ = getWeight(x_test)

        gamma = 2
        x_train_differ = np.where(x_train < 0, -1 * np.exp(x_train/gamma)+1, np.exp(-1 * x_train/gamma)-1)
        x_val_differ   = np.where(x_val < 0,   -1 * np.exp(x_val/gamma)+1,   np.exp(-1 * x_val/gamma)-1)
        x_test_differ  = np.where(x_test < 0,  -1 * np.exp(x_test/gamma)+1,  np.exp(-1 * x_test/gamma)-1)

        x_differ_map = {
            'train': x_train_differ,
            'val': x_val_differ,
            'test': x_test_differ,
        }

        # print(x_train_differ.shape)
        # print(x_val_differ.shape)
        # print(x_test_differ.shape)

        for category in ['train', 'val', 'test']:
            cat_data = np.load(os.path.join(dataset_dir, category + '.npz'))
            x_seq = cat_data['x']
            x_differ = x_differ_map[category]

            # # 실제 .npz 기준으로 자른 differ 사용
            # if category == 'train':
            #     x_differ = np.exp(-1 * x_train[:len(x_seq)])  # force match
            # elif category == 'val':
            #     x_differ = np.exp(-1 * x_val[:len(x_seq)])
            # else:
            #     x_differ = np.exp(-1 * x_test[:len(x_seq)])
            
            # reshape to (N, 36, 221, 1)
            x_differ_exp = np.expand_dims(x_differ, axis=1)  # (N, 1, 221, 1)
            x_differ_tiled = np.tile(x_differ_exp, (1, x_seq.shape[1], 1, 1))  # (N, 36, 221, 1)

            # merge
            # print(x_seq)
            # print(x_differ_tiled)
            # print(cat_data['y'])
            x_merged = np.concatenate([x_seq, x_differ_tiled], axis=-1)  # (N, 36, 221, 2)
            data['x_' + category] = x_merged
            data['y_' + category] = cat_data['y']




        # Data format  :: already processed
    # for category in ['train', 'val', 'test']:
    #     data['x_' + category][..., 0] = scaler.transform(data['x_' + category][..., 0])
    logger.info(
        "📊 Dataset shapes:\n"
        "   • train: x=%s, y=%s\n"
        "   • val:   x=%s, y=%s\n"
        "   • test:  x=%s, y=%s",
        datax['train'].shape, datay['train'].shape,
        datax['val'].shape,   datay['val'].shape,
        datax['test'].shape,  datay['test'].shape
    )
    Data['train_loader'] = DataLoaderM(xs=datax['train'], ys=datay['train'], batch_size=batch_size)
    Data['val_loader'] = DataLoaderM(xs=datax['val'], ys=datay['val'], batch_size=valid_batch_size)
    Data['test_loader'] = DataLoaderM(xs=datax['test'], ys=datay['test'], batch_size=test_batch_size)
    Data['scaler'] = scaler

    return Data


def masked_mse(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = (preds-labels)**2
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

def masked_rmse(preds, labels, null_val=np.nan):
    return torch.sqrt(masked_mse(preds=preds, labels=labels, null_val=null_val))


def masked_mae(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /=  torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

def masked_mape(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /=  torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)/labels
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def metric(pred, real):
    mae = masked_mae(pred,real,0.0).item()
    mape = masked_mape(pred,real,0.0).item()
    rmse = masked_rmse(pred,real,0.0).item()
    return mae,mape,rmse


def load_node_feature(path):
    fi = open(path)
    x = []
    for li in fi:
        li = li.strip()
        li = li.split(",")
        e = [float(t) for t in li[1:]]
        x.append(e)
    x = np.array(x)
    mean = np.mean(x,axis=0)
    std = np.std(x,axis=0)
    z = torch.tensor((x-mean)/std,dtype=torch.float)
    return z


def normal_std(x):
    return x.std() * np.sqrt((len(x) - 1.) / (len(x)))

def get_differ(seq_out_len, device, engine, dataloader, realIdx, loaderIdx):
    outputs = []
    scaler = dataloader['scaler']
    

#ymjun 신규 추가: differ를 구하는 부분 version 1: 단일 클러스터에서만 고려한 소스 코드드
def get_differ(seq_out_len, device, engine, dataloader, realIdx, loaderIdx, num_node):
    outputs = []
    scaler = dataloader['scaler']

    realy = torch.Tensor(dataloader[realIdx]).to(device)
    realy = realy.transpose(1, 3)[:, 0, :, :]  # [samples, nodes, seq_out_len]

    for _, (x, y) in enumerate(dataloader[loaderIdx].get_iterator()):
        testx = torch.Tensor(x).to(device)
        testx = testx.transpose(1, 3)
        with torch.no_grad():
            preds = engine.model(testx, None, num_node)
            preds = preds.transpose(1, 3)
        outputs.append(preds.squeeze())

    yhat = torch.cat(outputs, dim=0)[:realy.size(0), ...]

    if yhat.dim() == 4 and yhat.shape[1] > 12:
        yhat = yhat[:, -1, :, :]  # shape: [samples, nodes, seq_out_len]

    diffs = []
    for i in range(seq_out_len):
        pred_i = scaler.inverse_transform(yhat[:, :, i].detach().cpu().numpy())  # [samples, nodes]
        real_i = realy[:, :, i].detach().cpu().numpy()

        # # 디버깅 출력
        # if i == 0:
        #     print("Pred NaN:", np.isnan(pred_i).sum(), "Real NaN:", np.isnan(real_i).sum())

        # NaN 마스크 (실제값 기준)
        mask = ~np.isnan(real_i)
        diff = np.full_like(real_i, np.nan)
        diff[mask] = pred_i[mask] - real_i[mask]

        diffs.append(torch.from_numpy(diff).unsqueeze(-1))  # [samples, nodes, 1]

    diff_tensor = torch.cat(diffs, dim=-1)  # [samples, nodes, seq_out_len]

    return torch.nanmean(diff_tensor, dim=2)  # [samples, nodes]


def Make_Postgres_form(timestamp,Predict_result,conn,path_config,task_config,logger):
    interval = timedelta(minutes=task_config['interval'])
    ITS_Link_path = path_config['data']['ITS_info']
    df = DBF(ITS_Link_path, encoding='cp949')
    Linkdata = pd.DataFrame(iter(df))
    selectedLink = Linkdata[['LINK_ID', 'LENGTH']]
    data_to_insert =[]
    schema   = task_config['Postgres_db_info'].get("schema", "predict")  # 기본값 
    table    = task_config['Postgres_db_info']["table"]
    BATCH_SIZE = 288  # 24h * 60m / 5m
    for link_id, preds in tqdm(Predict_result.items(), desc="🔄 Processing Links"):
        T,_ = preds.shape  # (T, 12)
        preds = np.nan_to_num(preds, nan=1.0, posinf=1.0, neginf=1.0)  # NaN, inf 처리
        preds = np.clip(preds, a_min=1, a_max=None).astype('int')     # 1 미만 방지
        f=0
        LENGTH = selectedLink[selectedLink['LINK_ID']==link_id]['LENGTH'].values[0]
        n_batches = int(np.ceil(T / BATCH_SIZE))
        for b in range(n_batches):
            start = b * BATCH_SIZE
            end   = min(start + BATCH_SIZE, T)
            batch_preds      = preds[start:end, :]
            batch_timestamps = timestamp[start:end]
            mini_step=end-start

            # 1) 이 배치(하루)의 평균 속도
            avg_speed = float(batch_preds[:, 1].mean())
            target_date = pd.Timestamp(timestamp[start]).date()
            data = []
            for t_idx in range(mini_step):
                t_idxx= t_idx + start
                timestep = pd.Timestamp(timestamp[t_idxx])
                speed= preds[t_idxx, :].astype('int')
                data.append({
                    "time": timestep.strftime('%H:%M'),
                    "speed": speed.tolist() ,
                    "travel_sec": (LENGTH/speed*3.6).astype('float16').tolist() # float으로 변환 (np.float32 문제 방지)
                    # 3.6 = 3600/1000, km/h to m/s 변환
                })

            # INSERT할 row 구성
            data_to_insert.append((
                int(link_id),
                avg_speed,
                str(target_date),
                json.dumps(data, ensure_ascii=False)
            ))
            # DB insert
    cursor = conn.cursor()

    q = sql.SQL(
        "INSERT INTO {}.{} (link_id, avg_speed, target_date, data) VALUES %s"
    ).format(sql.Identifier(schema), sql.Identifier(table))

    execute_values(cursor, q, data_to_insert)
    conn.commit()
    cursor.close()
    data_to_insert =[]

    """
        for f in range(F): # 5분 이후 예측값으로 확장
            pred_time = timestep + f * interval
            records.append({
                "link_id": link_id,
                "timestamp": pred_time,
                "prediction": float(preds[t_idx, f])  # float으로 변환 (np.float32 문제 방지)
            })
    """

def Make_Postgres_form_far(
    Predict_result, link_ids, dates, conn, path_config, task_config, logger
):
    schema   = task_config['Postgres_db_info'].get("schema", "predict_dev")  # 기본값 
    table    = task_config['Postgres_db_info']["table"]
    T, D, L = Predict_result.shape
    interval = task_config['interval']
    df_info = pd.DataFrame(
        iter(DBF(path_config['data']['ITS_info'], encoding='cp949'))
    ).set_index('LINK_ID')
    lengths = df_info.loc[link_ids, 'LENGTH'].astype(float).values

    sql = sql.SQL(
        "INSERT INTO {}.{} (link_id, avg_speed, target_date, data) VALUES %s"
    ).format(sql.Identifier(schema), sql.Identifier(table))
    batch_size = 1000
    cursor = conn.cursor()

    for d_i, date_str in enumerate(dates):
        base = datetime.strptime(date_str, "%Y%m%d")
        target_date = base.date()

        # (T,L) → NaN/inf 처리 + 링크별 평균 미리 계산
        day_preds = np.nan_to_num(
            Predict_result[:, d_i, :], nan=1, posinf=1, neginf=1
        ).astype('int')
        avg_speeds =np.round(day_preds.mean(axis=0),2)  # float으로 변환 (np.float32 문제 방지)

        # 시간 문자열 한번 생성
        times = [
            (base + timedelta(minutes=interval * t)).strftime("%H:%M")
            for t in range(T)
        ]

        data_to_insert = []
        for start in tqdm(range(0, L, batch_size),desc=target_date.strftime("%Y-%m-%d")+": 🔄 Link 처리"):
            end = min(start + batch_size, L)
            for i in range(start, end):
                lid = link_ids[i]
                length = lengths[i]
                preds_link = day_preds[:, i]

                recs = [
                    {
                        "time": times[t_j],
                        "speed": int(preds_link[t_j]),
                        "travel_sec": (round(length/preds_link[t_j]*3.6,ndigits=2)).astype('float16').tolist() # float으로 변환 (np.float32 문제 방지)
                    }
                    for t_j in range(T)
                ]
                data_to_insert.append((
                    int(lid),
                    float(avg_speeds[i]),
                    str(target_date),
                    json.dumps(recs, ensure_ascii=False)
                ))

            # batch_size 링크마다 INSERT & clear
            execute_values(cursor, sql, data_to_insert)
            conn.commit()
            data_to_insert.clear()

        logger.info(f"[{date_str}] 완료: {L}개 링크 삽입")

    cursor.close()
    logger.info("✅ 모든 날짜 INSERT 완료")