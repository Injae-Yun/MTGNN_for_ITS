# -*- coding: utf-8 -*- 

import torch
import numpy as np
import yaml
import time
from Utils.share import *
from Core.trainer import Trainer
from Core import net
from Core import layer
from Core.net import gtnet

import pandas as pd
import logging
import os

def load_config(config_path=os.path.join("Config","MTGNN_config.yaml")):
    with open(config_path, "r", encoding='utf-8') as file:
        config = yaml.safe_load(file)
    return config

def get_device(config,logger):
    # GPUSet: 'RTX' 면 단일 GPU, 그 외면 멀티 GPU 모드로 가정
    gpu_set = config['environment']['GPUSet']
    logger.info(f"[get_device] GPUSet from config = {gpu_set}")
    # CUDA 사용 가능 여부 확인
    if torch.cuda.is_available():
        if gpu_set == 'RTX':
            dev, ddp = torch.device("cuda:0"), False
        else:
            dev, ddp = torch.device("cuda"), True
            
        logger.info(f"[get_device] → using GPU, device={dev}, ddpSet={ddp}")
        return dev, ddp, "cuda"
    else:
        # CPU fallback
        logger.warning("[get_device] → CUDA NOT AVAILABLE, fallback to CPU")
        return torch.device('cpu'), False, "cpu"
    
def build_model(config, 
                ddpSet, 
                output_dir_model,target, 
                mask_valid, logger,
                device=torch.device('cuda:0'), 
                dev_name="cuda",
                idx=[]):

    global_seq_in_len = config['data']['in_len']
    global_seq_differ_len = config['data']['differ_len']
    global_batch_size = config['data']['batch_size']

    dataPath = os.path.join(output_dir_model ,str(idx))
    mapPath = os.path.join('Data',target,"ClusterMap","Cmap_" +str(idx)+ ".npz")
    
    seq_out_len = config['data']['out_len']

    gcn_true = config['param']['gcn_true']
    buildA_true = config['param']['buildA_true']
    load_static_feature = config['param']['load_static_feature']
    cl = config['param']['cl']
    gcn_depth = config['param']['gcn_depth']
    subgraph_size = config['param']['subgraph_size']
    node_dim = config['param']['node_dim']
    dropout = config['param']['dropout']
    dilation_exponential = config['param']['dilation_exponential']
    conv_channels = config['param']['conv_channels']
    residual_channels = config['param']['residual_channels']
    skip_channels = config['param']['skip_channels']
    end_channels = config['param']['end_channels']
    in_dim = config['param']['in_dim']
    layers = config['param']['layers']
    learning_rate = config['param']['learning_rate']
    weight_decay = config['param']['weight_decay']
    clip = config['param']['clip']
    step_size1 = config['param']['step_size1']
    propalpha = config['param']['propalpha']
    tanhalpha = config['param']['tanhalpha']
    differsize = config['param']['differSize']
    seed = config['param']['seed']

    weight = None
    extraWeight = None
    batch_size = global_batch_size

    Scaler=os.path.join(output_dir_model,str(idx),'scaler.npy')

    linkCntPath = os.path.join(dataPath, str(idx)+".txt")

    

    seq_in_len = global_seq_in_len
    saveModelPrep = "None_ITS_"
    dataloader = load_dataset(dataPath, batch_size, Scaler, logger, None)
    differFlag = False

    predefined_A = load_adj(mapPath)
    predefined_A = torch.tensor(predefined_A)-torch.eye(len(predefined_A))
    if mask_valid is None:
        A_sub = predefined_A.to(device)
    else:
        # 남길 인덱스 리스트: mask_valid==True 인 위치
        keep_idxs = np.nonzero(mask_valid)[0]    # shape (N_valid,)
        # np.ix_로 행과 열 모두 슬라이싱
        A_sub = predefined_A[np.ix_(keep_idxs, keep_idxs)]  # shape (N_valid, N_valid)       


    earlyDropSwitch = 0

    scaler = dataloader['scaler']

    model = gtnet(gcn_true, buildA_true, gcn_depth, len(A_sub),
                  device, predefined_A=A_sub,
                  dropout=dropout, subgraph_size= subgraph_size,
                  node_dim=node_dim,
                  dilation_exponential=dilation_exponential,
                  conv_channels=conv_channels, residual_channels=residual_channels,
                  skip_channels=skip_channels, end_channels= end_channels,
                  seq_length=seq_in_len, in_dim=in_dim, out_dim=seq_out_len,
                  layers=layers, propalpha=propalpha, tanhalpha=tanhalpha, layer_norm_affline=True)
    model = model.to(device)          
    first_param = next(model.parameters())
    logger.info(f"[GPU 체크] model parameter device = {first_param.device}")
    engine = Trainer(ddpSet, model, learning_rate, weight_decay, clip, step_size1, seq_out_len, scaler, device, dev_name, cl)

    engine.differFlag = differFlag
    return engine,dataloader

def main_no_differ(cluster_links,
            Config_path,output_dir_model, target, logger,cid='Total',mode='train'):
    model_path = os.path.join(output_dir_model, f"{cid}_best.pth")

    # ✅ 모델이 이미 존재하고 retrain 모드가 아니면 skip
    if os.path.exists(model_path) and mode != 'retrain':
        logger.info(f"⏭️ 클러스터 {cid}는 이미 학습 완료됨. Skipping...")
        return
    valid_link_root = os.path.join(output_dir_model,str(cid),"valid_link_id.npy")
    logger.info(f"{valid_link_root}")
    valid_num_links = None
    if os.path.exists(valid_link_root):
        # 파일이 있으면 그대로 불러와서 cluster_links에 저장
        valid_cluster_links = np.load(valid_link_root, allow_pickle=True)
        valid_num_links = len(valid_cluster_links)
        logger.info(f"✅ 기존 valid_link_id 로딩: {valid_link_root} (개수={valid_num_links})")

    config = load_config(Config_path)
    device, ddpSet,dev_name = get_device(config,logger)
    num_link = len(cluster_links)
    if valid_num_links is None:
        valid_num_links = num_link
        mask_valid = None
    else:
        if not num_link == valid_num_links:
            #    cluster_links 배열에서 valid에 없는 ID의 위치(index)들을 찾음
            mask_valid = np.isin(cluster_links, valid_cluster_links)
            # mask_valid == False 인 위치가 제외 대상
            exclude_idxs = np.nonzero(~mask_valid)[0]  # e.g. array([3, 17, 42])
            valid_num_links= num_link
        else: 
            mask_valid = None
    logger.info(f"▶️ 클러스터 {cid} 처리 중... 링크 수: {valid_num_links}")

    engine,dataloader = build_model(config, ddpSet,
            output_dir_model, target, mask_valid, logger, device=device, dev_name=dev_name,idx=cid)
    epochs = config['param']['epochs']
    step_size2 = config['param']['step_size2']
    early_drop = config['param']['early_drop']
    num_split = config['param']['num_split']
    print_every = config['print']['print_every']
    
    ## 학습하는 부분
    his_loss =[]
    bestScore = -1
    minl = np.inf #ymjun 기본값이 1e5인데 부적합하다고 판단

    for i in range(1,epochs+1):
        train_loss = []
        train_mape = []
        train_rmse = []
        t1 = time.time()
        dataloader['train_loader'].shuffle()

        for iter, (x, y) in enumerate(dataloader['train_loader'].get_iterator()):
            trainx = torch.Tensor(x).to(device) # (B, 1, N, 36)
            trainy = torch.Tensor(y).to(device) # (B, 1, N, 12)
            if trainx.shape[2]==36:
                trainx= trainx.transpose(2, 3)
                trainy = trainy.transpose(2, 3)
            
            #logger.info(f"x,y shape : {x.shape},{y.shape}")

            if iter%step_size2==0:
                perm = np.random.permutation(range(valid_num_links))
            num_sub = int(valid_num_links/num_split)
            for j in range(num_split):
                if j != num_split-1:
                    id = perm[j * num_sub:(j + 1) * num_sub]
                else:
                    id = perm[j * num_sub:]
                id = torch.tensor(id).to(device)
                tx = trainx[:, :, id, :]
                ty = trainy[:, :, id, :]

                # print("tx에 NaN 존재:", torch.isnan(tx).any().item())
                # print("ty에 NaN 존재:", torch.isnan(ty).any().item())

                metrics = engine.train(tx, ty[:,0,:,:], None, id)
                train_loss.append(metrics[0])
                train_mape.append(metrics[1])
                train_rmse.append(metrics[2])
            # if iter % print_every == 0 :
            #     log = 'Iter: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}'
            #     logger.info(log.format(iter, train_loss[-1], train_mape[-1], train_rmse[-1]))

        t2 = time.time()
        #validation
        valid_loss = []
        valid_mape = []
        valid_rmse = []

        s1 = time.time()

        for iter, (x, y) in enumerate(dataloader['val_loader'].get_iterator()):
            testx = torch.Tensor(x).to(device)
            testy = torch.Tensor(y).to(device)
            if testx.shape[2]==36:
                testx= testx.transpose(2, 3)
                testy = testy.transpose(2, 3)
   
            metrics = engine.eval(testx, testy[:,0,:,:], None, num_link)
            valid_loss.append(metrics[0])
            valid_mape.append(metrics[1])
            valid_rmse.append(metrics[2])

        s2 = time.time()
        mtrain_loss = np.mean(train_loss)
        mtrain_mape = np.mean(train_mape)
        mtrain_rmse = np.mean(train_rmse)

        mvalid_loss = np.mean(valid_loss)
        mvalid_mape = np.mean(valid_mape)
        mvalid_rmse = np.mean(valid_rmse)
        his_loss.append(mvalid_loss)

        log = 'Epoch: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f} \n Valid Loss: {:.4f}, Valid MAPE: {:.4f}, Valid RMSE: {:.4f}, Training Time: {:.2f}+{:.2f}/epoch'
        logger.info(log.format(i, mtrain_loss, mtrain_mape, mtrain_rmse, mvalid_loss, mvalid_mape, mvalid_rmse, (t2 - t1),(s2-s1)))

        if mvalid_loss<minl:
            minl = mvalid_loss
            # bestScore = i
            # torch.save(engine.model.state_dict(), args.save + saveModelPrep + str(i) +".pth")
            # if(differ is None):

            if(ddpSet == True):
                torch.save(engine.model.module, model_path)
                torch.save()
            else:
                torch.save(engine.model, model_path)
            earlyDropSwitch = 0

        else:
            earlyDropSwitch += 1
            if(earlyDropSwitch == early_drop):
                break

    bestid = np.argmin(his_loss)
    logger.info(f"✅ 클러스터 {cid} 모델 저장 완료: {model_path}")
    logger.info(f"The valid loss on best model is {round(his_loss[bestid], 4)}")

    #return engine, dataloader, device, num_link

def training_MTGNN(Config_path,output_dir_model,sorted_clusters,target, logger,cluster_id=None):
    for key, link_list in sorted_clusters.items():
        if cluster_id is not None and key not in cluster_id: # 특정 클러스터 ID only 처리
            continue
        main_no_differ(link_list,
        Config_path,output_dir_model, target,logger,key,mode='train')
