import numpy as np
import os 
import pickle 
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import pandas as pd
from dbfread import DBF
from Utils import make_map as map
from Core.Train_MTGNN import training_MTGNN

def createMap(arrSize, linkData,logger):
    # 맵을 만드는 부분

    arrMap = map.initMatrix(arrSize)
    logger.info(f"init Matrix Finished")

    # # # # startTime = time.time()
    arrMap = map.getOneLinkDistancefromGPU(arrMap, linkData)
    # print(arrMap.shape)
    logger.info(f"getOneLinkDistance")

    # # # # startTime = time.time()
    arrMap = map.getShortDistancefromGPU(arrMap, arrSize)
    # print(arrMap.shape)

    return arrMap

def drawHeatMap(data,target, key, sorted = False, name = None):

    fileName = os.path.join('Data','Results',target,name)
    title = f"{target}_Heatmap of cluster_{key}"

    sorted_points = []
    if(sorted):
        # 상관 행렬에서 가장 유사한 포인트들을 기준으로 정렬 (여기서는 코사인 유사도)
        similarities = cosine_similarity(data)
        sorted_indices = np.argsort(np.sum(similarities, axis=1))  # 유사도 합으로 정렬
        sorted_points.extend(sorted_indices)

        # 4. 결과를 heatmap 형태로 시각화 (정렬된 데이터로 heatmap 생성)
        data = data[np.ix_(sorted_points, sorted_points)]

    plt.figure(figsize=(10, 8))
    plt.imshow(data, cmap='hot', interpolation='nearest')
    plt.title(title, fontsize=20)
    plt.colorbar()
    plt.savefig(fileName)
    
def saveMapdata(arrMap, fileName):
    
    #데이터를 저장하는 부분
    with open(fileName, "wb") as f:
        pickle.dump(arrMap, f)
    
def SortClusterMap(totalmap, link_ids, linkList, target, idx,logger):

    link_id_to_idx = {lid: i for i, lid in enumerate(link_ids)}
    availableIdx = [link_id_to_idx[link] for link in linkList if link in link_id_to_idx]

    if not availableIdx:
        logger.warning(f"⚠️ No matching links found for cluster {target}_{idx}")
        return None, []
    
    subMap = totalmap[np.ix_(availableIdx, availableIdx)]

    drawHeatMap(subMap,target,  sorted=True, name =f"Sorted_Map_Cluseter_{idx}")
    saveMapdata(subMap, f"ClusterMap_{idx}")
    logger.info(f"✅ SubMap 생성 완료: {target}_{idx}, size={subMap.shape}")



def generateTotalMap(ITS_Link_path,ITS_Node_path,target,logger):

    savemap_path= os.path.join('Data',target,'TotalMap')

    # 이미 생성된 경우 스킵
    if os.path.exists(savemap_path+'.pkl'):
        print(f"✅ 이미 생성된 SortedTotalMap 존재: {savemap_path}")
        with open(savemap_path+'pkl', "rb") as f:
            preprocessMap = pickle.load(f)
        return preprocessMap
    
    NodeData = pd.read_csv(ITS_Node_path, encoding="utf-8", low_memory=False)
    selectedNode = NodeData[['NODE_ID', 'X좌표', 'Y좌표']].copy()
    selectedNode['NODE_ID'] = selectedNode['NODE_ID'].astype(str)
    df = DBF(ITS_Link_path, encoding='cp949')  
    Linkdata = pd.DataFrame(iter(df)) 
    selectedLink = Linkdata[['F_NODE', 'T_NODE','MAX_SPD','LENGTH']]
    Table = selectedLink[
        selectedLink['F_NODE'].isin(selectedNode['NODE_ID']) |
        selectedLink['T_NODE'].isin(selectedNode['NODE_ID'])
    ].reset_index(drop=True)
    arrSize= len(Table)
    
    mapData = createMap(arrSize, Table,logger) # 
    preprocessMap = map.mapVelocityProcess(mapData, selectedLink)
    saveMapdata(preprocessMap, savemap_path)    #use this one

    sortedIdx, sortedMap = map.mapSortedProcess(selectedLink, preprocessMap) #sorting
    saveLocation = os.path.join("data", "Total_ITS.txt")
    
    with open(saveLocation, "w", encoding="utf-8") as f:
        for idx in sortedIdx:
            f.write(f"{idx}\n")
    savemap_path= os.path.join('Data',target,'SortedTotalMap')
    drawHeatMap(sortedMap, target, sorted=True, name = "SortedTotalMap") # only for figure

    return preprocessMap
def generateClusterMap(df, NodeData, linkList, target, cluster_idx, logger):
    # 기존 total map 생성 코드와 유사하지만 cluster link만 다룸
    # 캐시 확인
    cluster_map_path = os.path.join("Data", target,'ClusterMap', f"Cmap_{cluster_idx}.npz")
    if os.path.exists(cluster_map_path):
        logger.info(f"✅ Cluster {cluster_idx} already exists. Skipping...")
        with open(cluster_map_path, "rb") as f:
            subMap = pickle.load(f)
        return subMap

    # data preparation
    selectedNode = NodeData[['NODE_ID', 'X좌표', 'Y좌표']].copy()
    selectedNode['NODE_ID'] = selectedNode['NODE_ID'].astype(str)

    Linkdata = pd.DataFrame(iter(df))
    selectedLink = Linkdata[['LINK_ID','F_NODE', 'T_NODE', 'MAX_SPD', 'LENGTH']]

    Table = selectedLink[
        selectedLink['LINK_ID'].isin(linkList) |
        selectedLink['LINK_ID'].isin(linkList)
    ].reset_index(drop=True)
    arrSize= len(Table)

    mapData = createMap(arrSize, Table, logger)
    processedMap = map.mapVelocityProcess(mapData, Table)

    # 시각화 및 저장
    drawHeatMap(processedMap, target,cluster_idx, sorted=True, name=f"Sorted_Map_Cluster_{cluster_idx}")
    saveMapdata(processedMap, cluster_map_path)

    logger.info(f"✅ ClusterMap 생성 완료: {target}_{cluster_idx}, size={processedMap.shape}")
    return processedMap

def genmap(link_ids,sorted_clusters,target,path_config,logger,cluster_id=None):
    # if not exist or version different, make total map
    ITS_Node_path = path_config['data']['Node_info']
    ITS_Link_path = path_config['data']['ITS_info']
    #totalmap=generateTotalMap(ITS_Link_path,ITS_Node_path,target,logger)
    # 용량 이슈로 실패
    os.makedirs(os.path.join('Data','Results',target), exist_ok=True)
    os.makedirs(os.path.join("Data", target,'ClusterMap'), exist_ok=True)
    NodeData = pd.read_csv(ITS_Node_path, encoding="utf-8", low_memory=False)
    df = DBF(ITS_Link_path, encoding='cp949')
    for key, link_list in sorted_clusters.items():
        if cluster_id is not None and key not in cluster_id: # 특정 클러스터 ID only 처리
            continue
        #SortClusterMap(totalmap, link_ids, link_list, target, key, logger)
        logger.info(f"▶️ Cluster {key} 처리 중...")
        generateClusterMap(df, NodeData, link_list, target, key, logger)



def training_Model(Config_path, output_dir_model, sorted_clusters, target, logger):

    training_MTGNN(Config_path, output_dir_model, sorted_clusters,target, logger )