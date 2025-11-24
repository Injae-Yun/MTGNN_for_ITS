import pandas as pd
import numpy as np
import pickle
import scipy.sparse
from numba import cuda
import os
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import lil_matrix
from joblib import Parallel, delayed


def initMatrix(arrSize):

    """
    arrSize * arrSize 크기의 행렬 생성
    모든 값을 np.inf로 초기화
    """
    listArr = np.full((arrSize, arrSize), np.inf, dtype=np.float32)
    
    #자기자신으로 가는 거리는 0으로 설정하였음
    np.fill_diagonal(listArr, 0)

    # listArr = scipy.sparse.lil_matrix((arrSize, arrSize), dtype=np.float32)
    # listArr.setdiag(0)

    return scipy.sparse.csr_matrix(listArr)


@cuda.jit
def computeLinkDistance(tNode, fNode, length, arrMap):
    """
    CUDA GPU 커널 함수: 링크 간 거리 계산
    """

    xIdx = cuda.grid(1)  # 병렬 실행되는 인덱스
    if xIdx < tNode.shape[0]:
        for yIdx in range(fNode.shape[0]):
            if tNode[xIdx] == fNode[yIdx]:
                arrMap[xIdx, yIdx] = (length[xIdx] + length[yIdx]) / 2

def getOneLinkDistancefromGPU(arrMap, nodeLinkData):
    returnVal = np.array(arrMap.toarray().copy(), dtype=np.float32)

     # GPU 메모리로 데이터 전송
    T_NODE = nodeLinkData['T_NODE'].astype(np.int32).values
    F_NODE = nodeLinkData['F_NODE'].astype(np.int32).values
    LENGTH = nodeLinkData['LENGTH'].astype(np.float32).values
    arrMapDevice = cuda.to_device(returnVal)

    # GPU 병렬 실행
    threadsperblock = 32
    blockspergrid = (len(nodeLinkData) + (threadsperblock - 1)) // threadsperblock
    computeLinkDistance[blockspergrid, threadsperblock](T_NODE, F_NODE, LENGTH, arrMapDevice)

    # CPU로 결과 반환
    return scipy.sparse.csr_matrix(arrMapDevice.copy_to_host())

@cuda.jit
def floyd_kernel(arrMap, k, arrSize):
    i, j = cuda.grid(2)
    if i < arrSize and j < arrSize:
        via = arrMap[i, k] + arrMap[k, j]
        if via < arrMap[i, j]:
            arrMap[i, j] = via


def getShortDistancefromGPU(arrMap, arrSize):
    dist = np.array(arrMap.toarray(), dtype=np.float32)
    d_dist = cuda.to_device(dist)

    threads_per_block = (16, 16)
    blocks_per_grid_x = (arrSize + threads_per_block[0] - 1) // threads_per_block[0]
    blocks_per_grid_y = (arrSize + threads_per_block[1] - 1) // threads_per_block[1]

    for k in range(arrSize):
        floyd_kernel[(blocks_per_grid_x, blocks_per_grid_y), threads_per_block](d_dist, k, arrSize)

    return d_dist.copy_to_host()


def mapVelocityProcess(Map, linkData, threshold=0.01):
    #linkID = linkData['LINK_ID'].values  # Pandas Series -> Numpy 배열 변환
    velocities = linkData['MAX_SPD'].values  # 속도를 numpy 배열로 변환

   #Map = Map.toarray().astype(np.float64)  # sparse → dense 변환
    Map = np.array(Map, dtype=np.float64)  # NumPy 배열 변환
    # 벡터 연산을 활용하여 최적화
    gamma_matrix = (velocities[:, None] + velocities[None, :]) / 2 * (1000 / 60) * 5
    # 속도 평균 / 단위 환산 (km->m, h->m) / 5분 
    gamma_matrix /= 1.5 # 보정값(2)
    # 거리 기반 가우시안 변환 수행
    returnVal = np.exp(-pow((Map / gamma_matrix), 2.0)).astype(np.float16)
    returnVal[returnVal < threshold] = 0

    return returnVal

def mapSortedProcess(linkData, mapData):
    linkID = linkData['LINK_ID'].values
    
    sorted_points = []
    similarities = cosine_similarity(mapData)
    sorted_indices = np.argsort(np.sum(similarities, axis=1))  # 유사도 합으로 정렬
    sorted_points.extend(sorted_indices)
    sorted_linkID = linkID[sorted_indices]

    sorted_Map = mapData[np.ix_(sorted_points, sorted_points)]

    return sorted_linkID, sorted_Map
