import networkx as nx
from node2vec import Node2Vec
import numpy as np
import pandas as pd
import os
from itertools import combinations
import networkx as nx
from Utils.share import Scaler_tool
from gensim.models import Word2Vec
import logging

def generate_node2vec_embeddings(
    output_dir,
    Total_list,
    logger,
    link_ids: list[str],
    embed_dim: int = 32,
    walk_length: int = 40,
    num_walks: int = 20,
    p: float = 1.0,
    q: float = 1.0
) -> pd.DataFrame:
    """
    edge_list: [(linkA, linkB), ...]  # 도로망의 간선을 표현
    link_ids: 전체 link_id 리스트 (노드 순서)
    """
    save_path = os.path.join(output_dir, 'node2vec_embeddings.npy')
    if os.path.exists(save_path):
        logger.info(f"[link embedding이 이미 진행되었습니다, 스킵: {save_path}")
        return
    # 0. edge_list 처리
    Scaler=Scaler_tool()
    df_links = Total_list[Total_list['LINK_ID'].isin(link_ids)]
    df_links['MAX_SPD'] = Scaler.transform(df_links['MAX_SPD'].values)
    # 2) 교차로(node)별 incident link 모으기
    node2links: dict[str, list[str]] = {}
    for _, row in df_links.iterrows():
        lid = row['LINK_ID']
        for junc in (row['F_NODE'], row['T_NODE']):
            node2links.setdefault(junc, []).append(lid)

    # 3) line-graph 간선 생성: 동일 교차로에 연결된 링크들끼리 fully-connected
    link_edges = set()
    for links in node2links.values():
        if len(links) < 2:
            continue
        for u, v in combinations(links, 2):
            link_edges.add((str(u), str(v)))
    # 4) 그래프 초기화 및 노드/엣지 추가
    G = nx.Graph()
    # 4-1) 모든 관심 링크를 노드로 미리 추가 (isolated link 대비)
    G.add_nodes_from([str(lid) for lid in link_ids])
    # 4-2) line-graph 엣지 추가
    G.add_edges_from(link_edges)

    # 5) Node2Vec 학습
    n2v = Node2Vec(
        G,
        dimensions=embed_dim,
        walk_length=walk_length,
        num_walks=num_walks,
        p=p,
        q=q,
        workers=3,
        seed=42
    )

    walks = n2v.walks  # List[List[str]], 즉 모든 랜덤 워크
    #logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)
    # 2) gensim Word2Vec 로 학습 (진행 로그 보기)
    w2v = Word2Vec(
        sentences=walks,
        vector_size=embed_dim,
        window=5,
        min_count=1,
        sg=1,                  # skip-gram
        workers=8,
        epochs=4,              # epoch 수를 조절하세요
        compute_loss=True
    )
    model = w2v
    # 4) link_ids 순서대로 임베딩 매트릭스 생성
    embeddings = np.vstack([
        model.wv[str(lid)] for lid in link_ids
    ])  # shape = (L, embed_dim)

    # 5) DataFrame 으로 반환 (index=link_id, cols=emb_0...emb_n)
    cols = [f"emb_{i}" for i in range(embed_dim)]
    df_emb = pd.DataFrame(embeddings, index=link_ids, columns=cols)

    # 6) LANES, MAX_SPD 병합
    #    df_links 에서 필요한 static 컬럼만 뽑아서 emb에 join
    df_static = df_links.set_index('LINK_ID')[['LANES','MAX_SPD']]
    df_emb = df_emb.join(df_static, how='left')
    np.save(os.path.join(output_dir, 'node2vec_embeddings.npy'), df_emb.values.astype(np.float32))
    return