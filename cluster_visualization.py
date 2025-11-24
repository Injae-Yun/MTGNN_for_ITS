import os
import geopandas as gpd
import matplotlib.pyplot as plt
import yaml
import numpy as np
import contextily as ctx

# --- 환경 설정 및 데이터 로드 ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULT_DIR = os.path.join(BASE_DIR, 'Results')
os.makedirs(RESULT_DIR, exist_ok=True)

# datapath.yaml에서 공통 데이터 경로 로드
yaml_path = os.path.join(BASE_DIR, '..', 'datapath.yaml')
with open(yaml_path, 'r', encoding='utf-8') as f:
    dp = yaml.safe_load(f)
common = dp['common_data']
link_shp_path = common['moct_link_shp']

# 링크 shape 파일 로드
gdf = gpd.read_file(link_shp_path)

if 'cluster' not in gdf.columns:
    # 예시: cluster 정보가 별도 npy/pkl 파일에 있다면 아래처럼 불러와서 merge
    cluster_path = os.path.join(BASE_DIR, 'Data','Livinglab', 'connected_clusters.pkl')
    arr = np.load(cluster_path, allow_pickle=True)
    # arr: {cluster_id: [link_id, ...], ...}
    cluster_map = {}
    for cluster_id, link_list in arr.items():
        for link_id in link_list:
            link_id_str = str(link_id)
            if link_id_str not in cluster_map:
                cluster_map[link_id_str] = []
            cluster_map[link_id_str].append(int(cluster_id))
# 전체 영역(화성시) 지도 그리기 (회색)

# livinglab 전체 링크만 추출 (background)
all_livinglab_link_ids = set(cluster_map.keys())
livinglab_gdf = gdf[gdf['LINK_ID'].astype(str).isin(all_livinglab_link_ids)]

num_clusters = max([max(v) for v in cluster_map.values()]) + 1
num_groups = 4
clusters_per_group = (num_clusters + num_groups - 1) // num_groups
cmap = plt.get_cmap('tab20', clusters_per_group)

for group_idx in range(num_groups):
    start_cid = group_idx * clusters_per_group
    if group_idx == num_groups - 1:
        end_cid = num_clusters
    else:
        end_cid = min((group_idx + 1) * clusters_per_group, num_clusters)
    fig, ax = plt.subplots(figsize=(16, 16))
    livinglab_gdf_3857 = livinglab_gdf.to_crs(epsg=3857)
    livinglab_gdf_3857.plot(ax=ax, color='lightgray', linewidth=1, zorder=1)
    ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik, crs='EPSG:3857')
    for i, cluster_id in enumerate(range(start_cid, end_cid)):
        cluster_link_ids = set()
        for link_id, clusters in cluster_map.items():
            if cluster_id in clusters:
                cluster_link_ids.add(link_id)
        cluster_links = livinglab_gdf[livinglab_gdf['LINK_ID'].astype(str).isin(cluster_link_ids)]
        if not cluster_links.empty:
            cluster_links_3857 = cluster_links.to_crs(epsg=3857)
            color = cmap(i % cmap.N)
            cluster_links_3857.plot(ax=ax, color=color, linewidth=2, zorder=2, label=f'Cluster {cluster_id}')
    minx, miny, maxx, maxy = livinglab_gdf_3857.total_bounds
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    ax.set_title(f'LivingLab Clusters {start_cid}~{end_cid-1} Overlay', fontsize=20)
    ax.axis('off')
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize=10, ncol=2)
    out_path = os.path.join(RESULT_DIR, f'clusters_{start_cid}_{end_cid-1}_overlay.png')
    plt.savefig(out_path, bbox_inches='tight', dpi=200)
    plt.close(fig)
    print(f"[INFO] Saved: {out_path}")
