import subprocess, json, os, gc
import pandas as pd
import fastparquet
from pymongo import MongoClient
from datetime import datetime, timedelta
from subprocess import Popen, PIPE
import multiprocessing
from pymongo.errors import AutoReconnect
import time
import pyarrow as pa
import pyarrow.parquet as pq
import psycopg2
from psycopg2.extras import execute_values
from psycopg2 import OperationalError
from datetime import datetime, timedelta

class Postgres:
    #
    def __init__(self,db_config):
        self.db_config = db_config
        self._connect()

    def _connect(self):
        """새로운 커넥션을 생성합니다."""
        # 기존 커넥션이 있으면 닫고
        try:
            self.conn.close()
        except Exception:
            pass
        # 다시 연결
        self.conn = psycopg2.connect(
            host     = self.db_config['host'],
            port     = self.db_config['port'],
            database = self.db_config['database'],
            user     = self.db_config['user'],
            password = self.db_config['password'],
        )
        self.conn.autocommit = False

    # info.link
    # orgin data
    def select_links(self,
        date: datetime,
        link_ids: list[str],        
    ):
        cursor = self.conn.cursor()
        table_date = date.strftime("%y%m")
        start_date = date.strftime("%Y-%m-%d 00:00:00")
        end_date = (date + timedelta(days=1)).strftime("%Y-%m-%d 00:00:00")
        # 1) 임시 테이블 생성
        cursor.execute("DROP TABLE IF EXISTS temp_links")
        cursor.execute("CREATE TEMP TABLE temp_links(link_id TEXT PRIMARY KEY)")
        
        # 2) 임시 테이블에 link_ids 삽입 (bulk copy 사용)
        psycopg2.extras.execute_values(
            cursor,
            "INSERT INTO temp_links (link_id) VALUES %s",
            [(lid,) for lid in link_ids],
            page_size=10000  # 적당한 batch size 설정
        )
        # 3) JOIN을 활용한 쿼리 수행
        query = f"""
            SELECT il.*
            FROM info.link_{table_date} il
            JOIN temp_links tl ON il.link_id = tl.link_id
            WHERE il.reg_date >= %s AND il.reg_date < %s
        """
        cursor.execute(query, (start_date, end_date))

        results = cursor.fetchall()
        
        self.conn.commit()
        cursor.close()

        return results


def archive_to_parquet(
    archive_path,
    parquet_path,
    projection=None,
    logger=None
):
    # 병렬 해제를 위해 프로세스 수 결정
    n_threads = multiprocessing.cpu_count()

    # 1) pigz -dc -p N archive_path  |  bsondump
    p1 = Popen(
        ["pigz", "-dc", "-p", str(n_threads), archive_path],
        stdout=PIPE
    )
    p2 = Popen(
        ["bsondump"], 
        stdin=p1.stdout, 
        stdout=PIPE, 
        text=True
    )
    p1.stdout.close()  # 파이프 연결

    # 2) JSON → 리스트
    docs = []
    for line in p2.stdout:
        obj = json.loads(line)
        if projection:
            obj = {k: obj.get(k) for k,v in projection.items() if v==1}
        docs.append(obj)
    p2.stdout.close()
    p2.wait()
    p1.wait()

    # 이후 df → fastparquet.write(...) 동일
    df = pd.DataFrame(docs)
    if projection:
        cols = [k for k,v in projection.items() if v==1]
        df = df[cols]
    fastparquet.write(parquet_path, df,
                      compression="SNAPPY",
                      write_index=False)
    if logger:
        logger.info(f"✅ BSON→Parquet 완료: {parquet_path} ({len(docs)}건)")

def load_from_mongo(link_ids, dates, time_points,
                    projection, mongo_config,
                    output_dir, target, logger,
                    batch_size=10000, parquet_chunk=20000):
    os.makedirs(output_dir, exist_ok=True)
    client_uri = f"mongodb://{mongo_config['host']}:{mongo_config['port']}/{mongo_config['db']}"
    logger.info("🚀 시작 – MongoDB → Parquet 스트리밍")

    # time_points map: { 'YYYY-MM-DD': ['HH:MM', ...] }
    date_time_map = {}
    for tp in time_points or []:
        dt = tp if isinstance(tp, datetime) else datetime.strptime(tp, "%Y-%m-%d %H:%M")
        key = dt.strftime("%Y-%m-%d")
        date_time_map.setdefault(key, []).append(dt.strftime("%H:%M"))

    # link_id 타입 혼합 매칭 대비 (str/int 동시 포함)
    # - DB 필드가 int인데 리스트가 str이면 미스매치 발생 → 둘 다 넣어 매칭
    link_ids = link_ids.tolist()
    link_ids_mixed = []
    for lid in link_ids:
        link_ids_mixed.append(lid)
        try:
            link_ids_mixed.append(int(lid))
        except Exception:
            pass
    # 중복 제거
    try:
        from collections import OrderedDict
        link_ids_mixed = list(OrderedDict((x, None) for x in link_ids_mixed).keys())
    except Exception:
        link_ids_mixed = list(dict.fromkeys(link_ids_mixed))
    results = []
    cilent = MongoClient(mongo_config["host"], mongo_config["port"],
                serverSelectionTimeoutMS=60000,  # 서버 셀렉션 60초
                socketTimeoutMS=120000,          # 읽기 타임아웃 120초
                connectTimeoutMS=20000           # 연결 시도 타임아웃 20초                      
                )
#        logger.info(cilent)
#        logger.info(mongo_config)

    db = cilent[mongo_config["db"]]
    # 청크 크기: 대역폭/컬렉션 구조에 따라 조절
    interval = timedelta(hours=3) if target == 'Korea' else timedelta(hours=24)

    for date in dates:
        coll = db[f"traffic_linkdata_{date}"]
        logger.info(f"📡 [{date}] 컬렉션 연결")
        # build query
        iso = f"{date[:4]}-{date[4:6]}-{date[6:]}"
        # 기본 링크 필터
        query = {"link_id": {"$in": link_ids_mixed}}
        # 필요 시점이 정의되어 있으면 $in 으로만 제한 (불필요한 range 결합 제거)
        #   - 일부 환경에서 insert_time 저장 포맷이 HH:MM:SS인 경우도 있어
        #     range만 남겨두면 전체일 조회가 발생할 수 있음. (아래에서 보완)
        time_list = date_time_map.get(iso, [])
        if time_list:
            query["insert_time"] = {"$in": time_list}
        # prepare output
        out_path = os.path.join(output_dir, f"{date}.parquet")
        buffer, written = [], False
        all_docs = []
        # 00:00부터 다음날 00:00 직전까지 6시간씩 쪼개기
        day_start = datetime.strptime(iso, "%Y-%m-%d")
        chunk_start = day_start
        day_end = day_start + timedelta(days=1)-timedelta(minutes=1) # 23:59

        while chunk_start < day_end:
            chunk_end = min(chunk_start + interval, day_end)
            time_gte = chunk_start.strftime("%H:%M")
            time_lt  = chunk_end.strftime("%H:%M")

            # 이 구간 전용 쿼리 구성
            q = query.copy()
            if time_list:
                # 이 청크에 해당하는 분 단위 타임만 추리기
                times_in_chunk = [t for t in time_list if (t >= time_gte and t < time_lt)]
                if not times_in_chunk:
                    chunk_start = chunk_end
                    continue  # 이 구간엔 조회할 분이 없음
                # HH:MM 와 HH:MM:00 두 형식을 모두 포함하여 정확 매칭 ($in)만 사용
                times_exact = list(dict.fromkeys(
                    times_in_chunk + [t + ":00" for t in times_in_chunk]
                ))
                q["insert_time"] = {"$in": times_exact}
            else:
                # date_range 등: range 필터 적용
                time_filter = {"insert_time": {"$gte": time_gte, "$lt": time_lt}}
                if q:
                    q = {"$and": [q, time_filter]}
                else:
                    q = time_filter

            # retry 로직으로 이 구간 데이터 가져오기
            retry = 0
            max_retry = 12
            temp = None
            t0 = time.time()
            while retry < max_retry:
                try:
                    cursor = coll.find(q, projection, batch_size=batch_size)
                    temp = []
                    for doc in cursor:
                        temp.append(doc)
                    # 성공적으로 다 읽으면 break
                    break
                except AutoReconnect as e:
                    logger.warning(f"⚠️ AutoReconnect: {e}. 재시도 {retry+1}/{max_retry}")
                    retry += 1
                    time.sleep(5)
                finally:
                    try: cursor.close()
                    except: pass

            if temp is None:
                logger.error(f"❌ [{date} {time_gte}-{time_lt}] 최대 재시도 초과, 스킵")
            elif not temp:
                logger.info(f"ℹ️ [{date} {time_gte}-{time_lt}] 문서 없음")
            else:
                all_docs.extend(temp)
                dt_sec = time.time() - t0
                logger.info(f"✅ [{date} {time_gte}-{time_lt}] {len(temp)}건 조회 (%.2fs)" % dt_sec)

            # 다음 6시간으로
            chunk_start = chunk_end

        # 하루 전체 모은 뒤 Parquet 저장
        if not all_docs:
            logger.warning(f"⚠️ [{date}] 전체 데이터 없음, 다음 날짜로")
            continue
        # DataFrame 생성
        df = pd.DataFrame(all_docs)

        # projection이 있다면 필요한 컬럼만 추출
        if projection:
            cols = [k for k, v in projection.items() if v == 1]
            df = df[cols]

        # Parquet으로 한 번에 저장
        t0 = time.time()
        fastparquet.write(
            out_path, 
            df, 
            compression="SNAPPY", 
            write_index=False
        )
        logger.info(f"💾 저장 완료: {out_path} rows={len(df)} (%.2fs)" % (time.time() - t0))

        # 메모리 정리
        del  all_docs, df
        gc.collect()

    logger.info("📦 모든 날짜 Parquet 변환 완료")


def load_from_postgres(link_ids, dates, time_points,
                    Task_config, db_config,
                    output_dir, target, logger,
                    parquet_chunk=25000):
    os.makedirs(output_dir, exist_ok=True)
    pg = Postgres(db_config)
    logger.info("🚀 시작 – Postgres DB → Parquet 스트리밍")

    # time_points map
    date_time_map = {}
    for tp in time_points or []:
        dt = tp if isinstance(tp, datetime) else datetime.strptime(tp, "%Y-%m-%d %H:%M")
        key = dt.strftime("%Y-%m-%d")
        date_time_map.setdefault(key, []).append(dt.strftime("%H:%M"))
    link_ids = link_ids.tolist()

    # if target == 'Korea':
    #     interval = timedelta(hours=3)  # 6시간 간격
    # else:
    #     interval = timedelta(hours=24)
    max_retries =5
    backoff_sec= 5.0
    counts = 0

    interest = Task_config['far_time_process']["interest"]   # e.g. [-21,...,0]
    offsets  = interest[:]
    interval = Task_config['interval']  # in minutes, e.g. 5
    for date in dates:        
        if type(date) != str:
            date_dt = date.strftime("%Y%m%d")
        else:
            date_dt = datetime.strptime(date,"%Y%m%d")
        for ci, off in enumerate(offsets):
            d = date_dt + timedelta(days=off)
            fn = os.path.join(output_dir, d.strftime("%Y%m%d") + ".parquet")
        # 1) 이미 파일이 있으면 스킵
            if os.path.exists(fn):
                logger.info(f"[{d}] 이미 처리된 날짜, 스킵")
                continue
            writer = None
            try:
                for i in range(0, len(link_ids), parquet_chunk):
                    chunk = link_ids[i:i+parquet_chunk]

                    all_docs = pg.select_links(d, chunk)
                    # --- retry wrapper ---
                    for attempt in range(max_retries):
                        try:
                            all_docs = pg.select_links(d, chunk)
                            break
                        except OperationalError as e:
                            msg = str(e)
                            if "SSL SYSCALL error: EOF detected" in msg and attempt < max_retries - 1:
                                logger.warning(
                                    f"[{d:%Y-%m-%d}] SSL EOF, reconnecting and retrying "
                                    f"(attempt {attempt+1}/{max_retries})"
                                )
                                pg._connect()               # 커넥션 재생성
                                time.sleep(backoff_sec)     # backoff
                                continue
                            else:
                                logger.error(
                                    f"[{date:%Y-%m-%d}] select_links failed: {e!r}"
                                )
                                raise
                    else:
                        # retry 루프를 break 없이 빠져나왔다면 실패
                        raise RuntimeError(f"Failed after {max_retries} retries")
                    
                    # 하루 전체 모은 뒤 Parquet 저장
                    if not all_docs:
                        logger.warning(f"⚠️ [{date}] 전체 데이터 없음, 다음 날짜로")
                        df, df_final, table =[], [],[]
                        continue
                    # DataFrame 생성
                    cols = db_config["tables"]["info.link"]["columns"]
                    df = pd.DataFrame(all_docs, columns=cols)

                    # projection이 있다면 필요한 컬럼만 추출
                    projection = Task_config['far_time_process'].get('pg_projection', None)
                    explode_list = Task_config['far_time_process'].get('explode_list', None)
                    if projection:
                        cols = [k for k, v in projection.items() if v == 1]
                        df = df[cols]
                    df_exp = df.explode('data').reset_index(drop=True)
                    data_expanded = pd.json_normalize(df_exp['data'])
                    if explode_list:
                        # 안전하게, 존재하는 컬럼만 필터
                        valid_cols = [c for c in explode_list if c in data_expanded.columns]
                        data_expanded = data_expanded[valid_cols]
                    df_final = pd.concat([
                        df_exp.drop(columns=['data']),
                        data_expanded
                    ], axis=1)
                    # 1) speed, pty 컬럼을 pandas nullable Int64로 변환 (NaN 허용)
                    for col in ["speed", "pty"]:
                        if col in df_final.columns:
                            # NaN을 -1 등 특정 값으로 대체할 거면 fillna(-1) 후 astype('int64') 해도 됩니다.
                            df_final[col] = df_final[col].round().astype(pd.Int64Dtype())
                    table = pa.Table.from_pandas(df_final, preserve_index=False)

                    # 4) ParquetWriter 생성(첫 블록에만)
                    if writer is None:
                        writer = pq.ParquetWriter(
                            fn,
                            schema=table.schema,
                            compression='snappy'
                        )

                    # 5) RowGroup 단위로 append
                    writer.write_table(table)
                    logger.info(f"[{d:%Y-%m-%d}] chunk {i}-{i+parquet_chunk}: write_table")
                    # chunk 루프 끝나면 반드시 close
                    del  all_docs, df_final, table
                    gc.collect()

            except Exception as e:
                # 4) 에러 발생 시, 이미 만들어진 파일 삭제
                if writer:
                    writer.close()
                if os.path.exists(fn):
                    os.remove(fn)
                    logger.warning(f"[{d}] 처리 중 에러 발생, 임시파일 삭제: {fn}")
                raise  # 에러를 다시 던져서 호출부에서 알 수 있게

            else:
                # 정상 완료 시
                if writer:
                    writer.close()
                    logger.info(f"[{d}] parquet file completed: {d}")
                else:
                    logger.warning(f"[{d}] 전체 데이터 없음, parquet 파일 미생성")
            # 메모리 정리
            gc.collect()

    logger.info("📦 모든 날짜 Parquet 변환 완료")