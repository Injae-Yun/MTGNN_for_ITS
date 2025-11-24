from datetime import datetime, timedelta
import math


def get_focus_timepoints(task_config, logger):
    mode = task_config['mode']  # mode 변수 추가
    
    if mode == "date_range":
        start = datetime.strptime(task_config["date_range"]["start"], "%Y-%m-%d")
        end = datetime.strptime(task_config["date_range"]["end"], "%Y-%m-%d")
        
        # 기본 관심 기간
        base_dates = [
            start + timedelta(days=i)
            for i in range((end - start).days + 1)
        ]
        
        # interest와 interval로부터 필요한 과거 데이터 계산
        interval = task_config['interval']  # 5분
        interest = task_config['interest']  # [-12, -288, -2016] 
        
        # 가장 오래된 데이터가 필요한 시점 계산 (분 단위)
        max_lookback_minutes = abs(min(interest)) * interval    # 가장 오래된 기준점 + 12개 슬롯
        max_lookback_days = math.ceil(max_lookback_minutes / (24 * 60))   # 일 단위로 변환 + 여유분
        
        # 확장된 날짜 범위 계산
        extended_start = start - timedelta(days=max_lookback_days)
        extended_dates = [
            extended_start + timedelta(days=i)
            for i in range((end - extended_start).days + 1)
        ]
        
        # 포맷팅
        dates = [d.strftime("%Y%m%d") for d in extended_dates]
        
        logger.info(f"📆 date_range 모드: 총 {len(base_dates)}일 간 데이터 조회 예정 → {base_dates}")
        logger.info(f"📊 확장된 범위: {extended_start.strftime('%Y-%m-%d')} ~ {end.strftime('%Y-%m-%d')} (총 {len(dates)}개 파일)")
        logger.info(f"🔍 최대 lookback: {max_lookback_days}일 ({max_lookback_minutes}분)")
        
        time_points = []

    elif mode == "time_reference":
        # 기준 datetime
        time_ref = task_config["time_reference"]
        base_date = time_ref["date"]  # "2025-06-27"
        base_time = time_ref.get("time") or "00:00"
        base_dt = datetime.strptime(f"{base_date} {base_time}", "%Y-%m-%d %H:%M")

        # 총 36개 시점 생성: 12개 단위 간격 (1개는 -12~0, 1개는 -288~-276 등)
        interval = task_config['interval']  # 시간 간격 (분 단위)
        interest = task_config['interest']
        offsets = []
        for base in interest:
            offsets.extend([interval * i for i in range(base, base + 12)])
        time_points = [base_dt + timedelta(minutes=offset) for offset in offsets]

        # 날짜별 그룹화
        from collections import defaultdict
        time_index = defaultdict(list)
        for dt in time_points:
            date_str = dt.strftime("%Y%m%d")
            time_str = dt.strftime("%H:%M")
            time_index[date_str].append(time_str)

        dates = list(time_index.keys())
        logger.info(f"📌 기준 시점: {base_dt}")
        logger.info(f"📆 총 {len(dates)}개 날짜, {len(time_points)}개 시점 생성 완료")
        
    elif mode == "far_time_process":
        if task_config["date_range"]["start"] is None:
            start = datetime.strptime(datetime.now(), "%Y-%m-%d")
            end = datetime.strptime(datetime.now(), "%Y-%m-%d")
        else:
            start = datetime.strptime(task_config["date_range"]["start"], "%Y-%m-%d")
            end = datetime.strptime(task_config["date_range"]["end"], "%Y-%m-%d")
        
        base_dates = [
            start + timedelta(days=i)
            for i in range((end - start).days + 1)
        ]
        dates = [d.strftime("%Y%m%d") for d in base_dates]
        logger.info(f"📆 far_time_process 모드: 총 {len(dates)}일 간 데이터 조회 예정 → {base_dates}")
        time_points = []

    return dates, time_points
