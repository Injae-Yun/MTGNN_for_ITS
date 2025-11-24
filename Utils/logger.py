# utils/logger.py

import logging
import os
from datetime import datetime


def get_logger(name: str = "app", log_dir: str = "Logs", level: int = logging.INFO) -> logging.Logger:
    """
    로거 인스턴스를 생성하고 설정함.

    :param name: 로거 이름
    :param log_dir: 로그 파일 저장 디렉토리
    :param level: 로깅 레벨 (기본: INFO)
    :return: 설정된 로거 인스턴스
    """
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # 파일 이름: log_20250627_145501.log 같은 형식
    log_file = os.path.join(
        log_dir, f"log_{timestamp}.log"
    )

    # 로거 생성 및 설정
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()  # 중복 방지를 위해 기존 핸들러 제거
    logger.timestamp = timestamp # other module에서 timestamp 사용 가능

    # 파일 핸들러
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter(
        fmt="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    ))

    # 콘솔 핸들러
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter(
        fmt="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S"
    ))

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger
