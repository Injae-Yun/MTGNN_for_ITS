import numpy as np
import csv

from db.postgres_manager import Postgres



pg = Postgres()

file_path = 'valid_link_id.pickle.npy'
data = np.load(file_path)

# NumPy 문자열 객체를 일반 파이썬 문자열 리스트로 변환
cleaned_list = list(map(str, data))

start = '2025-06-22 00:00:00'
end = '2025-06-23 00:00:00'

results = pg.select_links(start, end, cleaned_list)



with open('results.csv', mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerows(results)

print("결과가 'results.csv'에 저장되었습니다.")