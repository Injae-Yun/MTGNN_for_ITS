import psycopg2
from psycopg2.extras import execute_values



class Postgres:

    #
    def __init__(self):
        self.conn = psycopg2.connect(
            host="192.168.0.27",
            port=5432,
            database="traffic",
            user="bluesignal",
            password="qmffntlrmsjf!"
        )

    # info.link
    # orgin data
    def select_links(self,
        start_date: str, 
        end_date: str, 
        link_ids: list[str],        
    ):
        cursor = self.conn.cursor()
        start_date
        placeholders = ', '.join(['%s'] * len(link_ids))
        #table_date = start_date("Month",'Day')   # 2507                   
        query = f"""
            SELECT il.*
            FROM info.link_{table_date} il
            WHERE il.reg_date >= %s
            AND il.reg_date <  %s
            AND il.link_id IN ({placeholders})
        """

        params = [start_date, end_date] + link_ids
        cursor.execute(query, params)

        results = cursor.fetchall()
        
        self.conn.commit()
        cursor.close()

        return results

    
    

    

    


