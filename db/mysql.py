import os

import dotenv
import pymysql
from pymysql.err import OperationalError


class MySQLConnection:
    def __init__(self):
        dotenv.load_dotenv()
        self.host = os.environ.get("host")
        self.mysql_port = int(os.environ.get("mysql_port"))
        self.mysql_user = os.environ.get("mysql_user")
        self.mysql_passwd = os.environ.get("mysql_passwd")
        self.mysql_db = os.environ.get("mysql_db")
        self.try_connect()

    def try_connect(self):
        self.conn = pymysql.connect(
            host=self.host,
            port=self.mysql_port,
            user=self.mysql_user,
            password=self.mysql_passwd,
            database=self.mysql_db,
        )

    def get_metadata(self, id):
        query = f"""
            SELECT A.id, image_url_small, atk, def, level, archetype, attribute, A.desc, frame_type, kor_desc, kor_name, name, race, type
            FROM card_model as A left join card_image as B on A.id=B.id where A.id={id};
        """
        cursor = self.execute_query(query)
        col_desc = cursor.description
        result = {}
        data = cursor.fetchone()
        for desc, value in zip(col_desc, data):
            key = desc[0]
            result[key] = value
        return result

    def execute_query(self, query, ttl=1):
        if ttl < 0:
            raise ConnectionError

        try:
            cursor = self.conn.cursor()
            cursor.execute(query)
            return cursor
        except OperationalError:
            self.try_connect()
            return self.execute_query(query, ttl - 1)

    def get_all_card_ids(self):
        query = f"SELECT id FROM card_model"
        cursor = self.execute_query(query)

        result = []
        for _ in range(cursor.rowcount):
            result.append(cursor.fetchone()[0])
        return result
