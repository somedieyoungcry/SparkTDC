import unittest
from pyspark.sql import SparkSession
import pyspark.sql.functions as f
from trys.transformations.transformations import Transformations


class TestAT(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.spark = SparkSession.builder.master("local[*]").appName("ATtest").getOrCreate()
        self.transformations = Transformations()

    @classmethod
    def tearDownClass(self):
        self.spark.stop()

    def test_no_null_records(self):
        data = [
            ("id1", "user1", "brand1", None),
            ("id2", None, "brand2", 20),
            (None, "user3", None, 15),
        ]
        schema = ["id_policy", "id_client", "id_car_brand", "accident_count"]
        df = self.spark.createDataFrame(data, schema)

        result_df = self.transformations.update_clasification(df)

        null_count = result_df.select(
            [f.count(f.when(f.col(c).isNull(), c)).alias(c) for c in result_df.columns]).collect()
        self.assertTrue(all(count == 0 for count in null_count[0]))

    def test_all_records_uppercase(self):
        data = [
            ("id1", "user1", "brand1", 20),
            ("id2", "user2", "brand2", 15),
            ("id3", "user3", "brand3", 10),
        ]
        schema = ["id_policy", "id_client", "id_car_brand", "accident_count"]
        df = self.spark.createDataFrame(data, schema)

        result_df = self.transformations.update_clasification(df)

        uppercase_cols = ["id_policy", "id_client", "id_car_brand"]
        uppercase_count = result_df.select(
            [f.count(f.when(f.upper(f.col(c)) == f.col(c), c)).alias(c) for c in uppercase_cols]).collect()
        self.assertTrue(all(count == 3 for count in uppercase_count[0]))


if __name__ == "__main__":
    unittest.main()
