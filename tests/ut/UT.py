import unittest
from pyspark.sql import SparkSession
from trys.transformations.transformations import Transformations


class TestUT(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.spark = SparkSession.builder.master("local[*]").appName("UTtest").getOrCreate()
        self.transformations = Transformations()

    @classmethod
    def tearDownClass(self):
        self.spark.stop()

    def test_rank_incidents(self):
        data = [
            ("brand1", 10),
            ("brand2", 20),
            ("brand3", 15),
        ]
        schema = ["id_car_brand", "accident_count"]
        df = self.spark.createDataFrame(data, schema)

        result_df = self.transformations.rank_incidents(df)
        expected_data = [
            ("brand2", 20, 1),
            ("brand3", 15, 2),
            ("brand1", 10, 3),
        ]
        expected_schema = ["id_car_brand", "accident_count", "rank"]
        expected_df = self.spark.createDataFrame(expected_data, expected_schema)

        self.assertTrue(result_df.collect() == expected_df.collect())


if __name__ == "__main__":
    unittest.main()
