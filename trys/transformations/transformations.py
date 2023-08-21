import pyspark.sql.functions as f
from pyspark.sql import Window
from pyspark.sql import DataFrame
import trys.constants.constants as c


class Transformations:
    def __init__(self):
        pass

    @staticmethod
    def joined_dfs(policy_df: DataFrame, clients_df: DataFrame, cars_df: DataFrame,
                   incidents_df: DataFrame) -> DataFrame:
        joined_df1 = policy_df.join(clients_df, policy_df[c.IDCLIENT] == clients_df[c.IDUSER], c.JOIN1)
        joined_df2 = joined_df1.join(cars_df, policy_df[c.CARID] == cars_df[c.BRANDID], c.JOIN1)
        joined_df3 = joined_df2.join(incidents_df, policy_df[c.IDPOLICY] == incidents_df[c.IDPOL], c.JOIN2)

        """selected_fields = [
            policy_df["id_policy"],
            policy_df["id_client"],
            clients_df["user_email"],
            policy_df["id_car_brand"],
            cars_df["brand_name"],
            cars_df["country_name"],
            policy_df["car_color"],
            policy_df["sum_assured"],
            policy_df["car_model"],
            policy_df["end_date_validity"],
            clients_df["user_age"],
            incidents_df["incident_id"],
            incidents_df["repair_amount"]
        ]

        selected_df = joined_df3.select(*selected_fields)"""

        return joined_df3

    @staticmethod
    def rank_incidents(df: DataFrame) -> DataFrame:
        count_incidents = df.groupBy(c.CARID).agg(f.count("*").alias(c.ACCIDENTCOUNT))
        window_spec = Window.orderBy(f.col(c.ACCIDENTCOUNT).desc())
        rank_incidents = count_incidents.withColumn(c.RANK, f.row_number().over(window_spec) - 1)

        return rank_incidents

    @staticmethod
    def rank_colors(df: DataFrame) -> DataFrame:
        count_colors = df.groupBy("car_color").agg(f.count("*").alias("accident_count"))
        window_func = Window.orderBy(f.col("accident_count").desc())
        rank_colors = count_colors.withColumn("rank", f.row_number().over(window_func) - 1)

        return rank_colors

    @staticmethod
    def rank_countrys(df: DataFrame) -> DataFrame:
        count_countries = df.groupBy("country_name").agg(f.count("*").alias("accident_count"))
        window_spec = Window.orderBy(f.col("accident_count").desc())
        rank_countries = count_countries.withColumn("rank", f.row_number().over(window_spec) - 1)

        return rank_countries

    @staticmethod
    def top10_ages(df: DataFrame) -> DataFrame:
        age_accident_count = df.groupBy("user_age").agg(f.count("*").alias("accident_count"))
        window_spec = Window.orderBy(f.col("accident_count").desc())
        rank_ages = age_accident_count.withColumn("rank", f.row_number().over(window_spec) - 1)
        top_10_ages = rank_ages.filter(rank_ages.rank <= 10)

        return top_10_ages

    def update_clasification(self, df: DataFrame) -> DataFrame:
        color_rank_df = self.rank_colors(df)
        max_accident_color = color_rank_df.select("car_color").first()["car_color"]
        country_rank_df = self.rank_countrys(df)
        max_accident_country = country_rank_df.select("country_name").first()["country_name"]
        top_10_ages_df = self.top10_ages(df)
        avg_age = top_10_ages_df.select(f.avg("user_age")).first()[0]
        incidents_rank_df = self.rank_incidents(df)
        incidents_rank_df = incidents_rank_df.withColumnRenamed("id_car_brand",
                                                                "rank_id_car_brand")
        df_with_incidents = df.join(incidents_rank_df, df["id_car_brand"] == incidents_rank_df["rank_id_car_brand"],
                                    "left_outer")
        updated_df = df_with_incidents.withColumn("points",
                                                  f.when(df_with_incidents["velocity_max_amount"] >= 200, 2).otherwise(
                                                      0) +
                                                  f.when(df_with_incidents["user_age"] > avg_age, 3).otherwise(0) +
                                                  f.when(df_with_incidents["car_color"] == max_accident_color,
                                                         1).otherwise(0) +
                                                  f.when(df_with_incidents["country_name"] == max_accident_country,
                                                         3).otherwise(0) +
                                                  f.when(df_with_incidents["accident_count"] ==
                                                         df_with_incidents.select(f.max("accident_count")).first()[0],
                                                         2).otherwise(0)
                                                  )

        updated_df = updated_df.withColumn("classification",
                                           f.when(updated_df["points"] <= 5, "C").otherwise(
                                               f.when(updated_df["points"] >= 9, "A").otherwise("B")
                                           )
                                           )

        updated_df = updated_df.na.drop("all")

        return updated_df
