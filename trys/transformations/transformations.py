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

        return joined_df3

    @staticmethod
    def rank_incidents(df: DataFrame) -> DataFrame:
        count_incidents = df.groupBy(c.CARID).agg(f.count("*").alias(c.ACCIDENT_COUNT))
        window_spec = Window.orderBy(f.col(c.ACCIDENT_COUNT).desc())
        rank_incidents = count_incidents.withColumn(c.RANK, f.row_number().over(window_spec) - 1)

        return rank_incidents

    @staticmethod
    def rank_colors(df: DataFrame) -> DataFrame:
        count_colors = df.groupBy(c.CAR_COLOR).agg(f.count("*").alias(c.ACCIDENT_COUNT))
        window_func = Window.orderBy(f.col(c.ACCIDENT_COUNT).desc())
        rank_colors = count_colors.withColumn(c.RANK, f.row_number().over(window_func) - 1)

        return rank_colors

    @staticmethod
    def rank_countrys(df: DataFrame) -> DataFrame:
        count_countries = df.groupBy(c.COUNTRY_NAME).agg(f.count("*").alias(c.ACCIDENT_COUNT))
        window_spec = Window.orderBy(f.col(c.ACCIDENT_COUNT).desc())
        rank_countries = count_countries.withColumn(c.RANK, f.row_number().over(window_spec) - 1)

        return rank_countries

    @staticmethod
    def top10_ages(df: DataFrame) -> DataFrame:
        age_accident_count = df.groupBy(c.USER_AGE).agg(f.count("*").alias(c.ACCIDENT_COUNT))
        window_spec = Window.orderBy(f.col(c.ACCIDENT_COUNT).desc())
        rank_ages = age_accident_count.withColumn(c.RANK, f.row_number().over(window_spec) - 1)
        top_10_ages = rank_ages.filter(rank_ages.rank <= 10)

        return top_10_ages

    def update_clasification(self, df: DataFrame) -> DataFrame:
        color_rank_df = self.rank_colors(df)
        max_accident_color = color_rank_df.select(c.CAR_COLOR).first()[c.CAR_COLOR]
        country_rank_df = self.rank_countrys(df)
        max_accident_country = country_rank_df.select(c.COUNTRY_NAME).first()[c.COUNTRY_NAME]
        top_10_ages_df = self.top10_ages(df)
        avg_age = top_10_ages_df.select(f.avg(c.USER_AGE)).first()[0]
        incidents_rank_df = self.rank_incidents(df)
        incidents_rank_df = incidents_rank_df.withColumnRenamed(c.CARID,
                                                                c.RANK_CAR_BRAND)
        df_with_incidents = df.join(incidents_rank_df, df[c.CARID] == incidents_rank_df[c.RANK_CAR_BRAND],
                                    c.JOIN3)
        updated_df = df_with_incidents.withColumn(c.POINTS,
                                                  f.when(df_with_incidents[c.VELOCITY_MAX] >= 200, 2).otherwise(
                                                      0) +
                                                  f.when(df_with_incidents[c.USER_AGE] > avg_age, 3).otherwise(0) +
                                                  f.when(df_with_incidents[c.CAR_COLOR] == max_accident_color,
                                                         1).otherwise(0) +
                                                  f.when(df_with_incidents[c.COUNTRY_NAME] == max_accident_country,
                                                         3).otherwise(0) +
                                                  f.when(df_with_incidents[c.ACCIDENT_COUNT] ==
                                                         df_with_incidents.select(f.max(c.ACCIDENT_COUNT)).first()[0],
                                                         2).otherwise(0)
                                                  )

        updated_df = updated_df.withColumn(c.CLASSIFICATION,
                                           f.when(updated_df[c.POINTS] <= 5, "C").otherwise(
                                               f.when(updated_df[c.POINTS] >= 9, "A").otherwise("B")
                                           )
                                           )

        updated_df = updated_df.na.drop(c.ALL)

        return updated_df
