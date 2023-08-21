from pyspark.sql import SparkSession
import pyspark.sql.functions as f
import trys.constants.constants as c
from trys.transformations.transformations import Transformations


def main():
    spark = SparkSession.builder.appName("sparkMain") \
        .master("local[*]") \
        .config("spark.sql.legacy.timeParserPolicy", "LEGACY") \
        .getOrCreate()

    t = Transformations()

    policy_df = spark.read.parquet(
        "resources/cuttofdate=2023-08-16_2000_3A00_3A00/part-00000-14be24e8-07ac-45eb-a56f-6fb1e56618ec.c000.snappy.parquet")
    clients_df = spark.read.parquet(
        "resources/user_charge_date=2023-08-17/part-00000-cd81d3c7-8ee7-44f5-8c28-91009fc70e91.c000.snappy.parquet")
    cars_df = spark.read.parquet("resources/CARS/part-00000-5c2c513b-9db5-4736-9e00-b2604c4f9ea4-c000.snappy.parquet")
    incidents_df = spark.read.parquet(
        "resources/load_date=2023-08-17/part-00000-29af0d9c-ea76-4b7a-8e51-ffbfa3a592cb.c000.snappy.parquet")

    policy_df.printSchema()
    clients_df.printSchema()
    cars_df.printSchema()
    incidents_df.printSchema()


    print("Hacemos el join")
    creditos_output = t.joined_dfs(policy_df, clients_df, cars_df, incidents_df)
    # creditos_output.printSchema()
    # creditos_output.show()

    print("Funcion rank 100")
    rankTop = t.rank_incidents(creditos_output)
    rankTop_null = rankTop.na.drop(subset=["id_car_brand"])
    rankTop_null.show()

    print("Rank por colores")
    rankColor = t.rank_colors(creditos_output)
    rankColor_null = rankColor.na.drop(subset=["car_color"])
    rankColor_null.show()

    print("Rank por paises")
    rankCountry = t.rank_countrys(creditos_output)
    rankCountry_null = rankCountry.na.drop(subset=["country_name"])
    rankCountry_null.show()

    print("Rank por edades")
    rankAge = t.top10_ages(creditos_output)
    rankAge_null = rankAge.na.drop(subset=["user_age"])
    rankAge_null.show()

    print("Clasificacion")
    clasification = t.update_clasification(creditos_output)
    clasification_drop = clasification.dropDuplicates()
    clasification_null = clasification_drop.na.drop("all")
    clasification_null.show()
    # clasification.printSchema()


if __name__ == "__main__":
    main()
