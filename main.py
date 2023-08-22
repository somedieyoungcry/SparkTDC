from pyspark.sql import SparkSession
import pyspark.sql.functions as f
import trys.constants.constants as c
from trys.transformations.transformations import Transformations


def main():
    spark = SparkSession.builder.appName(c.APP_NAME) \
        .master(c.MODE) \
        .config(c.POLICY, c.LEGACY) \
        .getOrCreate()

    t = Transformations()

    policy_df = spark.read.parquet(c.POLICY_PARQUET)
    clients_df = spark.read.parquet(c.CLIENTS_PARQUET)
    cars_df = spark.read.parquet(c.CARS_PARQUET)
    incidents_df = spark.read.parquet(c.INCIDENTS_PARQUET)


    print("Hacemos el join")
    seguros_output = t.joined_dfs(policy_df, clients_df, cars_df, incidents_df)
    seguros_output.printSchema()
    seguros_output.show()

    """print("Funcion rank 100")
    rankTop = t.rank_incidents(seguros_output)
    rankTop_null = rankTop.na.drop(subset=[c.CARID])
    rankTop_null.show()

    print("Rank por colores")
    rankColor = t.rank_colors(seguros_output)
    rankColor_null = rankColor.na.drop(subset=[c.CAR_COLOR])
    rankColor_null.show()

    print("Rank por paises")
    rankCountry = t.rank_countrys(seguros_output)
    rankCountry_null = rankCountry.na.drop(subset=[c.COUNTRY_NAME])
    rankCountry_null.show()

    print("Rank por edades")
    rankAge = t.top10_ages(seguros_output)
    rankAge_null = rankAge.na.drop(subset=[c.USER_AGE])
    rankAge_null.show()

    print("Clasificacion")
    clasification = t.update_clasification(seguros_output)
    clasification_drop = clasification.dropDuplicates()
    clasification_null = clasification_drop.na.drop("all")
    clasification_null.show()
    # clasification.printSchema()"""


if __name__ == "__main__":
    main()
