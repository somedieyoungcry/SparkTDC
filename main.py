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

    # policy_df.printSchema()
    # clients_df.printSchema()
    # cars_df.printSchema()
    # incidents_df.printSchema()


    print("Hacemos el join")
    creditos_output = t.joined_dfs(policy_df, clients_df, cars_df, incidents_df)
    # creditos_output.printSchema()
    creditos_output.show()

    print("Funcion rank 100")
    rankTop = t.rank_incidents(creditos_output)
    rankTop_null = rankTop.na.drop(subset=[c.CARID])
    rankTop_null.show()

    print("Rank por colores")
    rankColor = t.rank_colors(creditos_output)
    rankColor_null = rankColor.na.drop(subset=[c.CAR_COLOR])
    rankColor_null.show()

    print("Rank por paises")
    rankCountry = t.rank_countrys(creditos_output)
    rankCountry_null = rankCountry.na.drop(subset=[c.COUNTRY_NAME])
    rankCountry_null.show()

    print("Rank por edades")
    rankAge = t.top10_ages(creditos_output)
    rankAge_null = rankAge.na.drop(subset=[c.USER_AGE])
    rankAge_null.show()

    print("Clasificacion")
    clasification = t.update_clasification(creditos_output)
    clasification_drop = clasification.dropDuplicates()
    clasification_null = clasification_drop.na.drop("all")
    clasification_null.show()
    # clasification.printSchema()


if __name__ == "__main__":
    main()
