from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("Music Recommendation System") \
    .getOrCreate()

# defines the data path explicitly with a file scheme
data_path = "file:///home/mary/Downloads/merged_data.csv"

# reads the CSV file into a DataFrame
df = spark.read.csv(data_path, header=True, inferSchema=True)

# displays the first few rows
df.show(5)

# prints the schema of the DataFrame to check data types
print("Total number of entries: ", df.count())
df.printSchema()

spark.stop()
