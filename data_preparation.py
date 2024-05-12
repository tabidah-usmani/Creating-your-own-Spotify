from pyspark.sql import SparkSession

# Initialize a Spark session
spark = SparkSession.builder \
    .appName("Music Recommendation System") \
    .getOrCreate()

# Define the data path explicitly with a file scheme
data_path = "file:///home/mary/Downloads/merged_data.csv"

# Read the CSV file into a DataFrame
df = spark.read.csv(data_path, header=True, inferSchema=True)

# Display the first few rows of the DataFrame to confirm correct data loading
df.show(5)

# Print the schema of the DataFrame to check data types
print("Total number of entries: ", df.count())
df.printSchema()


# Stop the Spark session
spark.stop()

