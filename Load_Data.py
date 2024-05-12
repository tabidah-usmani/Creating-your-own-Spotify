# Load_Data.py

from pyspark.sql import SparkSession

# Create Spark session
spark = SparkSession.builder \
    .appName("LoadAudioData") \
    .getOrCreate()

# Load audio files
df = spark.read.format("binaryFile") \
    .load("file:///media/mary/New Volume/sample_audio/")

# Display data
df.show()

# Stop Spark session
spark.stop()



