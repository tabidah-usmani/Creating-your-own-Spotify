from pyspark.sql import SparkSession  

# creates Spark session
spark = SparkSession.builder \
    .appName("LoadAudioData") \
    .getOrCreate()

# loads audio files into DataFrame
df = spark.read.format("binaryFile") \
    .load("file:///media/mary/New Volume/sample_audio/")

# displays loaded data
df.show()

# stops the Spark session to free resources
spark.stop()
