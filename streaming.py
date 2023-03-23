from pyspark import SparkContext
from pyspark.streaming import StreamingContext
import sparknlp
from pyspark.ml import PipelineModel
from pyspark.sql.functions import explode
from pyspark.sql.functions import split
spark = sparknlp.start(m1=True)

"""
sc = spark.sparkContext
ssc = StreamingContext(sc, 1)

pipeline = PipelineModel.load("model")

lines = ssc.socketTextStream("localhost", 9999)

if lines:
	data = spark.createDataFrame([[lines]]).toDF("tweet")
	result = pipeline.transform(data).first()["category"][0]["result"]
	print(result)
ssc.start()
ssc.awaitTermination()
"""

lines = spark \
    .readStream \
    .format("socket") \
    .option("host", "localhost") \
    .option("port", 9999) \
    .load()

# Split the lines into words
tweet = lines.select(
   explode(
       split(lines.value, "\n")
   ).alias("tweet")
)
pipeline = PipelineModel.load("model")
result = pipeline.transform(tweet).select("tweet", "category.result")
# Generate running word count
#wordCounts = words.groupBy("word").count()

query = result \
    .writeStream \
    .format("console") \
    .option("truncate", False) \
    .start()

query.awaitTermination()