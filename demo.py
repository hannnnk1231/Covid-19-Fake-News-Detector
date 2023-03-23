
import sparknlp
from pyspark.ml import PipelineModel
spark = sparknlp.start(m1=True)

import streamlit as st

@st.cache(allow_output_mutation=True)
def load_pipeline(name):
    return PipelineModel.load(name)

@st.cache(allow_output_mutation=True)
def process_text(model_name, text):
    
    pipeline = load_pipeline(model_name)
    data = spark.createDataFrame([[text]]).toDF("tweet")
    return pipeline.transform(data).first()["category"][0]["result"]

# Init model
process_text("model", "init")

text_input = st.text_input("Enter any tweet 👇")

if text_input:
    st.write("This tweet is: ")
    result = process_text('model', text_input)
    if result == 'fake':
        st.error("Fake")
    else:
        st.success("Real")