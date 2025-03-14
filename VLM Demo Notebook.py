# Databricks notebook source
PIP_REQUIREMENTS = (
    "openai vllm>=0.7.2 httpx==0.27.2 "
    # "transformers==4.46.3 accelerate==1.0.0 "
    # "git+https://github.com/huggingface/transformers accelerate "
    " git+https://github.com/huggingface/transformers.git@336dc69d63d56f232a183a3e7f52790429b871ef "
    "mlflow==2.19.0 "
    "git+https://github.com/stikkireddy/mlflow-extensions.git@v0.17.0 "
    "qwen-vl-utils[decord]"
)
%pip install {PIP_REQUIREMENTS}

dbutils.library.restartPython()

# COMMAND ----------

PIP_REQUIREMENTS = (
    "openai vllm>=0.7.2 httpx==0.27.2 "
    # "transformers==4.46.3 accelerate==1.0.0 "
    # "git+https://github.com/huggingface/transformers accelerate "
    " git+https://github.com/huggingface/transformers.git@336dc69d63d56f232a183a3e7f52790429b871ef "
    "mlflow==2.19.0 "
    "git+https://github.com/stikkireddy/mlflow-extensions.git@v0.17.0 "
    "qwen-vl-utils[decord]"
)

# COMMAND ----------

# MAGIC %md
# MAGIC ###Change the below configs

# COMMAND ----------

CATALOG = 'kyra_wulffert'
SCHEMA = 'vision'
MODEL_NAME = 'qwen2_5_vl-7b'
ENDPOINT_NAME = 'qwen2_5_vl-7b_instruct'
# LOCAL_PATH_TO_MODEL = f"/Volumes/{CATALOG}/{SCHEMA}/hf_model/{MODEL_NAME}/""

# COMMAND ----------

from mlflow_extensions.serving.engines import VLLMEngineProcess
from mlflow_extensions.serving.engines.vllm_engine import VLLMEngineConfig
from mlflow_extensions.databricks.deploy.ez_deploy import EzDeployConfig, ServingConfig,EzDeployVllmOpenCompat

# COMMAND ----------

deployer = EzDeployVllmOpenCompat(
  config= EzDeployConfig(
    name="Qwen2.5-VL-7B-Instruct",
    engine_proc=VLLMEngineProcess,
    engine_config=VLLMEngineConfig(
          model="Qwen/Qwen2.5-VL-7B-Instruct", # copy the Hf link
          guided_decoding_backend="outlines",
          vllm_command_flags={
            "--gpu-memory-utilization": 0.98,
            "--distributed-executor-backend": "ray",
            "--enforce-eager": None,
            "--enable-auto-tool-choice": None,
            "--tool-call-parser" : "hermes",
          },
),
  serving_config=ServingConfig(
      # rough estimate for the engines this includes model weights + kv cache + overhead + intermediate states
      minimum_memory_in_gb=50,
  ),
  pip_config_override = PIP_REQUIREMENTS.split(" ")
),
  registered_model_name=f"{CATALOG}.{SCHEMA}.{MODEL_NAME}"
)

# COMMAND ----------

deployer.download()

# COMMAND ----------

deployer.register()

# COMMAND ----------

# MAGIC %md
# MAGIC # Below is the code to deploy the endpoint to model serving

# COMMAND ----------

deployer.deploy(ENDPOINT_NAME, scale_to_zero=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Kill the existing process and reload model from the UC (Run the below cell every time you want to restart the process)

# COMMAND ----------

from mlflow_extensions.testing.helper import kill_processes_containing

kill_processes_containing("vllm")
kill_processes_containing("ray")
kill_processes_containing("from multiprocessing")

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # Test Prompting on an Image
# MAGIC
# MAGIC ## Ask a question to an image

# COMMAND ----------

import urllib.request
from PIL import Image
from io import BytesIO 

image_url = "https://eoimages.gsfc.nasa.gov/images/imagerecords/87000/87551/iss045e032242_lrg.jpg"
with urllib.request.urlopen(image_url) as url:
    img = Image.open(BytesIO(url.read()))

display(img)

# COMMAND ----------

from openai import OpenAI
from mlflow.utils.databricks_utils import get_databricks_host_creds

workspace_host = spark.conf.get("spark.databricks.workspaceUrl")
model="qwen2_5_vl-7b_instruct"
endpoint = f"https://{workspace_host}/serving-endpoints/"
token = get_databricks_host_creds().token


# Initialize the OpenAI client with the Databricks serving endpoint
client = OpenAI(
    base_url=endpoint,
    api_key=token
)

prompt = "Analyze this satellite image of London at night. Identify areas with high, medium, and low lighting density. Categorize regions as residential, commercial, or industrial based on the light distribution. Estimate potential energy consumption variations across these zones and suggest energy-saving strategies such as smart lighting or renewable energy integration."

response = client.chat.completions.create(
  model=model,
  messages=[
        {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    # {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ]
            }
  ],
  temperature = 0.0,
  max_tokens = 1000
)

print(response.choices[0].message.content)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Compare to images and enforce an output schema

# COMMAND ----------

from IPython.display import display
from PIL import Image
import requests
from io import BytesIO
import matplotlib.pyplot as plt

url_1 = "https://images.unsplash.com/photo-1504198266287-1659872e6590"
url_2 = "https://www.arsenal.com/sites/default/files/styles/large_16x9/public/images/saka-celeb-bayern.png?h=3c8f2bed&auto=webp&itok=Twjeu8tug"

def load_image(url):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    return img

img1 = load_image(url_1)
img2 = load_image(url_2)

fig, ax = plt.subplots(1, 2, figsize=(10, 5))

ax[0].imshow(img1)
ax[0].axis('off')  
ax[0].set_title("Image 1")

ax[1].imshow(img2)
ax[1].axis('off')  
ax[1].set_title("Image 2")

plt.show()


# COMMAND ----------

from openai import OpenAI
from mlflow.utils.databricks_utils import get_databricks_host_creds
from pydantic import BaseModel, ValidationError
import typing as t
import json

class ImageAnalysis(BaseModel):
    image_details: str
    colors: t.List[str]
    has_human: bool
    human_gender: t.Literal["male", "female", "no human"]

workspace_host = spark.conf.get("spark.databricks.workspaceUrl")
model = "qwen2_5_vl-7b_instruct"
endpoint = f"https://{workspace_host}/serving-endpoints/"
token = get_databricks_host_creds().token

client = OpenAI(
    base_url=endpoint,
    api_key=token
)

schema_json = ImageAnalysis.model_json_schema()

prompt = f"""Analyze the given image and return a JSON response following this schema:
{schema_json}
Do not include markdown or extra text. Only return valid JSON."""

def analyze_image(image_url):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ]
            }
        ],
        temperature=0.0,
        max_tokens=1000
    )

    if response and response.choices and len(response.choices) > 0 and response.choices[0].message:
        return response.choices[0].message.content
    else:
        return None

# Image URLs
url_1 = "https://images.unsplash.com/photo-1504198266287-1659872e6590"
url_2 = "https://www.arsenal.com/sites/default/files/styles/large_16x9/public/images/saka-celeb-bayern.png?h=3c8f2bed&auto=webp&itok=Twjeu8tug"

image_1_result = analyze_image(url_1)
image_2_result = analyze_image(url_2)

try:
    image_1_data = ImageAnalysis.parse_raw(image_1_result)
    image_2_data = ImageAnalysis.parse_raw(image_2_result)

    # Step 7️⃣: Compare Results
    comparison_result = {
        "image_1_details": image_1_data.image_details,
        "image_2_details": image_2_data.image_details,
        "image_1_colors": image_1_data.colors,
        "image_2_colors": image_2_data.colors,
        "image_1_has_human": image_1_data.has_human,
        "image_2_has_human": image_2_data.has_human,
        "image_1_human_gender": image_1_data.human_gender,
        "image_2_human_gender": image_2_data.human_gender
    }

    print("Comparison Result:")
    print(json.dumps(comparison_result, indent=2))

except ValidationError as e:
    print("Schema Validation Failed:")
    print(e)


# COMMAND ----------

# MAGIC %md
# MAGIC #Load files from Volumes

# COMMAND ----------

# Point to your volume in unity catalog
catalog = "mlops_pj"
schema="gsk_gsc_cfu_count"
df_raw = (
    spark.readStream.format("cloudFiles")
    .option("cloudFiles.format", "binaryFile")
    .option("pathGlobfilter", f"*.jpg")
    .load(f"/Volumes/{catalog}/{schema}/jd_images")
)

# COMMAND ----------

df_raw.writeStream.trigger(availableNow=True).option(
        "checkpointLocation",
        f"/Volumes/{catalog}/{schema}/checkpoints/raw_imgs",
).toTable(f"{catalog}.{schema}.raw_img_bytes").awaitTermination()

# COMMAND ----------

df_img = spark.table(f"{catalog}.{schema}.raw_img_bytes")
display(df_img)

# COMMAND ----------

from pyspark.sql.functions import pandas_udf, regexp_replace
import pandas as pd


@pandas_udf("string")
def classify_img(images: pd.Series) -> pd.Series:

    from io import BytesIO 
    import base64
    from openai import OpenAI

    def classify_one_image(img): # We could update this to tak multiple parameters
        client = OpenAI(
                base_url=endpoint,
                api_key=token
            )

        image_file = BytesIO(img)
        image_base64 = base64.b64encode(image_file.read()).decode('utf-8')

        response = client.chat.completions.create(
            model="default",
            messages=[
                {
                    "role": "user", 
                    "content": [
                            {
                                 "type": "text", 
                                 #"text": "This images contains a human, your task is to tell me if the person is a male or female, make sure to #answer with only the letter M for male and F for female "
                                  "text": "This images contains a human, your task is to tell me what this person is doing and try to identify who they are "
                            },
                            {
                                "type": "image_url",
                                "image_url": { "url": f"data:image/png;base64,{image_base64}"} 
                            }
                    ]
                }
            ]
        )

        return response.choices[0].message.content.strip()
    return pd.Series([classify_one_image(img) for img in images])

# COMMAND ----------

df_inference = df_img.repartition(4).withColumn("vLLM_predict", classify_img("content"))

# COMMAND ----------

display(df_inference)

# COMMAND ----------

from mlflow_extensions.testing.helper import kill_processes_containing

kill_processes_containing("vllm")
kill_processes_containing("ray")
kill_processes_containing("from multiprocessing")