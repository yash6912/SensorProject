from pymongo.mongo_client import MongoClient
import pandas as pd
import json

# url
uri = "mongodb+srv://yash12:zeGZ34P8nLwEfpEs@cluster0.ucxauyl.mongodb.net/?appName=Cluster0"


# Create a new client and connect to the server
client = MongoClient(uri)

# Create a database name and collection name
Database_name = "Pwskills"
Collection_name = "waferfault"

df = pd.read_csv("/content/wafer_23012020_041211.csv")

df = df.drop(columns=["Unnamed: 0"], axis=1)

json_record = list(json.loads(df.T.to_json()).values())

client[Database_name][Collection_name].insert_many(json_record)