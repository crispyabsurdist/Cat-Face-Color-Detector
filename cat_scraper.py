#!/usr/bin/env python3

import os
import requests
import time
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()
API_KEY = os.getenv("PIXABAY_API_KEY")
URL_ENDPOINT = os.getenv("URL_ENDPOINT")
PER_PAGE = 200

# Change this to whatever you want to search for and change it after each run.
query = "cat"

PARAMS = {"q": query, "per_page": PER_PAGE, "page": 1}
ENDPOINT = URL_ENDPOINT + "?key=" + API_KEY

url_links = []

req = requests.get(url=ENDPOINT, params=PARAMS)
data = req.json()

num_pages = (data["totalHits"] // PER_PAGE) + 1

for image in data["hits"]:
    url_links.append(image["webformatURL"])

for page in range(2, num_pages + 1):
    time.sleep(3)
    PARAMS["page"] = page
    req = requests.get(url=ENDPOINT, params=PARAMS)
    data = req.json()
    for image in data["hits"]:
        url_links.append(image["webformatURL"])

index = 0
for image in tqdm(url_links):
    index += 1
    r = requests.get(image, allow_redirects=False)
    # change filename to whatever you are quering e.g black_cat_ or white_cat_
    file_name = "cat_" + str(index)
    script_dir = os.path.dirname(__file__)
    # change the path to the folder it should be saved to
    rel_path = "./dataset/validation/White/" + file_name + ".jpg"
    abs_file_path = os.path.join(script_dir, rel_path)
    open(abs_file_path, "wb").write(r.content)
