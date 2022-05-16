import urllib3
from bs4 import BeautifulSoup
from datetime import datetime
import json
from pathlib import Path
import os
import ssl


xmls = [
    "https://www.nber.org/wwpinfo/googleXML_h.xml",
    "https://www.nber.org/wwpinfo/googleXML_t.xml",
    "https://www.nber.org/wwpinfo/googleXML_w.xml",
]

month_dict = {
    1: "January",
    2: "February",
    3: "March",
    4: "April",
    5: "May",
    6: "June",
    7: "July",
    8: "February",
    9: "February",
    10: "February",
    11: "February",
    12: "February",
}


def get_article_data(article):
    """
    Extracts information from an `article` in the NBER xml file and exports it
    as a dictionary
    """
    article_data = {
        article.find("article-id").contents[0]: {
            "date": get_date(article),
            "title": get_title(article),
            "abstract": get_abstract(article),
        }
    }
    return article_data


def get_date(article):
    try:
        date = "".join(
            [
                month_dict[int(article.find("month").contents[0])],
                " ",
                article.find("year").contents[0],
            ]
        )
        return str(date)
    except AttributeError:
        return None


def get_title(article):
    try:
        title = article.find("article-title").contents[0]
        return str(title)
    except AttributeError:
        return None


def get_abstract(article):
    try:
        abstract = article.find("abstract").find("p").contents[0]
        return str(abstract)
    except AttributeError:
        return None


def download_all_data(link):
    """
    Donwloads and processes a NBER xml fine, returning a dictionary with all
    necessary metadata from the articles extracted by `get_article_data()`
    """
    start = datetime.now()
    http = urllib3.PoolManager()
    page = http.request("GET", link)
    xml = BeautifulSoup(page.data, "html.parser")
    print(f"{link} downloaded in {str(datetime.now() - start)}. Extracting data...")
    data = {}

    article = xml.find("article")
    while article is not None:
        article_data = get_article_data(article)
        data.update(article_data)
        article = article.find_next_sibling()

    print(f"{link} processed in {str(datetime.now() - start)}")
    return data


savefile = Path("data/NBER_wps.json")

if __name__ == "__main__":
    data = {}
    for link in xmls:
        # To bypass certificate error
        if (not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context', None)):
            ssl._create_default_https_context = ssl._create_unverified_context

        # Download and get data
        temp = download_all_data(link)
        data.update(temp)
        jdp = json.dumps(data)
    with savefile.open("w") as f:
        f.write(jdp)
