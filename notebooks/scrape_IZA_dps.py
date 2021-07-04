import requests
import json
import time
from bs4 import BeautifulSoup
from pathlib import Path
from datetime import datetime

prefix = "https://www.iza.org/publications/dp/"


def get_last_dp_number(frontpage, link_prefix=prefix) -> int:
    """
    Returns the number of the last IZA discussion paper found in the frontpage
    """
    page = requests.get(frontpage)
    soup = BeautifulSoup(page.content, "html.parser")

    links = [
        s.find_all("a", href=True)[0]["href"]
        for s in soup.find_all("div", class_="title")
        if s.find("a") is not None
    ]
    last_dp_n = max(
        [
            int(link.split(link_prefix)[1].split("/")[0])
            for link in links
            if link.startswith(link_prefix)
        ]
    )

    return last_dp_n


def get_dp_data(dp_number, prefix=prefix, max_retries=1) -> dict:
    """
    For a specific discussion paper number, returns a dictionary with title,
    abstract, and publication date of the discussion paper.

    Parameters:
    ----

    dp_number: `int`, number of the discussion paper

    prefix: `str`, prefix of URL string

    max_retries: `int`, number of time to retry scraping if not successful
    """
    dppage = requests.get(prefix + f"/{dp_number}")
    if dppage.status_code != 200:
        cnt = 0
        while cnt < max_retries and dppage.status_code != 200:
            dppage = requests.get(prefix + f"/{dp_number}")
            cnt += 1
    if dppage.status_code == 200:
        soup = BeautifulSoup(dppage.content, "html.parser")

        date = soup.find("div", class_="col-md-11").find("p").contents[0].strip()

        title = ":".join(
            soup.find("h2", class_="title")
            .contents[0]
            .replace("\n", "")
            .strip()
            .split(":")[1:]
        ).strip()

        abstract = (
            soup.find("div", class_="element-copyexpandable").find("p").contents[0]
        )

        keywords = ";".join(
            [
                link.contents[0].replace("\n", "").strip()
                for link in [
                    div
                    for div in soup.find_all("h3")
                    if div.contents[0].startswith("Keywords")
                ][0]
                .find_next_sibling()
                .find_all("a", href=True)
            ]
        )
        return {
            dp_number: {
                "date": date,
                "title": title,
                "abstract": abstract,
                "keywords": keywords,
            }
        }
    else:
        print(f"DP {dp_number} not found")
        return None


savefile = Path('../data/IZA_dps.json')

if __name__ == "__main__":
    start = datetime.now()
    last_dp_number = get_last_dp_number(prefix)
    print(f'Last DP number: {last_dp_number}. Starting scraping...')

    datadict = {}
    dp_number = 1
    while dp_number <= last_dp_number:
        if dp_number % 1000 == 0:
            print(f'Scraped {dp_number} out of {last_dp_number} in {str(datetime.now() - start)}')
        try:
            dpdata = get_dp_data(dp_number)
        except requests.exceptions.ConnectionError:
            print(f'Max retries exceeded at DP {dp_number}: Sleeping for 10 minutes...')
            time.sleep(5 * 60)
            print('Restarted scraping')

        if dpdata is not None:
            datadict.update(dpdata)
        dp_number += 1

    end = datetime.now()
    print(f'Scraping completed in {str(end-start)}. Saving in {savefile}')
    jdp = json.dumps(datadict)
    with savefile.open('w') as f:
        f.write(jdp)
