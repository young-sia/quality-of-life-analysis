import urllib.request
import json
import os
from dotenv import load_dotenv  # 이 패키지는 host가 따로 없을 때에 사용한다.

from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
import pandas as pd
from bs4 import BeautifulSoup
import requests
import re

from datetime import datetime

import logging

load_dotenv()
client_id = os.environ.get('NAVER_ID')
client_secret = os.environ.get("NAVER_PASSWORD")


def retrieve_search_news_data(start, keyword):
    enc_text = urllib.parse.quote(keyword)
    logging.info(f'about to search blog data with keyword {keyword}')
    display = 100

    url = "https://openapi.naver.com/v1/search/news?query="+ enc_text+\
          '&display='+str(display) + '&start='+str(start)+'&sort=sim' # JSON 결과

    request = urllib.request.Request(url)
    request.add_header("X-Naver-Client-Id", client_id)
    request.add_header("X-Naver-Client-Secret", client_secret)
    response = urllib.request.urlopen(request)
    rescode = response.getcode()
    if rescode == 200:
        response_body = response.read()
        logging.info(response_body.decode('utf-8'))
        response_json = json.loads(response_body)
        return response_json
    else:
        logging.info("Error Code:" + rescode)


def distinguish_naver_other_news(urls):
    naver_news_data = {'title': [], 'link': []}
    else_news_data = {'title': [], 'link': []}
    for data in urls['items']:
        if 'n.news.naver.com/' in data['link']:
            naver_news_data['title'].append(data['title'])
            naver_news_data['link'].append(data['link'])
        else:
            else_news_data['title'].append(data['title'])
            else_news_data['link'].append(data['link'])
    return naver_news_data, else_news_data


def get_content_data_from_naver(headers, naver_news):
    naver_news_all = pd.DataFrame(naver_news)
    naver_news_all = naver_news_all.drop_duplicates(subset = None, keep = 'first', inplace = False, ignore_index = False)
    contents = {'content': []}
    urls = naver_news['link']
    pattern1 = '<[^>]*>'
    pattern2 = """[\n\n\n\n\n// flash 오류를 우회하기 위한 함수 추가\nfunction _flash_removeCallback() {}"""
    for link in urls:
        original_html = requests.get(link, headers = headers)
        html = BeautifulSoup(original_html.text, "html.parser")
        content = html.select("#newsct_article")
        content = ''.join(str(content))

        content = re.sub(pattern = pattern1, repl = '', string = content)
        content = content.replace(pattern2, '')
        content = content.strip()
        contents['content'].append(content)

    print('finished scraping all contents')

    contents = pd.DataFrame(contents)
    naver_news_all = pd.concat([naver_news_all, contents], axis = 1)

    return naver_news_all


def main():
    start_number = [1, 101, 201]
    keywords = ['주 69시간', '주 60시간', '4.5일제', '주 4일제']
    retrieved_data = {'items': []}
    for keyword in keywords:
        for start in start_number:
            news_data = retrieve_search_news_data(start, keyword)
            retrieved_data['items'].extend(news_data['items'])

    naver_news, other_news = distinguish_naver_other_news(retrieved_data)
    other_news = pd.DataFrame(other_news)
    other_news = other_news.drop_duplicates(subset = None, keep = 'first', inplace = False, ignore_index = False)
    other_news.to_csv('other_news_final.csv', index=False, encoding="utf-8-sig")

    driver = webdriver.Chrome(ChromeDriverManager().install())
    driver.implicitly_wait(3)
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/98.0.4758.102"}
    naver_news_all = get_content_data_from_naver(headers, naver_news)
    naver_news_all.to_csv('naver_news_final.csv', index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
