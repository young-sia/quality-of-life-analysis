import urllib.request
import json
import os
from dotenv import load_dotenv  # 이 패키지는 host가 따로 없을 때에 사용한다.

import pandas as pd
import urllib.request
from selenium.common.exceptions import NoSuchElementException
from selenium import webdriver
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
import time
from bs4 import BeautifulSoup
import requests

import logging

load_dotenv()
client_id = os.environ.get('NAVER_ID')
client_secret = os.environ.get("NAVER_PASSWORD")


def retrieve_search_blog_data():
    keyword = '69시간'
    enc_text = urllib.parse.quote(keyword)
    logging.info(f'about to search blog data with keyword {keyword}')

    url = "https://openapi.naver.com/v1/search/blog?query=" + enc_text  # JSON 결과

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


def distinguish_naver_other_blog(blog_data):
    naver_blog_data = {'title': [], 'link': []}
    else_blog_data = {'title': [], 'link': []}
    for data in blog_data['items']:
        if 'blog.naver.com/' in data['bloggerlink']:
            naver_blog_data['title'].append(data['title'])
            naver_blog_data['link'].append(data['link'])
        else:
            else_blog_data['title'].append(data['title'])
            else_blog_data['link'].append(data['link'])
    return naver_blog_data, else_blog_data


def get_content_from_naver(driver, blog_data):
    naver_blog_all = pd.DataFrame(blog_data)
    contents = {'content': []}
    urls = blog_data['link']
    for link in urls:
        driver.get(link)
        time.sleep(1)
        # 블로그 안 본문이 있는 iframe에 접근하기
        driver.switch_to.frame("mainFrame")
        # 본문 내용 크롤링하기
        # 본문 내용 크롤링하기
        try:
            a = driver.find_element(By.CSS_SELECTOR, 'div.se-main-container').text
            contents['content'].append(a)
            # print('작동')
        # NoSuchElement 오류시 예외처리(구버전 블로그에 적용)
        except NoSuchElementException:
            a = driver.find_element(By.CSS_SELECTOR, 'div#content-area').text
            contents['content'].append(a)
            # print('작동안함')

    contents = pd.DataFrame(contents)
    naver_blog_all = pd.concat([naver_blog_all, contents], axis = 1)
    return naver_blog_all


def sentimental_analysis():
    pass


def main():
    blog_data = retrieve_search_blog_data()
    naver_blogs, other_blogs = distinguish_naver_other_blog(blog_data)
    # print(naver_blogs)
    # print(else_blogs)

    other_blogs = pd.DataFrame(other_blogs)
    other_blogs.to_csv('other_blogs.csv', index = False, encoding = "utf-8-sig")

    options = webdriver.ChromeOptions()
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option("useAutomationExtension", False)

    driver = webdriver.Chrome(ChromeDriverManager().install())
    driver.implicitly_wait(3)
    naver_blog_all = get_content_from_naver(driver, naver_blogs)

    naver_blog_all.to_csv('naver_blogs.csv', index = False, encoding = "utf-8-sig")


if __name__ == "__main__":
    main()



