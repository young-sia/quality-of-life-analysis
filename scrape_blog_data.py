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
import re
from bs4 import BeautifulSoup
import requests

import logging

load_dotenv()
client_id = os.environ.get('NAVER_ID')
client_secret = os.environ.get("NAVER_PASSWORD")


def retrieve_search_blog_data(start, keyword):

    enc_text = urllib.parse.quote(keyword)
    logging.info(f'about to search blog data with keyword {keyword}')
    display = 100

    url = "https://openapi.naver.com/v1/search/blog?query=" + enc_text +\
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

    pattern1 = '<[^>]*>'
    pattern2 = """[\n\n\n\n\n// flash 오류를 우회하기 위한 함수 추가\nfunction _flash_removeCallback() {}"""
    count = 0
    for link in urls:
        driver.get(link)
        time.sleep(1)
        # 블로그 안 본문이 있는 iframe에 접근하기
        driver.switch_to.frame("mainFrame")
        # 본문 내용 크롤링하기
        # 본문 내용 크롤링하기
        try:
            a = driver.find_element(By.CSS_SELECTOR, 'div.se-main-container').text
            content = re.sub(pattern = pattern1, repl = '', string = a)
            content = content.replace(pattern2, '')
            content = content.strip()
            contents['content'].append(content)
            count += 1
            print(count, '번째 기사글')
            print(content)
            # print('작동')
        # NoSuchElement 오류시 예외처리(구버전 블로그에 적용)
        except NoSuchElementException:
            a = driver.find_element(By.CSS_SELECTOR, 'div#content-area').text
            content = re.sub(pattern = pattern1, repl = '', string = a)
            content = content.replace(pattern2, '')
            content = content.strip()
            contents['content'].append(content)
            count += 1
            print(count, '번째 기사글')
            print(content)
            # print('작동안함')

    contents = pd.DataFrame(contents)
    naver_blog_all = pd.concat([naver_blog_all, contents], axis = 1)
    return naver_blog_all


def main():
    start_number = [1, 101, 201]
    keywords = ['주 69시간', '주 60시간']
    # keywords = [, '4.5일제', '주 4일제']
    retrieved_data = {'items': []}
    for keyword in keywords:
        for start in start_number:
            news_data = retrieve_search_blog_data(start, keyword)
            retrieved_data['items'].extend(news_data['items'])
    naver_blogs, other_blogs = distinguish_naver_other_blog(retrieved_data)
    naver_blogs_dataframe = pd.DataFrame(naver_blogs)
    # print(naver_blogs)
    # print(else_blogs)

    other_blogs = pd.DataFrame(other_blogs)
    other_blogs.to_csv('other_blogs_final.csv', index = False, encoding = "utf-8-sig")
    naver_blogs_dataframe.to_csv('naver_blogs_without_content.csv', index = False, encoding = "utf-8-sig" )

    naver_blogs = pd.read_csv('naver_blogs_without_content.csv')

    options = webdriver.ChromeOptions()
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option("useAutomationExtension", False)

    driver = webdriver.Chrome(ChromeDriverManager().install())
    driver.implicitly_wait(3)
    naver_blog_all = get_content_from_naver(driver, naver_blogs)

    naver_blog_all.to_csv('naver_blogs_final.csv', index = False, encoding = "utf-8-sig")


if __name__ == "__main__":
    main()



