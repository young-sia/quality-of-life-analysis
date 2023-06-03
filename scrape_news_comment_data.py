# step1. 관련 패키지 및 모듈 불러오기
from selenium import webdriver
import time
import pandas as pd
from selenium.webdriver.common.by import By
from sentimental_analyze import *
from analyze_data import *

def get_comment_urls(original_urls):
    modified_urls = []
    # count = 0
    # print(len(original_urls))

    for url in original_urls:
        # print(count)
        segments = url.split("/")
        segments.insert(5, "comment")
        modified_url = "/".join(segments)
        modified_urls.append(modified_url)
        # count += 1

    return modified_urls


# step2. 네이버 뉴스 댓글정보 수집 함수
def get_naver_news_comments(url, wait_time=5, delay_time=0.1):
    driver = webdriver.Chrome('./chromedriver')
    # (크롬)드라이버가 요소를 찾는데에 최대 wait_time 초까지 기다림 (함수 사용 시 설정 가능하며 기본값은 5초)
    driver.implicitly_wait(wait_time)
    # 인자로 입력받은 url 주소를 가져와서 접속
    driver.get(url)

    # 더보기가 안뜰 때 까지 계속 클릭 (모든 댓글의 html을 얻기 위함)
    while True:

        # 예외처리 구문 - 더보기 광클하다가 없어서 에러 뜨면 while문을 나감(break)
        try:
            more = driver.find_elements(By.CSS_SELECTOR, 'a.u_cbox_btn_more')
            more.click()
            time.sleep(delay_time)
        except:
            break

    # # 3)댓글 내용
    # # selenium으로 댓글내용 포함된 태그 모두 수집
    contents = driver.find_elements(By.CSS_SELECTOR, 'span.u_cbox_contents')
    # 리스트에 텍스트만 담기 (리스트 컴프리핸션 문법)
    list_contents = [content.text for content in contents]

    # 드라이버 종료
    driver.quit()

    # 함수를 종료하며 list_sum을 결과물로 제출
    return list_contents


def after_scraping(data=pd.read_csv('news_comment.csv')):
    data = data.rename(columns = {'document':'content'})
    wordcloud(data)


def main():
    # news_data = pd.read_csv('naver_news_final.csv')
    # news_data = news_data.dropna(axis = 0)
    # original_urls = news_data['link']
    # # original_urls = [
    # #     "https://n.news.naver.com/mnews/article/002/0002286908?sid=102",
    # #     "https://n.news.naver.com/mnews/article/243/0000045510?sid=101"
    # # ]
    # # 원하는 기사 url 입력
    # driver = webdriver.Chrome('./chromedriver')
    # comment_urls = get_comment_urls(original_urls)
    #
    # # 함수 실행
    # comments = list()
    # for url in comment_urls:
    #     comment = get_naver_news_comments(url)
    #     comments.extend(comment)
    #
    # print(comments)
    #
    # comments_file = {'document': comments}
    # comments_file = pd.DataFrame(comments_file)
    # comments_file.to_csv('news_comment.csv',  index = False, encoding = "utf-8-sig")

    # fixed_comments = removing_non_korean(comments_file)
    # print(fixed_comments)
    # for comment in comments:
    #     sentimental_analysis(comment)
    after_scraping()


if __name__ == '__main__':
    main()
