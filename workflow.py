import sys

import pandas as pd

from analyze_data import *
from sentimental_analyze import *
from scrape_news_data import *
from scrape_blog_data import *
from scrape_news_comment_data import *


def scrape_news_data(keywords):
    start_number = [1, 101, 201]
    retrieved_data = {'items': []}
    for keyword in keywords:
        for start in start_number:
            news_data = retrieve_search_news_data(start, keyword)
            retrieved_data['items'].extend(news_data['items'])

    naver_news, other_news = distinguish_naver_other_news(retrieved_data)
    other_news = pd.DataFrame(other_news)
    other_news = other_news.drop_duplicates(subset = None, keep = 'first', inplace = False, ignore_index = False)
    other_news.to_csv('other_news_final.csv', index = False, encoding = "utf-8-sig")

    driver = webdriver.Chrome(ChromeDriverManager().install())
    driver.implicitly_wait(3)
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/98.0.4758.102"}
    naver_news_all = get_content_data_from_naver(headers, naver_news)
    naver_news_all.to_csv('naver_news_final.csv', index = False, encoding = "utf-8-sig")

    print('scraped news data')


def scrape_blog_data(keywords):
    start_number = [1, 101, 201]
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


def scrape_comment_data(news_data):
    news_data = news_data.dropna(axis = 0)
    original_urls = news_data['link']

    # 원하는 기사 url 입력
    driver = webdriver.Chrome('./chromedriver')
    comment_urls = get_comment_urls(original_urls)

    # 함수 실행
    comments = list()
    for url in comment_urls:
        comment = get_naver_news_comments(url)
        comments.extend(comment)

    # print(comments)

    comments_file = {'document': comments}
    comments_file = pd.DataFrame(comments_file)
    comments_file.to_csv('news_comment.csv',  index = False, encoding = "utf-8-sig")

    fixed_comments = removing_non_korean(comments_file)
    print(fixed_comments)
    for comment in comments:
        sentimental_analysis(comment)


def summarize_news_data(naver_data):
    summarizations = list()
    for each_news in naver_data['content']:
        summarize = summarize_content(each_news)
        summarizations.append(summarize)
    print('finished summarization of each article')
    naver_data['summary'] = summarizations
    naver_data.to_csv('naver_news_final_with_summaries.csv', index = False, encoding = "utf-8-sig")
    summaries_of_each_news = ''.join(summarizations)
    total_summary = summarize_content(summaries_of_each_news)

    for time in range(1, 10):
        print(f'추가요약 {time}번째')
        total_summary = summarize_content(total_summary)

    print(total_summary)


def summarize_comment_data(naver_data):
    summaries_of_each_news = ''.join(naver_data['document'])
    summary = summarize_content(summaries_of_each_news)

    sys.stdout = open('summary_of_comment.txt', 'w', encoding = "utf-8-sig")
    print(summary)
    sys.stdout.close()


def sentimental_analysis_in_blog(blog_data):
    blog_data = blog_data.dropna(axis = 0)

    print('블로그 갯수:', len(blog_data['content']))
    sentimental_result = []
    count = 0
    exception_count = 0

    for content in blog_data['content']:
        print('start of analyzing')
        number_of_sentences = 0
        sum_of_sentiment = 0
        sentences = content.split('.')
        for sent in sentences:
            print('loading....')
            # print(sent)
            try:
                sum_of_sentiment += sentimental_analysis(sent)
                number_of_sentences += 1
            except:
                exception_count += 1
                pass
        count += 1
        if number_of_sentences == 0:
            print('존재하지 않습니다.')
            continue
        else:
            average_of_sentement = sum_of_sentiment / number_of_sentences
            sentimental_result.append(average_of_sentement)
            print(sentimental_result)
            print(count, '번째 분석 완료')
    print(exception_count, '개의 예외 문장 발생')
    blog_data['feelings'] = sentimental_result
    blog_data.to_csv('naver_blog_with_sentiment.csv', index = False, encoding = "utf-8-sig")


def sentimental_analysis_in_comment(comment_data):
    data = list()
    for content in comment_data['document']:
        content = content.replace('\n', '').replace('[', '').replace(']', '').replace('"', '').replace('~', '').\
            replace('/', '').replace('?', '').replace('.', '').replace('#', '').replace(',', '').replace('ㅋ', '').\
            replace('@', '').replace('ㅎ', '').replace('ㅡ', '').replace('-', '').replace('18', '').replace('ㅉ','')
        data.append(content)
    print('전처리 작업 완료')

    comment_data['fixed_comments'] = data
    comment_data = comment_data.dropna(axis = 0)
    print('뉴스댓글 총 갯수:', len(comment_data['fixed_comments']))
    sentimental_result = []
    count = 0
    exception_count = 0

    for content in comment_data['fixed_comments']:
        print('start of analyzing')
        number_of_sentences = 0
        sum_of_sentiment = 0
        sentences = content.split('.')
        for sent in sentences:
            print('loading....')
            # print(sent)
            try:
                sum_of_sentiment += sentimental_analysis(sent)
                number_of_sentences += 1
            except:
                exception_count += 1
                pass

        average_of_sentement = sum_of_sentiment / number_of_sentences
        sentimental_result.append(average_of_sentement)

        count += 1
        print(sentimental_result)
        print(count, '번째 분석 완료')
    print(exception_count, '개의 예외 문장 발생')
    comment_data['feelings'] = sentimental_result
    comment_data.to_csv('naver_comment_with_sentiment.csv', index = False, encoding = "utf-8-sig")


def main():
    blog_data = pd.read_csv('naver_blogs_final.csv')
    sentimental_analysis_in_blog(blog_data)
    print('69시간제의 감성분석 완료')
    # topic = int(input('오늘의 주제는 무엇인가요?(0: 전체, 1: 주 69시간제, 2: 주 4.5일제)'))
    # if topic == 0:
    #     key = ['주 69시간', '주 60시간', '4.5일제', '주 4일제']
    # elif topic == 1:
    #     key = ['주 69시간', '주 60시간']
    # elif topic == 2: h
    #     key = ['4.5일제', '주 4일제']
    # else:
    #     print('오류입니다. 종료합니다.')
    #     exit()
    # answer1 = int(input('수집 단계부터 진행하실건가요?(yes: 1, no: 0):'))
    # while True:
    #     if answer1 == 1:
    #         answer2 = int(input('어떤 수집을 진행하실 건가요?(뉴스면 1, 블로그면 2, 댓글이면 3)'))
    #         if answer2 == 1:
    #             keyword = '뉴스'
    #             scrape_news_data(key)
    #         elif answer2 == 2:
    #             keyword = '블로그'
    #             scrape_blog_data(key)
    #         elif answer2 == 3:
    #             try:
    #                 news_data = pd.read_csv('naver_news_final.csv')
    #                 scrape_comment_data(news_data)
    #                 keyword = '댓글'
    #             except:
    #                 answer3 = int(input('댓글 수집을 위한 뉴스 데이터가 존재하지 않습니다. 뉴스데이터부터 수집하시겠습니까?(no: 0, yes:1)'))
    #                 if answer3 == 1:
    #                     scrape_news_data(key)
    #                     news_data = pd.read_csv('naver_news_final.csv')
    #                     scrape_comment_data(news_data)
    #                     keyword = '뉴스 & 댓글'
    #                 else:
    #                     exit()
    #         else:
    #             print('잘못 입력하셨습니다. 종료하겠습니다')
    #             exit()
    #         answer4 = int(input(f'{key}주제의 {keyword}수집을 완료하였습니다. 분석 단계를 진행하시겠습니까?(no: 0 yes: 1)'))
    #         if answer4 == 1:
    #             answer1 = 0
    #             continue
    #         else:
    #             print('종료하겠습니다.')
    #             exit()
    #
    #     elif answer1 == 0:
    #         answer5 = int(input('어떤 자료를 분석하시겠습니까?(뉴스: 1, 블로그: 2, 댓글: 3)'))
    #         if answer5 == 1:
    #             answer6 = int(input('뉴스 - 어떤 분석을 하시겠습니까?(wordcloud: 0, 요약: 1)'))
    #             if answer6 == 0:
    #                 try:
    #                     news_data = pd.read_csv('naver_news_final.csv')
    #                     wordcloud(news_data)
    #                     keyword = '뉴스의 워드클라우드'
    #                 except:
    #                     print('분석을 위한 뉴스 자료는 존재하지 않습니다. 종료합니다.')
    #                     exit()
    #             elif answer6 == 1:
    #                 try:
    #                     news_data = pd.read_csv('naver_news_final.csv')
    #                     summarize_news_data(news_data)
    #                     keyword = '뉴스의 요약'
    #                 except:
    #                     print('분석을 위한 뉴스 자료는 존재하지 않습니다. 종료합니다.')
    #                     exit()
    #             else:
    #                 print('오류입니다. 전단계로 돌아갑니다.')
    #                 continue
    #         elif answer5 == 2:
    #             answer6 = int(input('블로그 - 어떤 분석을 하시겠습니까?(wordcloud: 0, 감성분석: 1)'))
    #             if answer6 == 0:
    #                 try:
    #                     blog_data = pd.read_csv('naver_blogs_final.csv')
    #                     wordcloud(blog_data)
    #                     keyword = '블로그의 워드클라우드'
    #                 except:
    #                     print('분석을 위한 블로그 자료는 존재하지 않습니다. 종료합니다.')
    #                     exit()
    #             elif answer6 == 1:
    #                 try:
    #                     blog_data = pd.read_csv('naver_blogs_final.csv')
    #                     sentimental_analysis_in_blog(blog_data)
    #                     keyword = '블로그의 감성분석'
    #                 except:
    #                     print('분석을 위한 블로그 자료는 존재하지 않습니다. 종료합니다.')
    #                     exit()
    #             else:
    #                 print('오류입니다. 전단계로 돌아갑니다.')
    #                 continue
    #         elif answer5 == 3:
    #             answer6 = int(input('댓글- 어떤 분석을 하시겠습니까?(wordcloud: 0, 요약: 1, 감성분석: 2)'))
    #             if answer6 == 0:
    #                 try:
    #                     comment_data = pd.read_csv('news_comment.csv')
    #                     wordcloud(comment_data)
    #                     keyword = '댓글의 워드클라우드'
    #                 except:
    #                     print('분석을 위한 댓글 자료는 존재하지 않습니다. 종료합니다.')
    #                     exit()
    #             elif answer6 == 1:
    #                 try:
    #                     comment_data = pd.read_csv('news_comment.csv')
    #                     summarize_comment_data(comment_data)
    #                     keyword = '댓글의 요약'
    #                 except:
    #                     print('분석을 위한 댓글 자료는 존재하지 않습니다. 종료합니다.')
    #                     exit()
    #             elif answer6 == 2:
    #                 try:
    #                     comment_data = pd.read_csv('news_comment.csv')
    #                     sentimental_analysis_in_comment(comment_data)
    #                     keyword = '댓글의 감성분석'
    #                 except:
    #                     print('분석을 위한 댓글 자료는 존재하지 않습니다. 종료합니다.')
    #                     exit()
    #             else:
    #                 print('오류입니다. 전단계로 돌아갑니다.')
    #                 continue
    #         else:
    #             print('잘못 입력하셨습니다. 종료하겠습니다')
    #             exit()
    #         answer4 = int(input(f'{key}주제의 {keyword}을/를 완료하였습니다. 어느 단계로 가시겠습니까?(종료: 0 수집: 1 분석: 2)'))
    #         if answer4 == 1:
    #             answer1 = 1
    #             continue
    #         elif answer4 == 2:
    #             answer1 = 0
    #             continue
    #         else:
    #             print('종료하겠습니다.')
    #             exit()
    #     else:
    #         print('잘못 입력하셨습니다. 종료합니다.')
    #         exit()


if __name__ == "__main__":
    main()
