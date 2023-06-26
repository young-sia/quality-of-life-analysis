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

    file = open('summary_of_comment.txt', 'w', encoding = "utf-8-sig")
    print(summary)
    file.close()


def sentimental_analysis_in_blog(blog_data):
    blog_data = blog_data.dropna(axis = 0)

    print('블로그 갯수:', len(blog_data['content']))
    sentimental_result = []
    count = 0
    exception_count = 0

    for content in blog_data['content']:
        # print('start of analyzing')
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
            average_of_sentement = 10
            sentimental_result.append(average_of_sentement)
            continue
        else:
            average_of_sentement = sum_of_sentiment / number_of_sentences
            sentimental_result.append(average_of_sentement)
            print(sentimental_result)
            print(count, '번째 분석 완료')
    print(exception_count, '개의 예외 문장 발생')
    blog_data['feelings'] = sentimental_result

    blog_data = blog_data.drop(blog_data.columns[blog_data['feelings'] == 10], axis = 1)
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

        if number_of_sentences == 0:
            print('존재하지 않습니다.')
            average_of_sentement = 10
            sentimental_result.append(average_of_sentement)
            continue
        else:
            average_of_sentement = sum_of_sentiment / number_of_sentences
            sentimental_result.append(average_of_sentement)
            print(sentimental_result)
            print(count, '번째 분석 완료')

        count += 1
        print(sentimental_result)
        print(count, '번째 분석 완료')
    print(exception_count, '개의 예외 문장 발생')
    comment_data['feelings'] = sentimental_result

    comment_data = comment_data[comment_data.feelings != 10]
    comment_data.to_csv('naver_comment_with_sentiment.csv', index = False, encoding = "utf-8-sig")


def main():
    blog_data = pd.read_csv('naver_blogs_final.csv')
    wordcloud(blog_data)
    # sent_score = [0.5423728813559322, 0.625, 0.46808510638297873, 0.45454545454545453, 0.5, 0.5873015873015873, 0.49019607843137253, 0.5365853658536586, 0.5576923076923077, 0.54, 0.43243243243243246, 0.6, 0.6666666666666666, 0.5, 0.45, 0.3870967741935484, 0.40540540540540543, 0.5217391304347826, 0.42105263157894735, 0.5294117647058824, 0.5, 0.532258064516129, 0.4090909090909091, 0.5576923076923077, 0.35294117647058826, 0.5070422535211268, 0.5454545454545454, 0.47368421052631576, 0.4810126582278481, 0.5416666666666666, 0.575, 0.5483870967741935, 0.6097560975609756, 0.6875, 0.4897959183673469, 0.58, 0.6176470588235294, 0.6, 0.5454545454545454, 0.5862068965517241, 0.5, 0.5, 0.45454545454545453, 0.2608695652173913, 0.5882352941176471, 0.47368421052631576, 0.7058823529411765, 0.7777777777777778, 0.43478260869565216, 0.6153846153846154, 0.4857142857142857, 0.5416666666666666, 0.5277777777777778, 0.5384615384615384, 0.55, 0.7058823529411765, 0.4090909090909091, 0.6, 0.7, 0.23076923076923078, 0.4666666666666667, 0.35294117647058826, 0.46153846153846156, 0.39285714285714285, 0.4074074074074074, 0.55, 0.4166666666666667, 0.45454545454545453, 0.0, 0.6666666666666666, 0.4782608695652174, 0.5, 0.48598130841121495, 0.2, 0.45714285714285713, 0.5789473684210527, 0.0, 0.5268817204301075, 0.5, 0.5671641791044776, 0.5, 0.6, 0.5853658536585366, 0.5, 0.5416666666666666, 0.5348837209302325, 0.5555555555555556, 0.4, 0.3333333333333333, 0.46017699115044247, 0.525, 0.5777777777777777, 0.4166666666666667, 0.4, 0.5555555555555556, 0.5294117647058824, 0.34615384615384615, 0.4745762711864407, 0.6296296296296297, 0.5, 0.6666666666666666, 0.4318181818181818, 0.6875, 0.5428571428571428, 0.38461538461538464, 0.5125, 0.6363636363636364, 0.5517241379310345, 0.42105263157894735, 0.38095238095238093, 0.5128205128205128, 0.6896551724137931, 0.5425531914893617, 0.40476190476190477, 0.41935483870967744, 0.48, 0.375, 0.6153846153846154, 0.391304347826087, 0.6956521739130435, 0.56, 0.5, 0.6111111111111112, 0.4583333333333333, 0.5, 0.5416666666666666, 0.4166666666666667, 0.6, 0.375, 0.2, 0.46153846153846156, 0.5227272727272727, 0.42105263157894735, 0.42857142857142855, 0.52, 0.5714285714285714, 0.6363636363636364, 0.4603174603174603, 0.6842105263157895, 0.5454545454545454, 0.5833333333333334, 0.5, 0.5689655172413793, 0.44285714285714284, 0.47058823529411764, 0.4225352112676056, 0.625, 0.56, 0.4444444444444444, 0.47058823529411764, 0.5217391304347826, 0.5333333333333333, 0.4117647058823529, 0.5166666666666667, 0.4, 0.6086956521739131, 0.5161290322580645, 0.3870967741935484, 0.75, 0.6538461538461539, 0.5945945945945946, 0.4, 0.5294117647058824, 0.42857142857142855, 0.55, 0.4864864864864865, 0.5416666666666666, 0.45454545454545453, 0.4772727272727273, 0.2857142857142857, 0.7083333333333334, 0.5333333333333333, 0.7333333333333333, 0.5609756097560976, 0.5333333333333333, 0.5483870967741935, 0.5, 0.47058823529411764, 0.5357142857142857, 0.39473684210526316, 0.5135135135135135, 0.36363636363636365, 0.5, 0.7058823529411765, 0.625, 0.625, 0.7142857142857143, 0.3076923076923077, 0.3333333333333333, 0.6046511627906976, 0.5833333333333334, 0.4, 0.5333333333333333, 0.375, 0.4583333333333333, 0.36363636363636365, 0.47619047619047616, 0.3333333333333333, 0.42857142857142855, 0.4, 0.7142857142857143, 0.6285714285714286, 0.7142857142857143, 0.36363636363636365, 0.26666666666666666, 0.5555555555555556, 0.543859649122807, 0.48936170212765956, 0.4772727272727273, 0.5227272727272727, 0.5, 0.5384615384615384, 0.6153846153846154, 0.5681818181818182, 0.5714285714285714, 0.625, 0.625, 0.5714285714285714, 0.5384615384615384, 0.3333333333333333, 0.31746031746031744, 0.4666666666666667, 0.5348837209302325, 0.37209302325581395, 0.56, 0.6153846153846154, 0.75, 0.25, 0.5, 0.5925925925925926, 0.3333333333333333, 0.5, 0.5384615384615384, 0.5, 0.45, 0.3076923076923077, 0.45714285714285713, 0.625, 0.5333333333333333, 0.5894736842105263, 0.7692307692307693, 0.4166666666666667, 0.47368421052631576, 0.5, 0.5833333333333334, 1.0, 0.4782608695652174, 0.375, 0.5542168674698795, 0.4, 0.5324675324675324, 1.0, 0.5483870967741935, 0.4105263157894737, 0.56, 0.4791666666666667, 0.4716981132075472, 0.5, 0.5714285714285714, 0.47058823529411764, 0.4358974358974359, 0.515625, 0.4375, 0.4444444444444444, 0.5121951219512195, 0.5555555555555556, 0.6153846153846154, 0.5306122448979592, 0.375, 0.6571428571428571, 0.59375, 0.36363636363636365, 0.6666666666666666, 0.5185185185185185, 0.5555555555555556, 0.5714285714285714, 0.5333333333333333, 0.5520833333333334, 0.5416666666666666, 0.45454545454545453, 0.5, 0.4375, 0.5070422535211268, 0.55, 0.6530612244897959, 0.391304347826087, 0.8, 0.5094339622641509, 0.5263157894736842, 0.47761194029850745, 0.4594594594594595, 0.475, 0.5294117647058824, 0.4642857142857143, 0.5454545454545454, 0.4523809523809524, 0.5348837209302325, 0.7307692307692307, 0.5631067961165048, 0.625, 0.5333333333333333, 0.45714285714285713, 0.6363636363636364, 0.4827586206896552, 0.5333333333333333, 0.5728155339805825, 0.5, 0.5, 0.4444444444444444, 0.47878787878787876, 0.5625, 0.5217391304347826, 0.5909090909090909, 0.4, 0.46153846153846156, 0.8, 0.35, 0.5333333333333333, 0.14285714285714285, 0.40425531914893614, 0.5454545454545454, 0.3684210526315789, 0.45454545454545453, 0.4, 0.6296296296296297, 0.5476190476190477, 0.23076923076923078, 0.3684210526315789, 0.5773195876288659, 0.42857142857142855, 0.42424242424242425, 0.5757575757575758, 0.4375, 0.3333333333333333, 0.6, 0.631578947368421, 0.7368421052631579, 0.5833333333333334, 0.5384615384615384, 0.45454545454545453, 0.2857142857142857, 0.6, 0.5333333333333333, 0.5789473684210527, 0.4444444444444444, 0.625, 0.40425531914893614, 0.6363636363636364, 0.6153846153846154, 0.8571428571428571, 0.5555555555555556, 0.5, 0.3793103448275862, 0.52, 0.47058823529411764, 0.3, 0.47058823529411764, 0.5454545454545454, 0.5, 0.631578947368421, 0.4117647058823529, 0.48148148148148145, 0.47368421052631576, 0.52, 0.2916666666666667, 0.4782608695652174, 0.5714285714285714, 0.5172413793103449, 0.5357142857142857, 0.5625, 0.5294117647058824, 0.7142857142857143, 0.47058823529411764, 0.5483870967741935, 0.5833333333333334, 0.5306122448979592, 0.4166666666666667, 0.6428571428571429, 0.5, 0.47058823529411764, 0.5, 0.4666666666666667, 0.4642857142857143, 0.391304347826087, 0.41935483870967744, 0.4657534246575342, 0.5833333333333334, 0.42857142857142855, 0.5324675324675324, 0.5, 0.19047619047619047, 0.7272727272727273, 0.62, 0.3333333333333333, 0.5882352941176471, 0.43434343434343436, 0.3333333333333333, 0.5625, 0.75, 0.3114754098360656, 0.4927536231884058, 0.42105263157894735, 0.45098039215686275, 0.5357142857142857, 0.47368421052631576, 0.6470588235294118, 0.5294117647058824, 0.4782608695652174, 0.5555555555555556, 0.4166666666666667, 0.5365853658536586, 0.55, 0.3333333333333333, 0.5352112676056338, 0.6666666666666666, 0.5319148936170213, 0.3333333333333333, 0.49056603773584906, 0.6666666666666666, 0.3684210526315789, 0.6666666666666666, 0.55, 0.5294117647058824, 0.55, 0.6666666666666666, 0.5555555555555556, 0.3888888888888889, 0.42857142857142855, 1.0, 0.44, 0.559322033898305, 0.3333333333333333, 0.5454545454545454, 0.5357142857142857, 0.4, 0.45652173913043476, 0.0, 0.43478260869565216, 0.5454545454545454, 0.5, 0.5714285714285714, 0.6, 0.5223880597014925, 0.4716981132075472, 1.0, 0.2972972972972973, 0.4666666666666667, 0.5, 0.6923076923076923, 0.3333333333333333, 0.5555555555555556, 0.36585365853658536, 0.35714285714285715, 0.55, 0.5769230769230769, 0.3333333333333333, 0.44871794871794873, 0.4, 0.55, 0.3684210526315789, 0.5, 0.45454545454545453, 0.5490196078431373, 0.3333333333333333, 0.3333333333333333, 0.5333333333333333, 0.5535714285714286, 0.6666666666666666, 0.5476190476190477, 0.47368421052631576, 0.4482758620689655, 0.43333333333333335, 0.3, 0.4444444444444444, 0.5862068965517241, 0.5, 1.0, 1.0, 0.2, 0.42857142857142855, 0.53125, 0.0, 0.4461538461538462, 0.43333333333333335, 0.4634146341463415, 0.5384615384615384, 0.5, 0.5384615384615384, 0.4, 0.47058823529411764, 0.47058823529411764, 0.34782608695652173, 0.4634146341463415, 0.47619047619047616, 0.5625, 0.6086956521739131, 0.16666666666666666, 0.5, 0.43478260869565216, 0.4528301886792453, 0.375, 0.5483870967741935, 0.7142857142857143, 0.5190839694656488, 0.6666666666666666, 0.4117647058823529, 0.5121951219512195, 0.4375, 0.46153846153846156, 0.5263157894736842, 0.421875, 0.4473684210526316, 0.6335877862595419, 0.42105263157894735, 0.4, 0.5352112676056338, 0.3944954128440367, 0.6206896551724138, 0.6136363636363636, 0.5714285714285714, 0.5789473684210527, 0.85, 0.16666666666666666, 0.5733333333333334, 0.8, 0.5063291139240507, 0.5, 0.5652173913043478, 0.5172413793103449, 0.4, 0.5909090909090909, 0.375, 0.6, 0.43478260869565216, 0.45454545454545453, 0.5609756097560976, 0.6538461538461539, 0.6086956521739131, 0.625, 0.3695652173913043, 0.5294117647058824, 0.65, 0.3, 0.6176470588235294, 0.3333333333333333, 0.5277777777777778, 0.5238095238095238, 1.0, 0.0, 0.5714285714285714, 0.35714285714285715, 0.8181818181818182, 0.5161290322580645, 0.4883720930232558, 0.4, 0.5, 0.42105263157894735, 0.7, 0.4807692307692308, 0.47058823529411764, 0.3404255319148936, 1.0]

    topic = int(input('오늘의 주제는 무엇인가요?(0: 전체, 1: 주 69시간제, 2: 주 4.5일제)'))
    if topic == 0:
        key = ['주 69시간', '주 60시간', '4.5일제', '주 4일제']
    elif topic == 1:
        key = ['주 69시간', '주 60시간']
    elif topic == 2:
        key = ['4.5일제', '주 4일제']
    else:
        print('오류입니다. 종료합니다.')
        exit()
    answer1 = int(input('수집 단계부터 진행하실건가요?(yes: 1, no: 0):'))
    while True:
        if answer1 == 1:
            answer2 = int(input('어떤 수집을 진행하실 건가요?(뉴스면 1, 블로그면 2, 댓글이면 3)'))
            if answer2 == 1:
                keyword = '뉴스'
                scrape_news_data(key)
            elif answer2 == 2:
                keyword = '블로그'
                scrape_blog_data(key)
            elif answer2 == 3:
                try:
                    news_data = pd.read_csv('naver_news_final.csv')
                    scrape_comment_data(news_data)
                    keyword = '댓글'
                except:
                    answer3 = int(input('댓글 수집을 위한 뉴스 데이터가 존재하지 않습니다. 뉴스데이터부터 수집하시겠습니까?(no: 0, yes:1)'))
                    if answer3 == 1:
                        scrape_news_data(key)
                        news_data = pd.read_csv('naver_news_final.csv')
                        scrape_comment_data(news_data)
                        keyword = '뉴스 & 댓글'
                    else:
                        exit()
            else:
                print('잘못 입력하셨습니다. 종료하겠습니다')
                exit()
            answer4 = int(input(f'{key}주제의 {keyword}수집을 완료하였습니다. 분석 단계를 진행하시겠습니까?(no: 0 yes: 1)'))
            if answer4 == 1:
                answer1 = 0
                continue
            else:
                print('종료하겠습니다.')
                exit()

        elif answer1 == 0:
            answer5 = int(input('어떤 자료를 분석하시겠습니까?(뉴스: 1, 블로그: 2, 댓글: 3)'))
            if answer5 == 1:
                answer6 = int(input('뉴스 - 어떤 분석을 하시겠습니까?(wordcloud: 0, 요약: 1)'))
                if answer6 == 0:
                    try:
                        news_data = pd.read_csv('naver_news_final.csv')
                        wordcloud(news_data)
                        keyword = '뉴스의 워드클라우드'
                    except:
                        print('분석을 위한 뉴스 자료는 존재하지 않습니다. 종료합니다.')
                        exit()
                elif answer6 == 1:
                    try:
                        news_data = pd.read_csv('naver_news_final.csv')
                        summarize_news_data(news_data)
                        keyword = '뉴스의 요약'
                    except:
                        print('분석을 위한 뉴스 자료는 존재하지 않습니다. 종료합니다.')
                        exit()
                else:
                    print('오류입니다. 전단계로 돌아갑니다.')
                    continue
            elif answer5 == 2:
                answer6 = int(input('블로그 - 어떤 분석을 하시겠습니까?(wordcloud: 0, 감성분석: 1)'))
                if answer6 == 0:
                    try:
                        blog_data = pd.read_csv('naver_blogs_final.csv')
                        blog_data = blog_data.rename(columns = {'document': 'content'})
                        wordcloud(blog_data)
                        keyword = '블로그의 워드클라우드'
                    except:
                        print('분석을 위한 블로그 자료는 존재하지 않습니다. 종료합니다.')
                        exit()
                elif answer6 == 1:
                    try:
                        blog_data = pd.read_csv('naver_blogs_final.csv')
                        sentimental_analysis_in_blog(blog_data)
                        keyword = '블로그의 감성분석'
                    except:
                        print('분석을 위한 블로그 자료는 존재하지 않습니다. 종료합니다.')
                        exit()
                else:
                    print('오류입니다. 전단계로 돌아갑니다.')
                    continue
            elif answer5 == 3:
                answer6 = int(input('댓글- 어떤 분석을 하시겠습니까?(wordcloud: 0, 요약: 1, 감성분석: 2)'))
                if answer6 == 0:
                    try:
                        comment_data = pd.read_csv('news_comment.csv')
                        comment_data = comment_data.rename(columns = {'document': 'content'})
                        wordcloud(comment_data)
                        keyword = '댓글의 워드클라우드'
                    except:
                        print('분석을 위한 댓글 자료는 존재하지 않습니다. 종료합니다.')
                        exit()
                elif answer6 == 1:
                    try:
                        comment_data = pd.read_csv('news_comment.csv')
                        summarize_comment_data(comment_data)
                        keyword = '댓글의 요약'
                    except:
                        print('분석을 위한 댓글 자료는 존재하지 않습니다. 종료합니다.')
                        exit()
                elif answer6 == 2:
                    try:
                        comment_data = pd.read_csv('news_comment.csv')
                        sentimental_analysis_in_comment(comment_data)
                        keyword = '댓글의 감성분석'
                    except:
                        print('분석을 위한 댓글 자료는 존재하지 않습니다. 종료합니다.')
                        exit()
                else:
                    print('오류입니다. 전단계로 돌아갑니다.')
                    continue
            else:
                print('잘못 입력하셨습니다. 종료하겠습니다')
                exit()
            answer4 = int(input(f'{key}주제의 {keyword}을/를 완료하였습니다. 어느 단계로 가시겠습니까?(종료: 0 수집: 1 분석: 2)'))
            if answer4 == 1:
                answer1 = 1
                continue
            elif answer4 == 2:
                answer1 = 0
                continue
            else:
                print('종료하겠습니다.')
                exit()
        else:
            print('잘못 입력하셨습니다. 종료합니다.')
            exit()


if __name__ == "__main__":
    main()
