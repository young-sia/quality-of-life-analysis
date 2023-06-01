import urllib.request
import json
import os
from dotenv import load_dotenv  # 이 패키지는 host가 따로 없을 때에 사용한다.

from datetime import datetime

import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
import pandas as pd
import seaborn as sns

import logging


load_dotenv()
client_id = os.environ.get('NAVER_ID')
client_secret = os.environ.get("NAVER_PASSWORD")

font_path = "C:/Windows/Fonts/gulim.ttc"
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)


def make_time(date):
    start_date = date['start_date']
    end_date = date['end_date']

    dates = pd.date_range(start_date, end_date, freq = 'M')
    return dates


def retrieve_search_trend_data():
    logging.info('about to request naver search result')
    url = "https://openapi.naver.com/v1/datalab/search"
    keyword_groups = [
        {'groupName': '제조업', 'keywords': ['제조업 채용', '제조업 근무', '제조업 일자리']},
        {'groupName': '전기, 가스, 증기 및 공기조절', 'keywords': ['전기 채용', '전기 근무', '전기 일자리', '가스 채용', '가스 근무', '가스 일자리']},
        {'groupName': '광업', 'keywords': ['광업 채용', '광업 근무', '광업 일자리']},
        {'groupName': '수도, 하수 및 폐기물 처리, 원료 재생업', 'keywords': ['', '']},
        {'groupName': '건설업', 'keywords': ['', '']},
        {'groupName': '도매 및 소매업', 'keywords': ['', '']},
        {'groupName': '운수 및 창고업', 'keywords': ['', '']},
        {'groupName': '숙박 및 음식점업', 'keywords': ['', '']},
        {'groupName': '정보통신업', 'keywords': ['', '']},
        {'groupName': '금융 및 보험업', 'keywords': ['', '']},
        {'groupName': '부동산업', 'keywords': ['', '']},
        {'groupName': '전문, 과학 및 기술 서비스업', 'keywords': ['', '']},
        {'groupName': '사업시설 관리, 사업 지원 및 임대 서비스업', 'keywords': ['', '']},
        {'groupName': '교육 서비스업', 'keywords': ['', '']},
        {'groupName': '보건업 및 사회복지 서비스업', 'keywords': ['', '']},
        {'groupName': '예술, 스포츠 및 여가관련 서비스업', 'keywords': ['', '']},
        {'groupName': '협회 및 단체, 수리 및 기타 개인 서비스업', 'keywords': ['', '']},
    ]
    date = {'start_date': '2022-01-01', 'end_date': '2023-02-28'}

    body = {
        'startDate': date['start_date'],
        'endDate': date['end_date'],
        'timeUnit': 'month',  # date일간, week주간, month월간
        'keywordGroups': keyword_groups}
    body = json.dumps(body)

    request = urllib.request.Request(url)
    request.add_header("X-Naver-Client-Id", client_id)
    request.add_header("X-Naver-Client-Secret", client_secret)
    request.add_header("Content-Type", "application/json")

    response = urllib.request.urlopen(request, data = body.encode("utf-8"))
    res_code = response.getcode()

    if res_code == 200:
        response_body = response.read()
        response_json = json.loads(response_body)
        logging.info(response_body.decode('utf-8'))
        return response_json, keyword_groups, date
    else:
        logging.info(f'Error Code:' + res_code)


def transform_search_trend_data_into_graph(trend_data):
    # print(trend_data)

    response_results_all = pd.DataFrame()

    response_results = pd.DataFrame()
    for data in trend_data['results']:
        result = pd.DataFrame(data['data'])
        result['title'] = data['title']

        response_results = pd.concat([response_results, result])

    response_results_all = pd.concat([response_results_all, response_results])
    response_results_all = response_results_all.sort_values('period')
    # print(response_results_all)
    titles = response_results['title'].unique()
    # print(titles)
    return titles, response_results_all


def plot_daily_trend(titles, response_results_all, dates):
    # print(response_results_all)
    ymd = list()
    for date in response_results_all['period']:
        ymd.append(datetime.strptime(date, '%Y-%m-%d'))
    response_results_all['date'] = ymd
    response_results_all['datetime_m'] = response_results_all['date'].dt.strftime('%Y-%m')
    month_chart = response_results_all.drop(['period', 'date'], axis = 'columns')

    dates_month = pd.DataFrame()
    dates_month['datetime_m'] = dates
    dates_month['datetime_m'] = dates_month['datetime_m'].dt.strftime('%Y-%m')

    for keyword in titles:
        for month in dates_month['datetime_m']:
            temp = month_chart[(month_chart['title'] == keyword)]
            if (temp['datetime_m'] != month).all():
                add_data = {'title': [keyword], 'datetime_m': [month], 'ratio': [0]}
                add_data = pd.DataFrame(add_data)

                month_chart = pd.concat([month_chart, add_data])

    month_chart['whole_ratio'] = month_chart['ratio'] / sum(month_chart['ratio']) * 100
    # print(month_chart)
    month_chart = month_chart.sort_values('datetime_m', ascending = True)

    fig = plt.figure(figsize = (6, 2))
    plt.title('통합트렌드', size = 20, weight = "bold")
    for title in titles:
        data = month_chart.loc[(month_chart['title'] == title), :]
        print(data)
        plt.plot(data['datetime_m'], data['whole_ratio'], label = title)
        plt.xticks(rotation = 90)
        plt.legend()

    plt.show()
    return month_chart


def main():
    trend_data, keyword, date = retrieve_search_trend_data()
    dates = make_time(date)
    # print(dates)
    # print(trend_data)
    title, response_chart = transform_search_trend_data_into_graph(trend_data)
    # print(response_chart)
    month_chart = plot_daily_trend(title, response_chart, dates)
    # month_chart.to_csv('통합트렌드.csv', index = False, encoding = "utf-8-sig")


if __name__ == "__main__":
    main()



