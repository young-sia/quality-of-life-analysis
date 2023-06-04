from scrape_news_data import *
from analyze_data import *


def main():
    start_number = [1, 101, 201]
    keywords = ['주 52시간제']
    # keywords = ['주 69시간', '주 60시간', '주 4일제', '주 4.5일제']
    retrieved_data = {'items': []}
    for keyword in keywords:
        for start in start_number:
            news_data = retrieve_search_news_data(start, keyword)
            retrieved_data['items'].extend(news_data['items'])
            print(f'finished {keyword}, {start} page')

    naver_news, other_news = distinguish_naver_other_news(retrieved_data)
    other_news = pd.DataFrame(other_news)
    other_news = other_news.drop_duplicates(subset = None, keep = 'first', inplace = False, ignore_index = False)
    other_news.to_csv('other_news_final.csv', index = False, encoding = "utf-8-sig")

    driver = webdriver.Chrome(ChromeDriverManager().install())
    driver.implicitly_wait(3)
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/98.0.4758.102"}
    naver_news_all = get_content_data_from_naver(headers, naver_news)
    naver_news_all.to_csv('naver_news_final.csv', index = False, encoding = "utf-8-sig")

    # text_cluster_analysis(naver_news_all)


if __name__ == "__main__":
    main()

