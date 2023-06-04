from konlpy.tag import Okt
from collections import Counter
import pandas as pd

import re
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc


from yellowbrick.cluster import KElbowVisualizer

from konlpy.tag import Komoran
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import seaborn as sns

import spacy
from string import punctuation
from heapq import nlargest

font_path = "C:/Windows/Fonts/gulim.ttc"
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)


def finding_number_of_clusters(df):
    k = 0
    kmeans = KMeans(n_clusters = k, random_state = 10)
    visualizer = KElbowVisualizer(kmeans, k = (1, 9), timings = False)
    visualizer.fit(df)
    visualizer.show()


def wordcloud(naver_data):
    okt = Okt()
    line = []
    all_words = list()

    for text in naver_data['content']:
        line = okt.pos(text)  # 빈칸 라인(line = [])에 품사 붙여주는 함수 pos 사용
        n_adj = []
        for word, tag in line:
            if tag in ["Noun", "Adjective"]:  # 명사 형용사 뽑을 것
                n_adj.append(word)
        # print(n_adj)
        all_words.extend(n_adj)
    # print(all_words)
    stop_words = "윤 지난 제 주 등 며 있다 고 이라고 는데 다는 다며 니다 기자 이어 으면 연합뉴스 아니 아서 아야 도록 때문 ㄴ다 ㄴ다고 ㄴ다는 ㅂ니다 에서 ' 마다 로 - " \
                 "입니다 있습니다 회 라는 라며 면서 부터 보다 에게 도 의 인 과 습니다 와 △ 을 를 전 난 일 걸 뭐 줄 만 그럼 하나 왜 자기 임 사람 발표 있는" \
                 "건 분 개 끝 잼 이거 번 중 듯 때 게 내 말 나 수 거 점 것 만큼 아 으로 던 다 에 이 것 는 가 그 YTN 국민 시간 생각 더 니 해 애 보고 하라 너 대통령" \
                 "안 입 본인 이제 뭘 답 알 저 얘기 명 달 배 위해 건 쉬 앞 살 질 함 네 찍 좀 있는 또 같은 지금 누가 먼저 안안 안  안 그냥 못 우리 해도 누구 언제 건가 기사" \
                 "당신 연차 이야기 날 정 계속 그게 열 탓 자 처 팔 데 없어 머리"
    stop_words = set(stop_words.split(" "))

    all_words = [word for word in all_words if not word in stop_words]
    counts = Counter(all_words)  # 각 단어 몇 번 등장했는지 카운팅
    # words_freq = pd.DataFrame.from_dict(counts, orient = 'index').reset_index()
    # print(counts)
    tags = counts.most_common(50)  # 내용이 많으면 오래걸림 예)10000개는 하루정도 걸림
    most_freq = pd.DataFrame.from_dict(counts, orient = 'index').reset_index()
    most_freq = most_freq.rename(columns={'index': 'event', 0: 'count'})

    freq_words = most_freq[most_freq['count'].rank(ascending = False) <= 150]
    # print(freq_words)
    freq_words = freq_words.sort_values('count', ascending = False)
    freq_words.to_csv('most_freq_150.csv', index = False, encoding = "utf-8-sig")

    word_cloud = WordCloud(background_color = "white",  # 바탕색 지정(색깔 자유롭게 지정)
                           max_font_size = 200,
                           colormap = "prism",
                           font_path = font_path).generate_from_frequencies(dict(tags))
    plt.figure(figsize = (10, 8))  # 가로, 세로 사이즈
    plt.imshow(word_cloud)
    plt.axis("off")
    plt.show()


def summarize_content(summary_need_content):
    per = 0.05
    nlp = spacy.load("ko_core_news_sm")
    table = str.maketrans('[]"=,()!@#$%^&*\'', '                ')
    news_fixed = summary_need_content.translate(table)
    doc = nlp(news_fixed)
    # tokens = [token.text for token in doc]
    word_frequencies = dict()
    for word in doc:
        if word.text not in punctuation:
            if word.text not in word_frequencies.keys():
                word_frequencies[word.text] = 1
            else:
                word_frequencies[word.text] += 1
    max_frequency = max(word_frequencies.values())
    for words in word_frequencies.keys():
        word_frequencies[words] = word_frequencies[words]/max_frequency
    sentence_tokens = [sentence for sentence in doc.sents]
    sentence_scores = dict()
    for sentence in sentence_tokens:
        for word in sentence:
            if word.text in punctuation:
                pass
            else:
                if sentence not in sentence_scores.keys():
                    sentence_scores[sentence] = word_frequencies[f'{word}']
                else:
                    sentence_scores[sentence] += word_frequencies[f'{word}']
    select_length = int(len(sentence_tokens)*per)
    if select_length < 2:
        select_length = 2
    else:
        pass
    summary = nlargest(select_length, sentence_scores, key = sentence_scores.get)
    final_summary = [word.text for word in summary]
    summary = ''.join(final_summary)
    return summary


def text_cluster_analysis(naver_data):
    print('start text clustering analysis')
    contents = list()
    print('start sentence split')
    for data in naver_data['content']:
        sentences = re.split(r"[.?!]", data)
        sentences = [s.strip() for s in sentences if s.strip()]
        contents.extend(sentences)

    print('finished sentence split')
    # print(contents)
    data = list()
    for content in contents:
        content = content.replace('\n', '')
        content = content.replace('[', '')
        content = content.replace(']', '')
        content = content.replace('"', '')
        content = content.replace('■', '')
        content = content.replace('/', '')
        content = content.replace('?', '')
        content = content.replace('.', '')
        content = content.replace('#', '')
        content = content.replace(',', '')
        data.append(content)

    data = ' '.join(data).split()

    # Text preprocessing
    print('start text preprocessing')
    komoran = Komoran()
    morphs = [komoran.morphs(x) for x in data]
    morphs = list(map(lambda x: " ".join(x), morphs))

    # Stop words
    stop_words = "이라고 는데 다는 다며 니다 기자 이어 으면 연합뉴스 아니 아서 아야 도록 때문 ㄴ다 ㄴ다고 ㄴ다는 ㅂ니다 에서 ' 마다 로 - " \
                 "라는 라며 면서 부터 보다 에게 도 의 인 과 습니다 와 △ 을 를 전 난 일 걸 뭐 줄 만 " \
                 "건 분 개 끝 잼 이거 번 중 듯 때 게 내 말 나 수 거 점 것 만큼 아 으로 던 다 에 이 것 는 가"
    stop_words = stop_words.split(" ")

    # Word vectorization
    print("start doing word vectorization")
    vectorizer = CountVectorizer(stop_words = stop_words)
    bow = vectorizer.fit_transform(morphs)

    # Convert to DataFrame
    print('convert into dataframe')
    columns = vectorizer.get_feature_names_out()
    df_words = pd.DataFrame(bow.toarray(), columns = columns)

    print('find the most used: 150')
    df_tdm = df_words.T
    df_tdm['total'] = df_tdm.sum(axis = 1)
    df_words = df_tdm[df_tdm['total'].rank(ascending = False) <= 150]
    df_words = df_words.drop('total', axis = 1)
    df_words.to_csv('cluster_of_news_whole.csv', index = True, encoding = "utf-8-sig")
    print('finished finding')
    # print(df_words[:5])

    # find best-fit k in k-means cluster
    print('about to find best fitting k ink k-means cluster')
    finding_number_of_clusters(df_words)
    cluster_number = int(input("클러스터 개수: "))

    # Clustering
    kmeans = KMeans(n_clusters = cluster_number)
    predict = kmeans.fit_predict(df_words)
    df_words['predict'] = predict
    # print(df_words[:5])
    print('start making csv files')
    df_csv = df_words[['predict']]
    df_csv.to_csv('cluster_of_news.csv', index = True, encoding = "utf-8-sig")

    # Dimensionality reduction
    pca = PCA(n_components = 2)
    word_pca = pca.fit_transform(df_words.iloc[:, :-1])
    df_pca = pd.DataFrame(data = word_pca, index = df_words.index, columns = ['main1', 'main2'])
    df_pca['predict'] = predict

    # Plotting
    sns.relplot(x = 'main1', y = 'main2', data = df_pca, height = 8, aspect = 1.5, hue = 'predict')

    # Labeling the plot
    for i, index in enumerate(df_pca.index):
        plt.text(df_pca.main1[i], df_pca.main2[i], index)

    plt.show()


def main():
    naver_data = pd.read_csv('naver_news_final.csv')
    # print(naver_data.head())
    # text_cluster_analysis(naver_data)
    wordcloud(naver_data)
    # summarizations = list()
    # for each_news in naver_data['content']:
    #     summarize = summarize_content(each_news)
    #     summarizations.append(summarize)
    # print('finished summarization of each article')
    # naver_data['summary'] = summarizations
    # naver_data.to_csv('naver_news_final_with_summaries.csv', index = False, encoding = "utf-8-sig")
    # summaries_of_each_news = ''.join(summarizations)
    # total_summary = summarize_content(summaries_of_each_news)
    #
    # for time in range(1, 10):
    #     print(f'추가요약 {time}번째')
    #     total_summary = summarize_content(total_summary)
    #
    # print(total_summary)


if __name__ == '__main__':
    main()
