import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import re
from tqdm import tqdm
import torch
import logging
from analyze_data import *


def removing_non_korean(df):
    for idx, row in tqdm(df.iterrows(), desc = 'removing_non_korean', total = len(df)):
            new_doc = re.sub('[^가-힣]', '', row['document']).strip()
            df.loc[idx, 'document'] = new_doc
    return df


def sentimental_analysis(text):
    # Step 1: Load the pre-trained koELECTRA model and tokenizer
    model_name = "monologg/koelectra-base-v3-discriminator"
    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    # model = AutoModelForSequenceClassification.from_pretrained(model_name)

    # Disable the logging level for the specific warning
    logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

    # Load the model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    # Step 2: Define the sentiment labels
    sentiment_labels = ["negative", "positive"]

    # Step 3: Prepare the input text
    inputs = tokenizer(text, return_tensors = "pt")
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    # Step 4: Perform sentiment analysis
    with torch.no_grad():
        logits = model(input_ids, attention_mask = attention_mask).logits
    predicted_label_idx = torch.argmax(logits, dim = 1).item()
    predicted_label = sentiment_labels[predicted_label_idx]

    # Step 5: Print the predicted sentiment label
    # print("Input text:", text)
    # print("Predicted sentiment:", predicted_label)

    if predicted_label == 'positive':
        return 1
    else:
        return 0


def main():
    blog_data = pd.read_csv('naver_blogs_final.csv')
    blog_data = blog_data.dropna(axis=0)
    # wordcloud(blog_data)

    # # 요약본을 감성분석할 때
    blog_data_1 = blog_data.sample(frac = 0.5, random_state = 2022)
    blog_data_2 = blog_data.drop(blog_data_1.index)
    blog_content_1 = blog_data_1['content']
    blog_content_2 = blog_data_2['content']

    content1 = list()
    for content in blog_content_1:
        # print(content)
        content1.append(content)

    content2 = list()
    for content in blog_content_2:
        content2.append(content)
    print('combined into 2 contents')
    summarization = list()
    for content in content1:
        if isinstance(content, str):
            summary = summarize_content(content)
            summarization.extend(summary)
        else:
            print('error')
            pass
    for content in content2:
        if isinstance(content, str):
            summary = summarize_content(content)
            summarization.extend(summary)
        else:
            print('error')
            pass
    print('finished summarizing 2 content')
    summarizations = summarize_content(summarization)
    # print(summarizations)
    print('finished combining and summarizing')

    with open('summary_of_blogs.txt', 'w', encoding = 'UTF-8') as f:
        for name in summarizations:
            f.write(name)

    print('start of analyzing')
    sentences = summarizations.split('.')
    number_of_sentences = 0
    sum_of_sentiment = 0
    score_of_sentences = list()

    for content in sentences:
        # print(content)
        print('loading.......')
        score_of_sentence = sentimental_analysis(content)
        score_of_sentences.append(score_of_sentence)
        sum_of_sentiment += score_of_sentence
        number_of_sentences += 1
        print(number_of_sentences, '번째 문장 분석 완료')

    average_of_sentiment = sum_of_sentiment/number_of_sentences
    print(average_of_sentiment)

    summary_sentimental_result = {'sentence': sentences, 'score': score_of_sentences}
    summary_sentimental_result = pd.DataFrame(summary_sentimental_result)
    summary_sentimental_result.to_csv('sentimental result of summary.csv', index = False, encoding = "utf-8-sig")

    # # 블로그 내용 하나하나 전부 다 감성분석할 때
    # sentimental_result = []
    # count = 0
    # for content in blog_data['content']:
    #     print('start of analyzing')
    #     number_of_sentences = 0
    #     sum_of_sentiment = 0
    #     sentences = content.split('.')
    #     for sent in sentences:
    #         print('loading....')
    #         # print(sent)
    #         sum_of_sentiment += sentimental_analysis(sent)
    #         number_of_sentences += 1
    #
    #     average_of_sentement = sum_of_sentiment / number_of_sentences
    #     sentimental_result.append(average_of_sentement)
    #
    #     count += 1
    #     print(sentimental_result)
    #     print(count, '번째 분석 완료')
    # blog_data['feelings'] = sentimental_result
    # blog_data.to_csv('naver_blog_with_sentiment.csv', index = False, encoding = "utf-8-sig")


if __name__ == '__main__':
    main()


