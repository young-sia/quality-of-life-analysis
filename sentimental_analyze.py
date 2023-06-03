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
    blog_data = pd.read_csv('naver_news_final.csv')
    sentimental_result = []
    count = 0

    summarizations = list()
    for each_blog in blog_data['content']:
        summarizations.extend(each_blog)

    for content in blog_data['content']:
        print('start of analyzing')
        number_of_sentences = 0
        sum_of_sentiment = 0
        sentences = content.split('.')
        for sent in sentences:
            print('loading....')
            # print(sent)
            sum_of_sentiment += sentimental_analysis(sent)
            number_of_sentences += 1

        average_of_sentement = sum_of_sentiment / number_of_sentences
        sentimental_result.append(average_of_sentement)

        count += 1
        print(sentimental_result)
        print(count, '번째 분석 완료')
    blog_data['feelings'] = sentimental_result
    blog_data.to_csv('naver_blog_with_sentiment.csv', index = False, encoding = "utf-8-sig")


if __name__ == '__main__':
    main()


