from bs4 import BeautifulSoup
from selenium import webdriver
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import time
import csv
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


def get_amazon_reviews(url):
    # 使用Selenium模擬瀏覽器行為
    driver = webdriver.Chrome()  
    driver.get(url)

    # 滾動頁面以觸發評論載入
    last_height = driver.execute_script("return document.body.scrollHeight")
    while True:
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(2)  # 等待頁面載入
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            break
        last_height = new_height

    # 等待評論載入完成
    try:
        element_present = EC.presence_of_element_located((By.CLASS_NAME, 'cr-original-review-content'))
        WebDriverWait(driver, 10).until(element_present)
    except TimeoutException:
        print("Timed out waiting for page to load")

    # 獲取評論
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    reviews = soup.find_all('span', {'class': 'cr-original-review-content'})
    
    driver.quit()  # 關閉瀏覽器

    return [review.get_text(strip=True) for review in reviews]

#emotion data
def perform_sentiment_analysis(reviews):
    sentiment_analysis = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")
    emotion_analyzer = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")

    results = []

    for review in reviews:
        sentiment_result = sentiment_analysis(review)
        sentiment_label = sentiment_result[0]['label']
        # 將情感分析的分數轉換為0到10的指數
        sentiment_score = round(sentiment_result[0]['score'] * 10, 2)

        emotion_result = emotion_analyzer(review)
        emotion_label = emotion_result[0]['label']
        emotion_score = emotion_result[0]['score']
         # 輸出結果以檢查
        print(f"Original Review: {review}")
        print(f"Sentiment Result: {sentiment_result}")
        print(f"Emotion Result: {emotion_result}")
        
        # 將特殊字符轉換為空字符串
        review_cleaned = ''.join(e for e in review if (e.isalnum() or e.isspace()))
        
        results.append({'Review': review_cleaned, 'Sentiment_Label': sentiment_label, 'Sentiment_Score': sentiment_score, 'Emotion_Label': emotion_label, 'Emotion_Score': emotion_score})

    return results


#Result storage
def save_results_to_csv(results, output_file):
    fields = ['Review', 'Sentiment_Label', 'Sentiment_Score', 'Emotion_Label', 'Emotion_Score']

    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)


if __name__=="__main__":
    amazon_url = "https://www.amazon.com/-/zh_TW/MageGee-%E6%A9%9F%E6%A2%B0%E9%81%8A%E6%88%B2%E9%8D%B5%E7%9B%A4-MK-Box-%E9%8D%B5%E8%BF%B7%E4%BD%A0%E6%9C%89%E7%B7%9A%E8%BE%A6%E5%85%AC%E5%AE%A4%E9%8D%B5%E7%9B%A4-Windows/dp/B098LG3N6R/ref=sr_1_1?_encoding=UTF8&content-id=amzn1.sym.eda9e82b-14da-4bca-aadc-b606e015822c&keywords=gaming%2Bkeyboard&pd_rd_r=9db113a1-cf36-4524-8ffa-4d259ea447e1&pd_rd_w=VcRRc&pd_rd_wg=0f3Tz&pf_rd_p=eda9e82b-14da-4bca-aadc-b606e015822c&pf_rd_r=0H0N9JT0XNVKGR4R0Y4T&qid=1706242766&sr=8-1&th=1"
    output_csv_file = "output_sentiment_analysis.csv"

    reviews = get_amazon_reviews(amazon_url)
    results = perform_sentiment_analysis(reviews)
    save_results_to_csv(results, output_csv_file)
