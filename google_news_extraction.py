from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from datetime import datetime, timedelta
import time
import fitz  # PyMuPDF
import re


class ExtractESGInformation:
    def __init__(self, company):
    # Setup Chrome in headless mode
        self.chrome_options = Options()
        self.chrome_options.add_argument("--headless")
        self.chrome_options.add_argument("--disable-gpu")
        self.chrome_options.add_argument("--no-sandbox")
        self.chrome_options.add_argument("--window-size=1920x1080")

        # Setup WebDriver
        self.service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=self.service, options=self.chrome_options)
        
        self.company = company
        self.esg_search_url = f"https://www.esgtoday.com/?s={self.company}"
        self.report_search_url = "https://www.knowesg.com/esg-ratings"

    def __extract_news_from_esgtoday(self):
        # Search settings
        self.driver.get(self.search_url)
        time.sleep(2)

        news_data = []
        one_month_ago = datetime.now() - timedelta(days=365*2)

        while True:
            articles = self.driver.find_elements(By.TAG_NAME, "article")
            
            for article in articles:
                try:
                    title_element = article.find_element(By.CLASS_NAME, "post-title")
                    title = title_element.text
                    url = title_element.find_element(By.TAG_NAME, "a").get_attribute("href")

                    date_str = article.find_element(By.CLASS_NAME, "post-date").text
                    date_obj = datetime.strptime(date_str, "%B %d, %Y")

                    # Stop if article is older than one month
                    if date_obj < one_month_ago:
                        self.driver.quit()
                        print("Reached articles older than 1 month.")
                        break

                    # Open article page
                    self.driver.execute_script("window.open('');")
                    self.driver.switch_to.window(self.driver.window_handles[1])
                    self.driver.get(url)
                    time.sleep(1)

                    content = self.driver.find_element(By.CLASS_NAME, "post-content").text
                    author = self.driver.find_element(By.CLASS_NAME, "author-name").text

                    news = {
                        "title": title,
                        "url": url,
                        "author": author,
                        "date": date_str,
                        "summary": content
                    }
                    news_data.append(news)

                    self.driver.close()
                    self.driver.switch_to.window(self.driver.window_handles[0])

                except Exception as e:
                    continue

            # Try to go to the next page
            try:
                next_button = self.driver.find_element(By.CLASS_NAME, "nextp")
                next_button_url = next_button.get_attribute("href")
                self.driver.get(next_button_url)
                time.sleep(2)
            except Exception as e:
                print("No more pages.")
                break
        print(f"Extracted {len(news_data)} recent news articles.")
    
    def extract(self, pdf_path):
        esg_keywords = [
        "environment", "emission", "climate", "carbon", "biodiversity", "sustainability",
        "social", "diversity", "equity", "inclusion", "human rights", "labor", "workforce",
        "governance", "ethics", "compliance", "board", "transparency", "anti-corruption"
        ]

        # Compile a regex for matching any ESG keyword (case-insensitive)
        esg_pattern = re.compile(r"|".join(esg_keywords), re.IGNORECASE)

        doc = fitz.open(pdf_path)
        extracted_text = []

        for page_num in range(len(doc)):
            text = doc[page_num].get_text()
            if esg_pattern.search(text):
                extracted_text.append(f"\n--- Page {page_num + 1} ---\n{text.strip()}")

        return "\n".join(extracted_text)



if __name__ == "__main__":
    company = input("Enter the name of the company: ")
    obj = ExtractESGInformation(company)
    path = "Apple_Environmental_Progress_Report_2024.pdf"
    # path = "mercedes-benz-sustainability-report-2023.pdf"
    text = obj.extract(path)
    with open("text.txt", "w") as file:
        file.write(text)
    
    