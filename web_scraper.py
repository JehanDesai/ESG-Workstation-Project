# import logging
# import time
# from typing import List, Dict

# from selenium import webdriver
# from selenium.webdriver.common.by import By
# from selenium.webdriver.chrome.service import Service
# from selenium.webdriver.chrome.options import Options
# from selenium.webdriver.support.ui import WebDriverWait
# from selenium.webdriver.support import expected_conditions as EC
# from selenium.common.exceptions import TimeoutException, NoSuchElementException, WebDriverException

# from webdriver_manager.chrome import ChromeDriverManager

# # Configure logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)

# class ESGNewsScraper:
#     def __init__(self, timeout=20):
#         #Initialize the web scraper with Chrome WebDriver
#         # Setup Chrome options for stealth and stability
#         chrome_options = Options()
#         chrome_options.add_argument("--no-sandbox")
#         chrome_options.add_argument("--disable-dev-shm-usage")
#         chrome_options.add_argument("--disable-extensions")
#         chrome_options.add_argument("--disable-gpu")
#         chrome_options.add_argument("--start-maximized")
#         chrome_options.add_argument("user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.93 Safari/537.36")
#         # Initialize WebDriver
#         try:
#             self.driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
#             self.driver.set_page_load_timeout(30)
#             self.timeout = timeout
#         except Exception as e:
#             logger.error(f"WebDriver initialization failed: {e}")
#             raise

#     def safe_get(self, url: str) -> bool:
#         #Safely navigate to a URL with error handling
#         try:
#             self.driver.get(url)
#             logger.info(f"Successfully navigated to {url}")
#             # Check for potential block pages
#             if "blocked" in self.driver.page_source.lower() or "captcha" in self.driver.page_source.lower():
#                 logger.warning("Possible blocking detected")
#                 return False
#             return True
#         except WebDriverException as e:
#             logger.error(f"Failed to load {url}: {e}")
#             return False

#     def search_esg_news(self, company_name: str) -> List[Dict]:
#         #Comprehensive search method for ESG news
#         articles = []
#         # Multiple search strategies
#         search_strategies = [f"https://esgnews.com/?s={company_name.replace(' ', '+')}"]
#         for search_url in search_strategies:
#             logger.info(f"Attempting search with URL: {search_url}")
#             # Reset driver to prevent session issues
#             try:
#                 self.driver.delete_all_cookies()
#             except:
#                 pass
#             # Try navigating to search URL
#             if not self.safe_get(search_url):
#                 logger.warning(f"Failed to load search URL: {search_url}")
#                 continue
#             # Wait and extract articles
#             try:
#                 # Multiple potential article selectors
#                 article_selectors = [".search-results article",".search-result","article.post",".news-item","div.article"]
#                 articles_found = False
#                 for selector in article_selectors:
#                     try:
#                         # Wait for articles with this selector
#                         WebDriverWait(self.driver, self.timeout).until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, selector)))
#                         # Find articles
#                         page_articles = self.driver.find_elements(By.CSS_SELECTOR, selector)
#                         if page_articles:
#                             logger.info(f"Found articles using selector: {selector}")
#                             articles_found = True
#                             # Process each article
#                             for article in page_articles:
#                                 try:
#                                     # Flexible title extraction
#                                     title_selectors = ["h2 a", "h3 a", ".title", ".article-title", "a.title", ".entry-title a"]
#                                     title_element = None
#                                     for title_sel in title_selectors:
#                                         try:
#                                             title_element = article.find_element(By.CSS_SELECTOR, title_sel)
#                                             break
#                                         except:
#                                             continue
#                                     if not title_element:
#                                         continue
#                                     title = title_element.text
#                                     link = title_element.get_attribute("href")
#                                     # Extract date
#                                     date_selectors = [".date", "time", ".published", ".entry-date", ".article-date"]
#                                     date = "Date not found"
#                                     for date_sel in date_selectors:
#                                         try:
#                                             date = article.find_element(By.CSS_SELECTOR, date_sel).text
#                                             break
#                                         except:
#                                             continue
#                                     # Extract excerpt
#                                     excerpt_selectors = ["p", ".excerpt", ".description", ".summary", ".article-excerpt"]
#                                     excerpt = "No excerpt available"
#                                     for excerpt_sel in excerpt_selectors:
#                                         try:
#                                             excerpt = article.find_element(By.CSS_SELECTOR, excerpt_sel).text
#                                             if len(excerpt) > 20:  # Ensure it's a meaningful excerpt
#                                                 break
#                                         except:
#                                             continue
#                                     # Create article dictionary
#                                     article_data = {"title": title,"date": date,"link": link,"excerpt": excerpt,"source": "ESG News"}
#                                     articles.append(article_data)
#                                 except Exception as e:
#                                     logger.warning(f"Error processing individual article: {e}")
#                             # Break if articles found
#                             break
#                     except TimeoutException:
#                         logger.debug(f"No articles found with selector: {selector}")
#                 # If articles found, exit search strategies
#                 if articles_found:
#                     break
#             except Exception as e:
#                 logger.error(f"Error during search: {e}")
#         return articles

#     def save_results(self, articles: List[Dict], filename: str = "esg_news_results.txt"):
#         #Save scraped articles to a text file
#         try:
#             with open(filename, 'w', encoding='utf-8') as f:
#                 for article in articles:
#                     f.write(f"Title: {article['title']}\n")
#                     f.write(f"Date: {article['date']}\n")
#                     f.write(f"Link: {article['link']}\n")
#                     f.write(f"Excerpt: {article['excerpt']}\n")
#                     f.write("-" * 50 + "\n")
#             logger.info(f"Results saved to {filename}")
#         except Exception as e:
#             logger.error(f"Error saving results: {e}")

#     def close(self):
#         #Close the WebDriver
#         try:
#             self.driver.quit()
#             logger.info("WebDriver closed successfully")
#         except Exception as e:
#             logger.error(f"Error closing WebDriver: {e}")

# def main():
#     scraper = ESGNewsScraper()
#     try:
#         # Get company name from user
#         company_name = input("Enter the company name to search for ESG news: ")
#         # Search for articles
#         articles = scraper.search_esg_news(company_name)
#         # Display and save results
#         if articles:
#             print(f"\nFound {len(articles)} ESG news articles for {company_name}:\n")
#             for idx, article in enumerate(articles, 1):
#                 print(f"{idx}. {article['title']}")
#                 print(f"   Date: {article['date']}")
#                 print(f"   Link: {article['link']}\n")
#             # Save results to file
#             scraper.save_results(articles)
#             print(f"Results saved to 'esg_news_results.txt'")
#         else:
#             print(f"No ESG news articles found for {company_name}")
    
#     except Exception as e:
#         logger.error(f"An unexpected error occurred: {e}")
    
#     finally:
#         # Ensure WebDriver is closed
#         scraper.close()

# if __name__ == "__main__":
#     main()


import logging
import time
from typing import List, Dict

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException, WebDriverException

from webdriver_manager.chrome import ChromeDriverManager

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ESGNewsScraper:
    def __init__(self, timeout=20):
        # Initialize the web scraper with Chrome WebDriver
        chrome_options = Options()
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-extensions")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--start-maximized")
        chrome_options.add_argument("user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.93 Safari/537.36")
        
        try:
            self.driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
            self.driver.set_page_load_timeout(30)
            self.timeout = timeout
        except Exception as e:
            logger.error(f"WebDriver initialization failed: {e}")
            raise

    def safe_get(self, url: str) -> bool:
        # Safely navigate to a URL with error handling
        try:
            self.driver.get(url)
            logger.info(f"Successfully navigated to {url}")
            if "blocked" in self.driver.page_source.lower() or "captcha" in self.driver.page_source.lower():
                logger.warning("Possible blocking detected")
                return False
            return True
        except WebDriverException as e:
            logger.error(f"Failed to load {url}: {e}")
            return False

    def extract_article_content(self, url: str) -> str:
        """
        Extract the full content of an article from its URL
        
        Args:
            url (str): URL of the article to extract
        
        Returns:
            str: Full text content of the article
        """
        try:
            # Navigate to the article URL
            if not self.safe_get(url):
                logger.warning(f"Failed to load article URL: {url}")
                return "Content extraction failed"

            # List of potential content selectors
            content_selectors = [
                # Common article content selectors
                "article .entry-content",  # WordPress-style
                "div.article-body",
                "div.post-content",
                "div.content",
                "div.article-text",
                "article",
                "main .content",
                "body article",
                "#main-content",
                ".article-body",
                "div[itemprop='articleBody']",
                "div.articleBody",
                "div.entry",
                "div.post"
            ]

            # Try each selector to find article content
            for selector in content_selectors:
                try:
                    # Wait for content element
                    content_element = WebDriverWait(self.driver, self.timeout).until(
                        EC.presence_of_element_located((By.CSS_SELECTOR, selector))
                    )
                    
                    # Extract text content
                    content = content_element.text
                    
                    # Basic content validation
                    if content and len(content) > 50:
                        logger.info(f"Successfully extracted content using selector: {selector}")
                        return content
                
                except (TimeoutException, NoSuchElementException):
                    # If this selector doesn't work, continue to next
                    continue

            # If no content found
            logger.warning(f"Could not extract content for URL: {url}")
            return "Content extraction failed"

        except Exception as e:
            logger.error(f"Error extracting article content: {e}")
            return "Content extraction failed"

    def search_esg_news(self, company_name: str) -> List[Dict]:
        # Existing search method with slight modification to include content extraction
        articles = []
        search_strategies = [f"https://esgnews.com/?s={company_name.replace(' ', '+')}"]
        
        for search_url in search_strategies:
            logger.info(f"Attempting search with URL: {search_url}")
            
            try:
                self.driver.delete_all_cookies()
            except:
                pass
            
            if not self.safe_get(search_url):
                logger.warning(f"Failed to load search URL: {search_url}")
                continue
            
            try:
                article_selectors = [".search-results article", ".search-result", "article.post", ".news-item", "div.article"]
                articles_found = False
                
                for selector in article_selectors:
                    try:
                        WebDriverWait(self.driver, self.timeout).until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, selector)))
                        page_articles = self.driver.find_elements(By.CSS_SELECTOR, selector)
                        
                        if page_articles:
                            logger.info(f"Found articles using selector: {selector}")
                            articles_found = True
                            
                            for article in page_articles:
                                try:
                                    # Flexible title extraction
                                    title_selectors = ["h2 a", "h3 a", ".title", ".article-title", "a.title", ".entry-title a"]
                                    title_element = None
                                    for title_sel in title_selectors:
                                        try:
                                            title_element = article.find_element(By.CSS_SELECTOR, title_sel)
                                            break
                                        except:
                                            continue
                                    
                                    if not title_element:
                                        continue
                                    
                                    title = title_element.text
                                    link = title_element.get_attribute("href")
                                    
                                    # Extract article content
                                    content = self.extract_article_content(link)
                                    
                                    # Date extraction
                                    date_selectors = [".date", "time", ".published", ".entry-date", ".article-date"]
                                    date = "Date not found"
                                    for date_sel in date_selectors:
                                        try:
                                            date = article.find_element(By.CSS_SELECTOR, date_sel).text
                                            break
                                        except:
                                            continue
                                    
                                    # Excerpt extraction
                                    excerpt_selectors = ["p", ".excerpt", ".description", ".summary", ".article-excerpt"]
                                    excerpt = "No excerpt available"
                                    for excerpt_sel in excerpt_selectors:
                                        try:
                                            excerpt = article.find_element(By.CSS_SELECTOR, excerpt_sel).text
                                            if len(excerpt) > 20:
                                                break
                                        except:
                                            continue
                                    
                                    # Create article dictionary with full content
                                    article_data = {
                                        "title": title,
                                        "date": date,
                                        "link": link,
                                        "excerpt": excerpt,
                                        "content": content,
                                        "source": "ESG News"
                                    }
                                    articles.append(article_data)
                                
                                except Exception as e:
                                    logger.warning(f"Error processing individual article: {e}")
                            
                            # Break if articles found
                            break
                    
                    except TimeoutException:
                        logger.debug(f"No articles found with selector: {selector}")
                
                # If articles found, exit search strategies
                if articles_found:
                    break
            
            except Exception as e:
                logger.error(f"Error during search: {e}")
        
        return articles

    def save_results(self, articles: List[Dict], filename: str = "esg_news_results.txt"):
        # Modified to include full article content
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                for article in articles:
                    f.write(f"Title: {article['title']}\n")
                    f.write(f"Date: {article['date']}\n")
                    f.write(f"Link: {article['link']}\n")
                    f.write(f"Excerpt: {article['excerpt']}\n")
                    f.write("Full Content:\n")
                    f.write(f"{article['content']}\n")
                    f.write("-" * 50 + "\n")
            logger.info(f"Results saved to {filename}")
        except Exception as e:
            logger.error(f"Error saving results: {e}")

    def close(self):
        # Close the WebDriver
        try:
            self.driver.quit()
            logger.info("WebDriver closed successfully")
        except Exception as e:
            logger.error(f"Error closing WebDriver: {e}")

def main():
    scraper = ESGNewsScraper()
    try:
        # Get company name from user
        company_name = input("Enter the company name to search for ESG news: ")
        
        # Search for articles
        articles = scraper.search_esg_news(company_name)
        
        # Display and save results
        if articles:
            print(f"\nFound {len(articles)} ESG news articles for {company_name}:\n")
            for idx, article in enumerate(articles, 1):
                print(f"{idx}. {article['title']}")
                print(f"   Date: {article['date']}")
                print(f"   Link: {article['link']}")
                print(f"   Content Preview: {article['content'][:200]}...\n")
            
            # Save results to file
            scraper.save_results(articles)
            print(f"Results saved to 'esg_news_results.txt'")
        else:
            print(f"No ESG news articles found for {company_name}")
    
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
    
    finally:
        # Ensure WebDriver is closed
        scraper.close()

if __name__ == "__main__":
    main()