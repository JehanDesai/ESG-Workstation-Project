from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import pandas as pd
import time
import re
import os
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ESGCompanyScraper:
    def __init__(self, headless=True, timeout=15):
        """Initialize the ESG company-targeted scraper with browser settings"""
        self.options = Options()
        if headless:
            self.options.add_argument("--headless=new")  # Updated headless mode
        self.options.add_argument("--no-sandbox")
        self.options.add_argument("--disable-dev-shm-usage")
        self.options.add_argument("--disable-gpu")
        self.options.add_argument("--window-size=1920,1080")
        self.options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
        
        self.timeout = timeout  # Configurable timeout
        
        try:
            # Initialize the Chrome driver
            self.driver = webdriver.Chrome(
                service=Service(ChromeDriverManager().install()),
                options=self.options
            )
            self.driver.set_page_load_timeout(self.timeout)
            logger.info("Browser initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize browser: {e}")
            raise
        
        # Company name variations dictionary to handle different forms of company names
        self.company_variations = {
            "jpmorgan": ["jpmorgan", "jp morgan", "j.p. morgan", "jpm", "jpmc", "j.p. morgan chase", "jpmorgan chase"],
            "goldman sachs": ["goldman sachs", "goldman", "goldman sachs group", "gs"],
            "blackrock": ["blackrock", "black rock"],
            "morgan stanley": ["morgan stanley", "ms"],
            "citigroup": ["citigroup", "citi", "citibank"],
            "bank of america": ["bank of america", "bofa", "bac", "bank of america corporation"],
            "wells fargo": ["wells fargo", "wf", "wells"],
            "microsoft": ["microsoft", "msft", "microsoft corporation"],
            "apple": ["apple", "aapl", "apple inc", "apple inc."],
            "amazon": ["amazon", "amzn", "amazon.com", "amazon.com inc", "amazon inc"],
            "google": ["google", "googl", "goog", "alphabet", "alphabet inc"],
            "facebook": ["facebook", "meta", "fb", "meta platforms"],
            "tesla": ["tesla", "tsla", "tesla inc", "tesla motors"],
            "nvidia": ["nvidia", "nvda"],
            "exxonmobil": ["exxonmobil", "exxon mobil", "exxon", "xom"],
            "chevron": ["chevron", "cvx", "chevron corporation"],
            "shell": ["shell", "royal dutch shell", "rdsa"],
            "bp": ["bp", "british petroleum", "bp plc"],
            "unilever": ["unilever", "ul"],
            "nestle": ["nestle", "nestlé", "nsrgy"]
        }
        
    def normalize_company_name(self, company_name):
        """Convert company name to standard form and get variations"""
        company_lower = company_name.lower()
        
        # Find the standardized company name
        for standard_name, variations in self.company_variations.items():
            if company_lower in variations:
                return standard_name, variations
        
        # If not found in predefined list, use original name and simple variations
        variations = [company_lower, company_lower.replace(" ", "")]
        return company_lower, variations
    
    def safe_get(self, url, retries=3, retry_delay=2):
        """Safely navigate to a URL with retries"""
        for attempt in range(retries):
            try:
                logger.info(f"Navigating to: {url} (attempt {attempt+1}/{retries})")
                self.driver.get(url)
                return True
            except Exception as e:
                logger.warning(f"Failed to load {url}: {e}")
                if attempt < retries - 1:
                    logger.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    logger.error(f"Failed to load {url} after {retries} attempts")
                    return False
    
    def search_esg_today_for_company(self, company_name, max_pages=5):
        """Search ESG Today for articles about a specific company"""
        standard_name, name_variations = self.normalize_company_name(company_name)
        logger.info(f"Searching for articles about {standard_name.title()} (and variations)")
        
        articles = []
        
        # Method 1: Search using site's search functionality
        search_url = f"https://www.esgnews.com/?s={company_name.replace(' ', '+')}"
        logger.info(f"Searching ESG Today via: {search_url}")
        
        if not self.safe_get(search_url):
            logger.warning("Failed to load search URL, trying alternative approach")
        else:
            try:
                # Wait for search results
                WebDriverWait(self.driver, self.timeout).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "main#main, .search-results, .archive"))
                )
                
                # Take a screenshot for debugging (optional)
                self.driver.save_screenshot(f"{standard_name}_search_results.png")
                
                # Check if no results were found
                if "No results found" in self.driver.page_source or "Nothing Found" in self.driver.page_source:
                    logger.info("No search results found directly.")
                else:
                    # Multiple possible selectors to handle potential site changes
                    article_selectors = [
                        "article.post", 
                        ".post", 
                        ".article", 
                        ".entry",
                        ".search-post"
                    ]
                    
                    for selector in article_selectors:
                        search_articles = self.driver.find_elements(By.CSS_SELECTOR, selector)
                        if search_articles:
                            logger.info(f"Found {len(search_articles)} articles using selector: {selector}")
                            break
                    
                    for article in search_articles:
                        try:
                            # Multiple possible title selectors
                            title_element = None
                            for title_selector in ["h2.entry-title a", "h1.entry-title a", ".entry-title a", "h3 a"]:
                                try:
                                    title_element = article.find_element(By.CSS_SELECTOR, title_selector)
                                    break
                                except:
                                    continue
                            
                            if not title_element:
                                logger.warning("Could not find title element, skipping article")
                                continue
                                
                            title = title_element.text
                            link = title_element.get_attribute("href")
                            
                            # Get date if available
                            date = "Date not found"
                            for date_selector in ["time.entry-date", ".entry-date", ".date", ".published"]:
                                try:
                                    date = article.find_element(By.CSS_SELECTOR, date_selector).text
                                    break
                                except:
                                    continue
                            
                            # Get excerpt if available
                            excerpt = "Excerpt not available"
                            for excerpt_selector in ["div.entry-content p", ".excerpt", ".summary", ".entry-summary p"]:
                                try:
                                    excerpt = article.find_element(By.CSS_SELECTOR, excerpt_selector).text
                                    break
                                except:
                                    continue
                            
                            article_data = {
                                "title": title,
                                "date": date,
                                "link": link,
                                "excerpt": excerpt,
                                "source": "ESG Today",
                                "search_method": "direct_search",
                                "company": standard_name
                            }
                            
                            articles.append(article_data)
                            logger.info(f"Found article via search: {title}")
                            
                        except Exception as e:
                            logger.warning(f"Error extracting search result: {e}")
                    
                    logger.info(f"Found {len(search_articles)} articles via direct search.")
            except Exception as e:
                logger.error(f"Error during direct search: {e}")
        
        # Method 2: Browse recent news and filter for company mentions
        if len(articles) < 3:
            logger.info("Searching through recent ESG news pages...")
            
            for page in range(1, max_pages + 1):
                try:
                    # Try multiple URL patterns (site might have changed structure)
                    news_urls = [
                        f"https://www.esgnews.com/category/news/page/{page}/",
                        f"https://www.esgnews.com/news/page/{page}/",
                        f"https://www.esgnews.com/page/{page}/"
                    ]
                    
                    for news_url in news_urls:
                        logger.info(f"Checking page {page}: {news_url}")
                        if not self.safe_get(news_url):
                            continue
                        
                        # Check if we got a valid page
                        if "Page not found" in self.driver.page_source:
                            logger.warning(f"Page not found: {news_url}")
                            continue
                        
                        # Wait for the articles to load with multiple possible selectors
                        for selector in ["article.post", ".post", ".article", ".entry"]:
                            try:
                                WebDriverWait(self.driver, 8).until(
                                    EC.presence_of_all_elements_located((By.CSS_SELECTOR, selector))
                                )
                                logger.info(f"Found articles using selector: {selector}")
                                break
                            except:
                                continue
                        
                        # Get page source for preliminary filtering
                        page_source = self.driver.page_source.lower()
                        
                        # Check if any variation of the company name appears on the page
                        company_mentioned = any(variation in page_source for variation in name_variations)
                        
                        if company_mentioned:
                            logger.info(f"Company mentioned on page {page}")
                            
                            # Try multiple selectors for articles
                            page_articles = []
                            for selector in ["article.post", ".post", ".article", ".entry"]:
                                page_articles = self.driver.find_elements(By.CSS_SELECTOR, selector)
                                if page_articles:
                                    break
                            
                            for article in page_articles:
                                try:
                                    # Get article text for company name matching
                                    article_text = article.text.lower()
                                    
                                    # Check if article mentions the company
                                    if any(variation in article_text for variation in name_variations):
                                        # Extract article data with multiple possible selectors
                                        title_element = None
                                        for title_selector in ["h2.entry-title a", "h1.entry-title a", ".entry-title a", "h3 a"]:
                                            try:
                                                title_element = article.find_element(By.CSS_SELECTOR, title_selector)
                                                break
                                            except:
                                                continue
                                        
                                        if not title_element:
                                            logger.warning("Could not find title element, skipping article")
                                            continue
                                            
                                        title = title_element.text
                                        link = title_element.get_attribute("href")
                                        
                                        # Get date if available
                                        date = "Date not found"
                                        for date_selector in ["time.entry-date", ".entry-date", ".date", ".published"]:
                                            try:
                                                date = article.find_element(By.CSS_SELECTOR, date_selector).text
                                                break
                                            except:
                                                continue
                                        
                                        # Get excerpt if available
                                        excerpt = "Excerpt not available"
                                        for excerpt_selector in ["div.entry-content p", ".excerpt", ".summary", ".entry-summary p"]:
                                            try:
                                                excerpt = article.find_element(By.CSS_SELECTOR, excerpt_selector).text
                                                break
                                            except:
                                                continue
                                        
                                        article_data = {
                                            "title": title,
                                            "date": date,
                                            "link": link,
                                            "excerpt": excerpt,
                                            "source": "ESG Today",
                                            "search_method": "browse_filter",
                                            "company": standard_name
                                        }
                                        
                                        # Skip if already found via search
                                        if not any(a["link"] == link for a in articles):
                                            articles.append(article_data)
                                            logger.info(f"Found article via browsing: {title}")
                                
                                except Exception as e:
                                    logger.warning(f"Error processing article during browsing: {e}")
                        
                        # Break out of URL loop if we've successfully processed this page
                        break
                            
                    # Sleep to avoid hammering the server
                    time.sleep(3)
                    
                except Exception as e:
                    logger.error(f"Error during browsing page {page}: {e}")
        
        # Method 3: Try direct Google search with site: operator
        if len(articles) < 2:
            try:
                logger.info("Trying Google search as fallback method")
                google_search_url = f"https://www.google.com/search?q=site:esgtoday.com+{company_name.replace(' ', '+')}"
                
                if self.safe_get(google_search_url):
                    # Look for search results
                    search_results = self.driver.find_elements(By.CSS_SELECTOR, "div.g")
                    
                    for result in search_results[:5]:  # Process top 5 results
                        try:
                            link_element = result.find_element(By.CSS_SELECTOR, "a")
                            link = link_element.get_attribute("href")
                            
                            # Check if link is from ESG Today
                            if "esgtoday.com" in link:
                                # Get title
                                title_element = result.find_element(By.CSS_SELECTOR, "h3")
                                title = title_element.text
                                
                                # Get snippet
                                snippet = "Snippet not available"
                                try:
                                    snippet_element = result.find_element(By.CSS_SELECTOR, "div.VwiC3b")
                                    snippet = snippet_element.text
                                except:
                                    pass
                                
                                article_data = {
                                    "title": title,
                                    "date": "Date not found",  # We'll get this when visiting the page
                                    "link": link,
                                    "excerpt": snippet,
                                    "source": "ESG Today",
                                    "search_method": "google_search",
                                    "company": standard_name
                                }
                                
                                # Skip if already found
                                if not any(a["link"] == link for a in articles):
                                    articles.append(article_data)
                                    logger.info(f"Found article via Google search: {title}")
                        except Exception as e:
                            logger.warning(f"Error processing Google search result: {e}")
            except Exception as e:
                logger.error(f"Error during Google search fallback: {e}")
        
        # Method 4: Try ESG News API (another approach if available)
        # This could be implemented here...
        
        logger.info(f"Total articles found for {standard_name.title()}: {len(articles)}")
        return articles
        
    def search_multiple_esg_sites(self, company_name, sites=None):
        """Search multiple ESG sites for company mentions"""
        if sites is None:
            sites = [
                {"name": "ESG Today", "function": self.search_esg_today_for_company},
                # Add more sites here
            ]
        
        all_articles = []
        
        for site in sites:
            try:
                logger.info(f"\nSearching {site['name']} for articles about {company_name}...")
                site_articles = site["function"](company_name)
                all_articles.extend(site_articles)
                logger.info(f"Found {len(site_articles)} articles on {site['name']}")
            except Exception as e:
                logger.error(f"Error searching {site['name']}: {e}")
        
        return all_articles
    
    def get_article_content(self, articles):
        """Scrape the full content of articles from their URLs"""
        logger.info(f"Retrieving full content for {len(articles)} articles...")
        
        for i, article in enumerate(articles):
            try:
                logger.info(f"[{i+1}/{len(articles)}] Getting content: {article['title']}")
                
                if not self.safe_get(article['link']):
                    logger.warning(f"Could not load article page: {article['link']}")
                    article['full_content'] = "Content extraction failed - page could not be loaded"
                    article['mention_count'] = 0
                    article['relevant_paragraphs'] = []
                    continue
                
                # Wait for the article content to load - try multiple selectors
                content_element = None
                for content_selector in ["div.entry-content", "article", ".post-content", ".article-content"]:
                    try:
                        WebDriverWait(self.driver, self.timeout).until(
                            EC.presence_of_element_located((By.CSS_SELECTOR, content_selector))
                        )
                        content_element = self.driver.find_element(By.CSS_SELECTOR, content_selector)
                        logger.info(f"Found content using selector: {content_selector}")
                        break
                    except:
                        continue
                
                if not content_element:
                    logger.warning(f"Could not find content element for article: {article['title']}")
                    article['full_content'] = "Content extraction failed - content element not found"
                    article['mention_count'] = 0
                    article['relevant_paragraphs'] = []
                    continue
                
                # Extract paragraphs
                paragraphs = content_element.find_elements(By.TAG_NAME, "p")
                content = "\n".join([p.text for p in paragraphs if p.text])
                
                # If content is too short, try other methods
                if len(content) < 100:
                    logger.warning(f"Content seems too short ({len(content)} chars), trying alternative extraction")
                    content = content_element.text
                
                # Extract categories/tags if available
                try:
                    categories = []
                    for cat_selector in ["span.cat-links a", ".categories a", ".tags a", ".post-categories a"]:
                        cat_elements = self.driver.find_elements(By.CSS_SELECTOR, cat_selector)
                        if cat_elements:
                            categories = [cat.text for cat in cat_elements]
                            break
                    article['categories'] = categories
                except:
                    article['categories'] = []
                
                # Try to extract the date if it wasn't found earlier
                if article['date'] == "Date not found":
                    for date_selector in ["time.entry-date", ".entry-date", ".date", ".published", "time[datetime]"]:
                        try:
                            date_element = self.driver.find_element(By.CSS_SELECTOR, date_selector)
                            article['date'] = date_element.text or date_element.get_attribute("datetime")
                            if article['date']:
                                break
                        except:
                            continue
                
                # Add content to the article dictionary
                article['full_content'] = content
                
                # Check for company mentions in the full content
                standard_name, name_variations = self.normalize_company_name(article['company'])
                
                # Count mentions of the company
                mention_count = 0
                for variation in name_variations:
                    # Use word boundaries for more accurate counting
                    pattern = r'\b' + re.escape(variation) + r'\b'
                    mentions = len(re.findall(pattern, content.lower(), re.IGNORECASE))
                    mention_count += mentions
                
                article['mention_count'] = mention_count
                
                # Extract relevant paragraphs that mention the company
                relevant_paragraphs = []
                paragraphs_text = [p.text for p in paragraphs if p.text]
                
                for paragraph in paragraphs_text:
                    if any(variation in paragraph.lower() for variation in name_variations):
                        relevant_paragraphs.append(paragraph)
                
                article['relevant_paragraphs'] = relevant_paragraphs
                
                # Sleep to avoid hammering the server
                time.sleep(3)
                
            except Exception as e:
                logger.error(f"Error getting content for article {i+1}: {e}")
                article['full_content'] = f"Content extraction failed: {str(e)}"
                article['mention_count'] = 0
                article['relevant_paragraphs'] = []
        
        return articles
    
    def analyze_esg_themes(self, articles):
        """Analyze ESG themes in the articles about the company"""
        # Common ESG themes/keywords to look for
        esg_themes = {
            "environmental": [
                "climate", "carbon", "emissions", "renewable", "sustainable", 
                "green", "environment", "pollution", "waste", "recycling",
                "biodiversity", "conservation", "energy efficiency", "net zero",
                "greenhouse gas", "ghg", "carbon neutral", "carbon footprint"
            ],
            "social": [
                "diversity", "inclusion", "equity", "human rights", "labor", 
                "community", "health", "safety", "wellbeing", "stakeholder",
                "employee", "workforce", "social impact", "ethical", "dei",
                "equal opportunity", "gender", "racial", "discrimination", 
                "accessibility", "social justice", "fair trade"
            ],
            "governance": [
                "board", "executive", "compensation", "transparency", "disclosure", 
                "compliance", "risk management", "ethics", "corruption", "bribery",
                "accountability", "shareholder", "voting", "audit", "regulatory",
                "corporate governance", "oversight", "fiduciary", "code of conduct",
                "conflict of interest", "whistleblower", "diversity targets"
            ]
        }
        
        logger.info(f"Analyzing ESG themes in {len(articles)} articles")
        
        for article in articles:
            if 'full_content' not in article or not article['full_content']:
                logger.warning(f"Skipping theme analysis for article with no content: {article.get('title', 'Unknown')}")
                continue
                
            content = article['full_content'].lower()
            
            # Analyze ESG themes
            theme_counts = {
                "environmental": 0,
                "social": 0,
                "governance": 0
            }
            
            theme_keywords = {
                "environmental": [],
                "social": [],
                "governance": []
            }
            
            for theme, keywords in esg_themes.items():
                for keyword in keywords:
                    # Use word boundaries for more accurate matching
                    pattern = r'\b' + re.escape(keyword) + r'\b'
                    matches = re.findall(pattern, content)
                    count = len(matches)
                    if count > 0:
                        theme_counts[theme] += count
                        theme_keywords[theme].append(keyword)
            
            # Add theme analysis to article
            article['esg_theme_counts'] = theme_counts
            article['esg_keywords'] = {
                theme: list(set(keywords)) for theme, keywords in theme_keywords.items()
            }
            
            # Determine primary ESG focus
            max_theme = max(theme_counts.items(), key=lambda x: x[1])
            article['primary_esg_focus'] = max_theme[0] if max_theme[1] > 0 else "undetermined"
            
            logger.info(f"Analysis for '{article['title']}': Primary focus: {article['primary_esg_focus']}, E:{theme_counts['environmental']}, S:{theme_counts['social']}, G:{theme_counts['governance']}")
        
        return articles
    
    def save_to_json(self, articles, company_name, filename=None):
        """Save the company-specific ESG articles to a JSON file"""
        if not articles:
            logger.warning(f"No articles to save for {company_name}")
            return None
            
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_company = company_name.replace(" ", "_").lower()
            filename = f"{safe_company}_esg_articles_{timestamp}.json"
        
        pd.DataFrame(articles).to_json(filename, orient='records', indent=4)
        logger.info(f"Saved {len(articles)} articles about {company_name} to {filename}")
        return filename
    
    def add_sustainable_finance_data(self, articles):
        """Add sustainable finance specific data analysis"""
        # Keywords specific to sustainable finance
        sustainable_finance_keywords = [
            "green bond", "sustainable bond", "social bond", 
            "sustainability-linked", "transition finance", "esg fund",
            "impact investing", "socially responsible", "sri", 
            "sustainable finance", "green finance", "climate finance",
            "net zero financing", "carbon neutral investment", "stranded assets"
        ]
        
        for article in articles:
            if 'full_content' not in article or not article['full_content']:
                continue
                
            content = article['full_content'].lower()
            
            # Count sustainable finance mentions
            finance_mentions = 0
            finance_keywords_found = []
            
            for keyword in sustainable_finance_keywords:
                pattern = r'\b' + re.escape(keyword) + r'\b'
                matches = re.findall(pattern, content)
                if matches:
                    finance_mentions += len(matches)
                    finance_keywords_found.append(keyword)
            
            article['sustainable_finance_mentions'] = finance_mentions
            article['sustainable_finance_keywords'] = list(set(finance_keywords_found))
            
            # Flag if article focuses on sustainable finance
            article['sustainable_finance_focus'] = finance_mentions > 3
        
        return articles
    
    def search_alternative_esg_sources(self, company_name, max_results=5):
        """Search alternative ESG news sources as a fallback"""
        # List of alternative ESG news sites
        alternative_sources = [
            {"name": "Responsible Investor", "url": f"https://www.responsible-investor.com/search-results?searchterm={company_name.replace(' ', '+')}"},
            {"name": "ESG Investor", "url": f"https://www.esginvestor.net/?s={company_name.replace(' ', '+')}"},
            {"name": "Sustainable Views", "url": f"https://www.sustainableviews.com/?s={company_name.replace(' ', '+')}"}
        ]
        
        results = []
        
        for source in alternative_sources:
            try:
                logger.info(f"Searching alternative source: {source['name']}")
                
                if not self.safe_get(source['url']):
                    continue
                
                # Wait for page to load and search for articles
                time.sleep(3)
                
                # Try multiple selectors for article elements
                article_elements = []
                for selector in ["article", ".post", ".result", ".search-result", ".article"]:
                    article_elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                    if article_elements:
                        break
                
                for i, article in enumerate(article_elements[:max_results]):
                    try:
                        # Extract article data
                        title_element = None
                        for title_selector in ["h2 a", "h3 a", ".title a", ".entry-title a"]:
                            try:
                                title_element = article.find_element(By.CSS_SELECTOR, title_selector)
                                break
                            except:
                                continue
                                
                        if not title_element:
                            continue
                            
                        title = title_element.text
                        link = title_element.get_attribute("href")
                        
                        # Extract date if available
                        date = "Date not found"
                        for date_selector in [".date", "time", ".meta-date", ".published"]:
                            try:
                                date_element = article.find_element(By.CSS_SELECTOR, date_selector)
                                date = date_element.text
                                break
                            except:
                                continue
                                
                        # Extract excerpt
                        excerpt = "Excerpt not available"
                        for excerpt_selector in [".excerpt", ".summary", "p", ".content p"]:
                            try:
                                excerpt_element = article.find_element(By.CSS_SELECTOR, excerpt_selector)
                                excerpt = excerpt_element.text
                                break
                            except:
                                continue
                                
                        # Add to results
                        results.append({
                            "title": title,
                            "date": date,
                            "link": link,
                            "excerpt": excerpt,
                            "source": source["name"],
                            "search_method": "alternative_source",
                            "company": company_name
                        })
                        
                    except Exception as e:
                        logger.warning(f"Error processing alternative source article: {e}")
                
                # Break after successful search to avoid too many requests
                if results:
                    break
                    
            except Exception as e:
                logger.error(f"Error searching alternative source {source['name']}: {e}")
                
        logger.info(f"Found {len(results)} articles from alternative sources")
        return results
        
    def detect_sentiment(self, articles):
        """Basic sentiment analysis on article content"""
        # Simple sentiment dictionaries
        positive_words = [
            "positive", "success", "innovation", "leadership", "achieve", 
            "improve", "progress", "sustainable", "responsible", "ethical",
            "benefit", "advance", "growth", "excellence", "quality",
            "commitment", "partnership", "transparency", "trust", "award"
        ]
        
        negative_words = [
            "negative", "fail", "risk", "concern", "issue", "problem", 
            "controversy", "violation", "penalty", "fine", "litigation",
            "criticism", "damage", "greenwashing", "investigation", "scandal",
            "warning", "insufficient", "dangerous", "harmful", "misleading"
        ]
        
        for article in articles:
            if 'full_content' not in article or not article['full_content']:
                continue
                
            content = article['full_content'].lower()
            
            # Count positive and negative words
            positive_count = sum(content.count(" " + word + " ") for word in positive_words)
            negative_count = sum(content.count(" " + word + " ") for word in negative_words)
            
            # Calculate simple sentiment score
            total = positive_count + negative_count
            if total > 0:
                sentiment_score = (positive_count - negative_count) / total
            else:
                sentiment_score = 0
                
            # Add sentiment data to article
            article['sentiment_score'] = round(sentiment_score, 2)
            article['sentiment'] = "positive" if sentiment_score > 0.1 else ("negative" if sentiment_score < -0.1 else "neutral")
            article['positive_words'] = positive_count
            article['negative_words'] = negative_count
            
        return articles
        
    def extract_quotes(self, articles):
        """Extract quotes from articles, especially from company executives"""
        for article in articles:
            if 'full_content' not in article or not article['full_content']:
                continue
                
            content = article['full_content']
            company_name = article['company']
            
            # Find quotes using various patterns
            quotes = []
            
            # Pattern 1: Standard quotes with attribution
            pattern1 = r'"([^"]+)"\s*,?\s*(?:said|says|according to|commented|noted|explained|added)\s+([^,.]+)'
            matches1 = re.findall(pattern1, content)
            for match in matches1:
                quotes.append({
                    "text": match[0],
                    "speaker": match[1],
                    "pattern": "attribution"
                })
                
            # Pattern 2: Attribution followed by quotes
            pattern2 = r'(?:said|says|according to|commented|noted|explained|added)\s+([^,.]+)[,.]?\s*"([^"]+)"'
            matches2 = re.findall(pattern2, content)
            for match in matches2:
                quotes.append({
                    "text": match[1],
                    "speaker": match[0],
                    "pattern": "preceding_attribution"
                })
                
            # Filter quotes related to the company
            standard_name, name_variations = self.normalize_company_name(company_name)
            
            company_quotes = []
            for quote in quotes:
                # Check if quote is from or about the company
                is_company_quote = False
                
                # Check if speaker is from the company
                for variation in name_variations:
                    if variation in quote["speaker"].lower():
                        is_company_quote = True
                        break
                        
                # Check if quote mentions the company
                if not is_company_quote:
                    for variation in name_variations:
                        if variation in quote["text"].lower():
                            is_company_quote = True
                            break
                            
                if is_company_quote:
                    company_quotes.append(quote)
                    
            article['quotes'] = company_quotes
            article['quote_count'] = len(company_quotes)
            
        return articles
        
    def cleanup(self):
        """Clean up resources"""
        try:
            logger.info("Cleaning up resources")
            self.driver.quit()
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
            
    def generate_report(self, articles, company_name, output_dir=None):
        """Generate a comprehensive report of ESG mentions for the company"""
        if not articles:
            logger.warning(f"No articles to generate report for {company_name}")
            return None
            
        if not output_dir:
            output_dir = os.getcwd()
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_company = company_name.replace(" ", "_").lower()
        filename = os.path.join(output_dir, f"{safe_company}_esg_report_{timestamp}.html")
        
        # Aggregate ESG theme data
        theme_totals = {"environmental": 0, "social": 0, "governance": 0}
        primary_focus_count = {"environmental": 0, "social": 0, "governance": 0, "undetermined": 0}
        sentiment_count = {"positive": 0, "neutral": 0, "negative": 0}
        
        for article in articles:
            # Count primary ESG focus
            if 'primary_esg_focus' in article:
                primary_focus_count[article['primary_esg_focus']] = primary_focus_count.get(article['primary_esg_focus'], 0) + 1
                
            # Count ESG theme mentions
            if 'esg_theme_counts' in article and isinstance(article['esg_theme_counts'], dict):
                for theme, count in article['esg_theme_counts'].items():
                    theme_totals[theme] += count
                    
            # Count sentiment
            if 'sentiment' in article:
                sentiment_count[article['sentiment']] = sentiment_count.get(article['sentiment'], 0) + 1
        
        # Generate HTML report
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>ESG Report for {company_name}</title>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; margin: 0; padding: 20px; color: #333; }}
                h1, h2, h3 {{ color: #2c3e50; }}
                .container {{ max-width: 1200px; margin: 0 auto; }}
                .summary {{ background-color: #f9f9f9; padding: 20px; border-radius: 5px; margin-bottom: 20px; }}
                .chart {{ height: 300px; margin: 20px 0; }}
                .article {{ border-bottom: 1px solid #eee; padding: 15px 0; }}
                .article h3 {{ margin-bottom: 5px; }}
                .article-meta {{ color: #7f8c8d; font-size: 0.9em; margin-bottom: 10px; }}
                .article-excerpt {{ margin-bottom: 10px; }}
                .positive {{ color: #27ae60; }}
                .negative {{ color: #c0392b; }}
                .neutral {{ color: #7f8c8d; }}
                .badge {{ display: inline-block; padding: 3px 8px; border-radius: 3px; font-size: 0.8em; margin-right: 5px; color: white; }}
                .badge.environmental {{ background-color: #2ecc71; }}
                .badge.social {{ background-color: #3498db; }}
                .badge.governance {{ background-color: #9b59b6; }}
                .badge.undetermined {{ background-color: #95a5a6; }}
                table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
                th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #f2f2f2; }}
                .quote {{ font-style: italic; padding: 10px; border-left: 3px solid #3498db; margin: 10px 0; background-color: #f8f9fa; }}
                .quote-speaker {{ font-weight: bold; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>ESG Report for {company_name}</h1>
                <p>Generated on {datetime.now().strftime("%B %d, %Y at %H:%M")}</p>
                
                <div class="summary">
                    <h2>Summary</h2>
                    <p>Total articles analyzed: <strong>{len(articles)}</strong></p>
                    <p>ESG Theme Distribution:</p>
                    <ul>
                        <li><span class="badge environmental">E</span> Environmental: {theme_totals['environmental']} mentions ({primary_focus_count.get('environmental', 0)} articles primarily focused)</li>
                        <li><span class="badge social">S</span> Social: {theme_totals['social']} mentions ({primary_focus_count.get('social', 0)} articles primarily focused)</li>
                        <li><span class="badge governance">G</span> Governance: {theme_totals['governance']} mentions ({primary_focus_count.get('governance', 0)} articles primarily focused)</li>
                    </ul>
                    <p>Sentiment Analysis:</p>
                    <ul>
                        <li class="positive">Positive: {sentiment_count.get('positive', 0)} articles</li>
                        <li class="neutral">Neutral: {sentiment_count.get('neutral', 0)} articles</li>
                        <li class="negative">Negative: {sentiment_count.get('negative', 0)} articles</li>
                    </ul>
                </div>
                
                <h2>Key ESG Articles</h2>
        """
        
        # Sort articles by mention count and add to report
        sorted_articles = sorted(articles, key=lambda x: x.get('mention_count', 0), reverse=True)
        
        for article in sorted_articles[:10]:  # Top 10 articles by mention count
            primary_focus = article.get('primary_esg_focus', 'undetermined')
            sentiment = article.get('sentiment', 'neutral')
            
            html_content += f"""
                <div class="article">
                    <h3><a href="{article['link']}" target="_blank">{article['title']}</a></h3>
                    <div class="article-meta">
                        {article.get('date', 'Date not found')} | Source: {article.get('source', 'Unknown')} | 
                        Company mentions: {article.get('mention_count', 0)} | 
                        <span class="badge {primary_focus}">{primary_focus.upper()[0]}</span>
                        <span class="{sentiment}">Sentiment: {sentiment}</span>
                    </div>
                    <div class="article-excerpt">{article.get('excerpt', 'No excerpt available')}</div>
            """
            
            # Add ESG keywords if available
            if 'esg_keywords' in article and isinstance(article['esg_keywords'], dict):
                html_content += "<div class='keywords'><strong>ESG Keywords:</strong> "
                for theme, keywords in article['esg_keywords'].items():
                    if keywords:
                        html_content += f"<span class='badge {theme}'>{theme[0].upper()}</span> {', '.join(keywords)} "
                html_content += "</div>"
                
            # Add quotes if available
            if 'quotes' in article and article['quotes']:
                html_content += "<div class='quotes'><strong>Relevant Quotes:</strong>"
                for quote in article['quotes'][:3]:  # Show up to 3 quotes
                    html_content += f"""
                        <div class="quote">
                            "{quote['text']}" - <span class="quote-speaker">{quote['speaker']}</span>
                        </div>
                    """
                html_content += "</div>"
                
            html_content += "</div>"
            
        # Complete the HTML
        html_content += """
                <h2>Methodology</h2>
                <p>This report was generated by automatically scraping and analyzing ESG-related articles. The analysis includes keyword matching for ESG themes, basic sentiment analysis, and extraction of relevant quotes.</p>
            </div>
        </body>
        </html>
        """
        
        # Write the report to file
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(html_content)
            logger.info(f"Generated ESG report at {filename}")
            return filename
        except Exception as e:
            logger.error(f"Error generating report: {e}")
            return None
    
    def run_complete_analysis(self, company_name):
        """Run a complete ESG analysis workflow for a company"""
        try:
            logger.info(f"Starting complete ESG analysis for {company_name}")
            
            # Step 1: Search for articles
            articles = self.search_esg_today_for_company(company_name)
            
            # Step 2: If not enough articles, try alternative sources
            if len(articles) < 3:
                logger.info(f"Not enough articles from primary source, searching alternatives")
                alt_articles = self.search_alternative_esg_sources(company_name)
                articles.extend(alt_articles)
            
            if not articles:
                logger.warning(f"No articles found for {company_name}")
                return None
                
            # Step 3: Get article content
            articles = self.get_article_content(articles)
            
            # Step 4: Analyze ESG themes
            articles = self.analyze_esg_themes(articles)
            
            # Step 5: Add sustainable finance data
            articles = self.add_sustainable_finance_data(articles)
            
            # Step 6: Add sentiment analysis
            articles = self.detect_sentiment(articles)
            
            # Step 7: Extract quotes
            articles = self.extract_quotes(articles)
            
            # Step 8: Save results
            json_file = self.save_to_json(articles, company_name)
            
            # Step 9: Generate report
            report_file = self.generate_report(articles, company_name)
            
            logger.info(f"Complete analysis finished for {company_name}")
            logger.info(f"Results saved to: {json_file}, {report_file}")
            
            return {
                "articles": articles,
                "json_file": json_file,
                "report_file": report_file,
                "article_count": len(articles)
            }
            
        except Exception as e:
            logger.error(f"Error during complete analysis: {e}")
            return None
        finally:
            self.cleanup()

if __name__ == "__main__":
    try:
        companies = ["BlackRock", "JPMorgan Chase", "Microsoft"]
        
        for company in companies:
            logger.info(f"\n{'='*50}\nProcessing {company}\n{'='*50}")
            scraper = ESGCompanyScraper(headless=True)
            result = scraper.run_complete_analysis(company)
            
            if result:
                logger.info(f"Analysis complete for {company}. Found {result['article_count']} articles.")
            else:
                logger.error(f"Analysis failed for {company}")
                
            # Allow some time between companies to avoid rate limiting
            time.sleep(10)
            
    except Exception as e:
        logger.critical(f"Fatal error in main program: {e}")