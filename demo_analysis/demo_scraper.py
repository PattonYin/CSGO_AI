from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
import time
import pandas as pd
from tqdm import tqdm
import re
from requests_html import HTMLSession
from pyppeteer import launch
import os
import patoolib
import logging


class Scraper:
    def __init__(self, the_map='de_dust2'):
        self.map = the_map
        self.match_links = []
        self.dates = []
        self.date_pattern = r"(?i)(?:January|February|March|April|May|June|July|August|September|October|November|December)\s\d{1,2}(?:st|nd|rd|th)?\s\d{4}"
        self.url = f"https://www.hltv.org/results?content=demo&map={self.map}"
        self.setup_driver()

    def setup_driver(self):
        path_to_driver = r"X:\chrome_webdriver\chromedriver-win64\chromedriver.exe"
        chrome_options = Options()
        chrome_options.add_argument("--headless")  # Run in headless mode
        self.driver = webdriver.Chrome(service=Service(path_to_driver), options=chrome_options)

    def run(self):
        self.driver.get(self.url)
        self.driver.implicitly_wait(10)  # Wait up to 10 seconds for elements to appear

        try:
            # Locate the button by its ID and click it
            cookie_button = self.driver.find_element(By.ID, "CybotCookiebotDialogBodyLevelButtonLevelOptinAllowAll")
            cookie_button.click()
            print("Cookie consent given!")
        except Exception as e:
            print("Error clicking the button:", e)


        # Use Selenium to find elements
        results_all = self.driver.find_elements(By.CSS_SELECTOR, '.results-all')[-1]
        per_day = results_all.find_elements(By.CSS_SELECTOR, '.results-sublist')
        for day in tqdm(per_day):            
            text_all = day.text
            date = re.findall(self.date_pattern, text_all)[0]

            links = day.find_elements(By.TAG_NAME, "a")
            for link in links:
                href = link.get_attribute('href')
                self.match_links.append(href)
                self.dates.append(date)
                
        # Clean up: close the browser window
        self.driver.quit()
        
    def output_links(self):
        df_out = pd.DataFrame(columns=["date", "match_links"])
        df_out["date"] = self.dates
        df_out["match_links"] = self.match_links
        df_out.to_csv(f"demo_analysis/demo/links/{self.map}_links.csv", index=False)
    
class LinkExtractor:
    def __init__(self):
        self.chromium_path = r"X:\chrome_webdriver\chrome-win64\chrome.exe"
        self.driver_path = r"X:\chrome_webdriver\chromedriver-win64\chromedriver.exe"  # Adjust path as necessary
        # self.setup_driver()

    def setup_driver(self):
        chrome_options = Options()
        chrome_options.binary_location = self.chromium_path
        # chrome_options.add_argument("--headless")  # Run in headless mode (no GUI)
        # chrome_options.add_argument("--disable-gpu")  # Disable GPU hardware acceleration
        # chrome_options.add_argument("--no-sandbox")  # Bypass OS security model
        # chrome_options.add_argument("--disable-dev-shm-usage")  # Overcome limited resource problems
        self.driver = webdriver.Chrome(service=Service(self.driver_path), options=chrome_options)

    def fetch_links_after_js(self, url):
        self.setup_driver()
        self.driver.get(url)
        self.driver.implicitly_wait(10)
        
        link_elements = self.driver.find_elements(By.CSS_SELECTOR, "a.stream-box")

        links = [element.get_attribute('data-demo-link') for element in link_elements]
        
        self.driver.quit()

        if len(links) == 0:
            return None
        return links[0]
    
    def fetch_all_links(self, df):
        # add column for the df
        df['demo_links'] = None
        for index, row in tqdm(df.iterrows(), total=len(df), desc="Extracting demo links"):
            demo_links = self.fetch_links_after_js(row['match_links'])
            print(f"the link: {demo_links}")
            df.at[index, 'demo_links'] = demo_links
            
        return df
    
class Downloader:
    def __init__(self):
        self.base_url = "https://hltv.org"
        
    def download_with_selenium(self, url):
        options = webdriver.ChromeOptions()
        options.add_experimental_option('prefs', {
            "download.default_directory": r"X:\code\CSGO_AI\demo_analysis\demo\demos_rar\de_dust2",
            "download.prompt_for_download": False,
            "download.directory_upgrade": True,
            "safebrowsing.enabled": True
        })
        driver = webdriver.Chrome(options=options)
        driver.get(url) 
        time.sleep(20)
        driver.quit()
        
    def download_all_links(self, df):
        for index, row in tqdm(df.iterrows(), total=len(df)):
            if row['demo_links'] is not None:
                url = self.base_url + row['demo_links']
                self.download_with_selenium(url)
            else:
                print(f"No demo link found for match: {row['match_links']}")
                
class ExtractRAR:
    def extract_rar(self, file_path, extract_to):
        if not os.path.exists(extract_to):
            os.makedirs(extract_to)
        try:
            patoolib.extract_archive(file_path, outdir=extract_to)
            print(f"Files extracted successfully to {extract_to}")
        except Exception as e:
            print(f"An error occurred while extracting files: {e}")
            
    def extract_all(self, folder_path, extract_to):
        rars = os.listdir(folder_path)
        for rar in tqdm(rars):
            self.extract_rar(os.path.join(folder_path, rar), extract_to)
            
    def cleanup(self, directory, suffix):
        # Create a full path pattern to match all files in the directory
        path_pattern = os.path.join(directory, '*')
        
        # List all files in the directory
        files = glob.glob(path_pattern)
        
        # Filter and delete files that do not end with the specified suffix
        for file in files:
            if not file.endswith(suffix):
                os.remove(file)
                print(f"Deleted: {file}")
            
if __name__ == '__main__':
    # scraper = Scraper()
    # scraper.run()
    # scraper.output_links()


    # Extractor = LinkExtractor()
    # df = pd.read_csv("demo_analysis/demo/links/de_dust2_links.csv")
    # df_out = Extractor.fetch_all_links(df)
    # df_out.to_csv("demo_analysis/demo/links/de_dust2_links.csv", index=False)
    
    # Downloader = Downloader()
    # df = pd.read_csv("demo_analysis/demo/links/de_dust2_links.csv")
    # Downloader.download_all_links(df)
    
    Extractor = ExtractRAR()
    Extractor.extract_all(r"X:\code\CSGO_AI\demo_analysis\demo\demos_rar\de_dust2", r"X:\code\CSGO_AI\demo_analysis\demo\demos_extracted\de_dust2")
    directory_path = r'X:\code\CSGO_AI\demo_analysis\demo\demos_extracted\de_dust2'  # Replace with the path to your directory
    file_suffix = 'dust2.dem'
    Extractor.cleanup(directory_path, file_suffix)