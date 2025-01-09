## This script was generated via Anthropic's Claude

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
import requests
import time
import os
import schedule
import datetime
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('avalanche_download.log'),
        logging.StreamHandler()
    ]
)

def setup_chrome_driver():
    """Configure Chrome driver with appropriate options"""
    chrome_options = Options()
    chrome_options.add_argument('--disable-gpu')
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    
    driver_path = "/Users/chrishyatt/Desktop/chromedriver-mac-arm64/chromedriver"
    service = Service(driver_path)
    
    return webdriver.Chrome(service=service, options=chrome_options)

def wait_for_export_completion(driver, timeout=300):
    """
    Wait for the export to complete by monitoring the progress elements
    Returns True when export is successful
    """
    start_time = time.time()
    last_percentage = 0
    
    while time.time() - start_time < timeout:
        try:
            # Check for success message first
            success_elements = driver.find_elements(By.CLASS_NAME, "page-title")
            if any("Data export successful" in element.text for element in success_elements):
                print("\nExport completed successfully!")
                return True
                
            # If not successful yet, check progress
            percentage_element = driver.find_element(By.CLASS_NAME, "percentage")
            message_element = driver.find_element(By.CLASS_NAME, "message")
            
            if percentage_element and message_element:
                current_percentage = int(percentage_element.text.strip('%'))
                if current_percentage > last_percentage:
                    print(f"\rProgress: {percentage_element.text} - {message_element.text}", end='', flush=True)
                    last_percentage = current_percentage
            
        except Exception as e:
            # If elements not found, might be transitioning between states
            pass
            
        time.sleep(1)
    
    print("\nTimeout waiting for export completion")
    return False

def download_csv():
    """Download CSV from the avalanche center website"""
    driver = setup_chrome_driver()
    wait = WebDriverWait(driver, 20)

    try:
        # Step 1: Navigate to the page
        print("Navigating to webpage...")
        driver.get("https://utahavalanchecenter.org/avalanches")
        
        # Step 2: Wait for and find the download button
        print("Waiting for download button...")
        download_button = wait.until(
            EC.presence_of_element_located((By.XPATH, '//a[@href="/avalanches/details/csv"]'))
        )
        
        # Step 3: Click the button
        print("Clicking download button...")
        download_button.click()
        
        # Step 4: Wait for export to complete while showing progress
        if not wait_for_export_completion(driver):
            print("Export process failed or timed out")
            return False
        
        time.sleep(2)  # Brief pause to ensure the download link is ready
        
        # Step 5: After successful export, get the download URL
        try:
            # Look for any download links that appeared after success
            download_links = driver.find_elements(By.TAG_NAME, "a")
            download_url = None
            
            for link in download_links:
                href = link.get_attribute('href')
                if href and 'csv' in href:
                    download_url = href
                    break
            
            if not download_url:
                # Fallback to default URL if no specific download link found
                base_url = "https://utahavalanchecenter.org"
                download_url = base_url + "/avalanches/details/csv"
            
            # Step 6: Download using requests with session cookies
            print(f"\nDownloading CSV from: {download_url}")
            cookies = driver.get_cookies()
            session = requests.Session()
            
            for cookie in cookies:
                session.cookies.set(cookie['name'], cookie['value'])
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'text/csv,application/csv,text/plain',
                'Referer': driver.current_url
            }
            
            response = session.get(download_url, headers=headers)
            
            if response.status_code == 200:
                output_file = "avalanches.csv"
                with open(output_file, "wb") as f:
                    f.write(response.content)
                
                # Verify the content looks like CSV
                with open(output_file, 'r') as f:
                    first_line = f.readline().strip()
                    if ',' in first_line and not '<html' in first_line.lower():
                        print(f"CSV downloaded successfully as {output_file}!")
                        print(f"File size: {os.path.getsize(output_file)} bytes")
                        return True
                    else:
                        print("Downloaded file doesn't appear to be CSV format")
                        return False
            else:
                print(f"Failed to download CSV. Status code: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"Error during download: {str(e)}")
            return False
            
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return False
        
    finally:
        driver.quit()

def create_backup():
    """Create backup of existing CSV if it exists"""
    output_file = "avalanches.csv"
    if os.path.exists(output_file):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = Path("backups")
        backup_dir.mkdir(exist_ok=True)
        backup_file = backup_dir / f"avalanches_{timestamp}.csv"
        os.rename(output_file, backup_file)
        logging.info(f"Created backup: {backup_file}")

def scheduled_download():
    """Wrapper function for the scheduled task"""
    logging.info("Starting scheduled download...")
    try:
        create_backup()
        success = download_csv()
        if success:
            logging.info("Scheduled download completed successfully")
        else:
            logging.error("Scheduled download failed")
    except Exception as e:
        logging.error(f"Error during scheduled download: {str(e)}")

def run_scheduler():
    """Set up and run the scheduler"""
    # Schedule the job to run every two weeks
    # schedule.every(2).weeks.at("02:00").do(scheduled_download)  # Runs at 2 AM
    schedule.every(14).days.at("02:00").do(scheduled_download)  # Runs at 2 AM

    logging.info("Scheduler started. Will download every 2 weeks.")
    
    while True:
        try:
            schedule.run_pending()
            time.sleep(60)  # Check every minute for pending tasks
        except Exception as e:
            logging.error(f"Scheduler error: {str(e)}")
            time.sleep(300)  # Wait 5 minutes before retrying if there's an error

if __name__ == "__main__":    
    # Comment out the following lines during testing
    logging.info("Starting avalanche data scheduler...")
    run_scheduler()