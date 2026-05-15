from playwright.sync_api import sync_playwright
import time

def run():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        
        print("Navigating to live-rvr...")
        page.goto("http://rvrcamd.imd.gov.in:5000/live-rvr")
        
        # Wait for the page to load
        page.wait_for_timeout(2000)
        
        print("Clicking dropdown to select airport...")
        # Since it's a combobox, we'll click it
        page.click("text=Select Airport")
        page.wait_for_timeout(1000)
        
        # Look for "New Delhi" or "Delhi"
        print("Clicking New Delhi Airport...")
        page.click("text=New Delhi Airport")
        
        print("Waiting for data to load...")
        page.wait_for_timeout(5000)
        
        # Dump the text
        text = page.locator("body").inner_text()
        print("--- Page text ---")
        print(text[:1500])
        print("-----------------")
        
        # We can also capture a screenshot to see what it looks like
        page.screenshot(path="data/realtime/rvr_screenshot.png")
        print("Saved screenshot to data/realtime/rvr_screenshot.png")
        
        browser.close()

if __name__ == "__main__":
    run()
