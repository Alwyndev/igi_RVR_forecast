from playwright.sync_api import sync_playwright
import time
import json

def run():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        
        print("Navigating to live-rvr...")
        page.goto("http://rvrcamd.imd.gov.in:5000/live-rvr")
        
        # Wait a bit for websocket data to load
        print("Waiting for data to render...")
        page.wait_for_timeout(5000)
        
        # Look for table or specific text
        text = page.locator("body").inner_text()
        print("--- Page text ---")
        print(text[:1000])
        print("-----------------")
        
        # Let's see if we can find all RVR values. We are looking for TDZ, MID, BEG etc.
        # Maybe let's just dump all text so we can parse it
        with open("data/realtime/rvr_page_text.txt", "w", encoding="utf-8") as f:
            f.write(text)
            
        print("Saved to data/realtime/rvr_page_text.txt")
        browser.close()

if __name__ == "__main__":
    run()
