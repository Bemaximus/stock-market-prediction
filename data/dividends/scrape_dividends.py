import numpy as np
import pandas as pd

# from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.support.ui import Select
# from selenium.webdriver.common.keys import Keys
import time
import lxml
import requests
import re

# driver = webdriver.Firefox()
driver = None
week_selector = None
upcoming_dividends_dates_url = "https://www.marketbeat.com/dividends/ex-dividend-date-list/"


### Function that returns a DataFrame of dividend dates from a given url
def get_upcoming_dividend_dates(html=None):

	# get the html from marketbeat
	response = html if html else requests.get(upcoming_dividends_dates_url).content

	# convert the html to an easily manageable DataFrame
	upcoming_dividends_df = pd.read_html(response)[0]
	upcoming_dividends_df = upcoming_dividends_df[upcoming_dividends_df["Company"].str.len() < 100]

	return upcoming_dividends_df

def get_dividends_by_date():
	"""

	"""

	# start browser
	global driver, week_selector
	if driver == None:
		driver = webdriver.Firefox()

	# navigate to page and identify the week dropdown
	response = driver.get(upcoming_dividends_dates_url)
	week_selector = Select(driver.find_element_by_id("cphPrimaryContent_ddlWeek"))

	# close all popups
	driver.execute_script("closeIframeModal()") # The big green square notification
	# clickButton("onesignal-slidedown-cancel-button") # The notifications "cancel" button

	# initialize the DataFrame
	total_df = None
	prev_entries = None
	try:
		# iterate over all dropdown options
		dropdown_index = 0
		while True:
			print(f"Option {dropdown_index}")

			# select the new dropdown option
			week_selector = Select(driver.find_element_by_id("cphPrimaryContent_ddlWeek"))
			week_selector.select_by_index(dropdown_index)

			new_entries = get_upcoming_dividend_dates(html=driver.page_source)
			# print(new_entries.loc[0, "Company"], prev_entries.loc[0, "Company"])
			while prev_entries is not None and new_entries.loc[0, "Company"] == prev_entries.loc[0, "Company"]:
				time.sleep(0.5)
				new_entries = get_upcoming_dividend_dates(html=driver.page_source)

			total_df = total_df.append(new_entries) if total_df is not None else new_entries

			prev_entries = new_entries
			dropdown_index += 1
	except Exception as e:
		print(e)
		# no more dropdown options
		return total_df

	total_df["Ticker"] = total_df["Company"].apply(get_lazy_ticker_from_company_name)

	return total_df


def clickButton(b_id, tries=0):
	global driver
	try:
		driver.find_element_by_css_selector(f"button.{b_id}").click()
	except Exception:
		if(tries > 100):
			raise Exception("Wifi is Terrible!")
		time.sleep(0.5)
		# print("clickDelay", b_id)
		clickButton(b_id=b_id, tries=tries+1)

def get_lazy_ticker_from_company_name(agg_string: str):
	ticker, _ = re.findall(r"([A-Z]+)([A-Z].*)", agg_string)[0]
	if 1 <= len(ticker) <= 4:
		return ticker
	else:
		return None

if __name__ == "__main__":
	upcoming_dividends_df = get_dividends_by_date()
	upcoming_dividends_df.to_csv("all_entries_filtered.csv")