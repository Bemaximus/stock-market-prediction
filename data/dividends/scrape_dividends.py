import numpy as np
import pandas as pd

from bs4 import BeautifulSoup
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
def get_upcoming_dividend_dates(html=None) -> pd.DataFrame:

	# get the html from marketbeat
	html = html if html else requests.get(upcoming_dividends_dates_url).content

	html = split_ticker_and_company(html)
	# convert the html to an easily manageable DataFrame
	upcoming_dividends_df = pd.read_html(html, parse_dates=True)[0]
	upcoming_dividends_df = upcoming_dividends_df[upcoming_dividends_df["Ticker"] != '']

	return upcoming_dividends_df

def get_dividends_by_date() -> pd.DataFrame:
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

def filter_ex_dividend_data(data_df) -> pd.DataFrame:
	pass



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

def split_ticker_and_company(html) -> str:

	# create BeautifulSoup object
	soup = BeautifulSoup(html, "lxml")
	table = soup.find('table')
	table_body = table.find('tbody')

	# iterate over all rows in the table
	rows = table_body.find_all('tr')
	for row in rows:

		# access the first column's entry
		ticker_and_company = row.find('td')
		
		# extract the ticker and company name, if applicable
		try:
			ticker = row.select_one('div .ticker-area').string or ''
			company = row.select_one('div .title-area').string or ''
		except:
			ticker = company = ''

		finally:
			# create new cells to store this data
			ticker_td = soup.new_tag('td')
			ticker_td.string = ticker

			company_td = soup.new_tag('td')
			company_td.string = company

			# append this data to the end of the table
			row.append(ticker_td)
			row.append(company_td)

	# add ticker and company columns to the thead
	table_head_row = table.select_one('thead tr')

	ticker_th = soup.new_tag("th")
	ticker_th.string = "Ticker"
	table_head_row.append(ticker_th)

	company_th = soup.new_tag("th")
	company_th.string = "Company Name"
	table_head_row.append(company_th)

	return str(table)

def get_lazy_ticker_from_company_name(agg_string):
	ticker, _ = re.findall(r"([A-Z]+)([A-Z].*)", agg_string)[0]
	if 1 <= len(ticker) <= 4:
		return ticker
	else:
		return None

if __name__ == "__main__":
	upcoming_dividends_df = get_dividends_by_date()
	upcoming_dividends_df.to_csv("all_entries_filtered.csv")