import numpy as np
import pandas as pd

from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support.ui import Select
# from selenium.webdriver.common.keys import Keys
import time
import lxml
import requests
import re

driver = None
base_url = "https://www.thestreet.com/dividends/index.html"
ultimate_df = pd.DataFrame()

def get_all_dividends():
	
	# initialize Firefox
	global driver
	if driver == None:
		driver = webdriver.Firefox()

	# navigate to the website
	driver.get(base_url)

	# find latest dividend dates and work your way back in time
	find_latest_dividend()

	while select_days_on_calendar():
		shift_calendar_left_three()


def find_latest_dividend():

	global driver
	assert driver != None

	# get all days where calendar has dividends
	all_dividend_days = driver.find_elements_by_css_selector(
		"#cal1Container .selected.selectable")

	# reach a set of three months where no dividend dates exist
	while len(all_dividend_days) > 0:
		shift_calendar_right_one()
		all_dividend_days = driver.find_elements_by_css_selector(
			"#cal1Container .selected.selectable")		

	# move back three months
	shift_calendar_left_three()

def shift_calendar_left_three():

	for i in range(3):
		shift_calendar_left_one()

def shift_calendar_left_one():

	global driver
	assert driver != None

	left_button = driver.find_element_by_css_selector("#cal1Container .calheader .calnavleft")

	left_button.click()

def shift_calendar_right_one():

	global driver
	assert driver != None

	right_button = driver.find_element_by_css_selector("#cal1Container .calheader .calnavright")

	right_button.click()

def select_days_on_calendar() -> bool:

	global driver
	assert driver != None

	all_days = driver.find_elements_by_css_selector("#cal1Container .selected.selectable")
	if len(all_days) == 0:
		return False

	for i in range(len(all_days)):
		day = driver.find_elements_by_css_selector("#cal1Container .selected.selectable")[i]
		day.click()
		# time.sleep(0.25)
		get_table()
		try:
			while True:
				click_next_page_button()
				get_table()
		except Exception as e:
			continue

	return True

def click_next_page_button():

	global driver
	assert driver != None
	actions = ActionChains(driver)

	try:
		# click the next button in JavaScript, selenium was a bit buggy with this
		driver.execute_script("document.querySelector('a.yui-pg-next').click()")
		
		# failed python code for reference
		# next_page_button = driver.find_element_by_class_name("yui-pg-next")
		# next_page_button.location_once_scrolled_into_view
		# actions.move_to_element(next_page_button)
		# actions.pause(.25)
		# actions.click(next_page_button)
		# actions.perform()
		# driver.execute_script("arguments[0].scrollIntoView();", next_page_button)
		# next_page_button.click()
		print("clicked the button")
	except Exception as e:
		raise Exception("Cannot click the next button")

def get_table():

	global driver
	assert driver != None
	global ultimate_df

	dividend_dates_html = driver.find_element_by_id("listed_divdates").get_attribute("innerHTML")
	current_page_df = pd.read_html(dividend_dates_html)[0]

	print(current_page_df.head(2))

	if ultimate_df.shape[0] == 0:
		# initialize the dataframe if it doesn't exist yet
		ultimate_df = current_page_df
	else:
		# append this data to the original dataframe
		ultimate_df = ultimate_df.append(current_page_df)

if __name__ == "__main__":
	get_all_dividends()
	ultimate_df.to_csv("all_thestreet_dividends.csv", index=False)