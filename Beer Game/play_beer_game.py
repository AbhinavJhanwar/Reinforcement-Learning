
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import re
import time
import _pickle as pickle
import configparser
import numpy as np


def max_dict(d):
    # returns the argmax (key) and max (value) from a dictionary
    max_value = np.max(d)
    max_key = np.argmax(d)
    return max_key, max_value


def read_Q(file_name='Q_value.pickle'):
    # read Q table
    with open(file_name, 'rb') as file:
        Q = pickle.load(file)
    return Q


def predict_outgoing_order(inventory, backlog):
    s = min(n_bins, max(-n_bins, int((inventory - backlog) *
                                     n_bins/max_inventory)))
    return all_possible_actions[max_dict(Q[s])[0]]


if __name__ == '__main__':
    # read configuration file
    config = configparser.ConfigParser()
    config.read('config.conf')

    # number of levels to divide the total states into
    # -n_bins (for backlog) to +nbins (for inventory)
    n_bins = eval(config['supply_chain_params']['n_bins'])

    # maximum inventory that I can hold in the store
    max_inventory = eval(config['supply_chain_params']['max_inventory'])

    all_possible_actions = eval(config['supply_chain_params']
                                ['all_possible_actions'])

    chrome_driver = eval(config['selenium']['chrome_driver'])

    # read Q value from pickle file
    Q = read_Q()

    # start playing beer game
    EXE_PATH = chrome_driver
    driver = webdriver.Chrome(executable_path=EXE_PATH)
    driver.get('http://beergame.transentis.com/')
    time.sleep(5)

    # click on cockpit button
    element = driver.find_element_by_class_name('button')
    element.click()
    time.sleep(3)

    for i in range(23):
        all_spans = \
        driver.find_elements_by_css_selector("div[class^='instrument']")
        # get incoming order from customer
        incoming_order = int(re.findall(r'[0-9]*', all_spans[1].text)[0])

        # get incoming delivery from plant
        incoming_delivery = int(re.findall(r'[0-9]*', all_spans[4].text)[0])

        # get current inventory
        inventory = int(re.findall(r'[0-9]*', all_spans[5].text)[0])

        # get backlog
        backlog = int(re.findall(r'[0-9]*', all_spans[2].text)[0])

        # get outgoing delivery
        outgoing_delivery = int(re.findall(r'[0-9]*', all_spans[6].text)[0])

        print('incoming delivery:', incoming_delivery)
        print('outgoing delivery:', incoming_delivery)
        print('incoming order:', incoming_order)
        print('backlog:', backlog)
        print('inventory:', inventory)

        if incoming_order >= inventory + incoming_delivery:
            backlog = backlog + incoming_order - outgoing_delivery
            inventory = 0
        elif incoming_order <= inventory + incoming_delivery - backlog:
            backlog = 0
            inventory = inventory + incoming_delivery - outgoing_delivery
        elif incoming_order > inventory + incoming_delivery - backlog:
            backlog = backlog + incoming_order - outgoing_delivery
            inventory = 0

        outgoing_order = predict_outgoing_order(inventory, backlog)
        print('outgoing order:', outgoing_order)

        # fill outgoing_order_value
        element = driver.find_element_by_xpath("//div[@class='input']")
        e2 = element.find_element_by_xpath(".//div")
        e2.send_keys(Keys.BACK_SPACE*4)
        e2.send_keys(outgoing_order)

        # click on order
        element = driver.find_elements_by_xpath("//button")
        element[0].click()

        time.sleep(1)
