# INVENTORY MANAGEMENT #
Let's understand the inventory management optimization using reinforcement learning by training the agent to play the beer game as below.

Beer Game
=========
## Introduction ##
You are a **Retailer** within a supply chain that delivers beer from a brewery via a distributor, a wholesaler and a retailer to the end consumer.<br>
Your challenge is to ensure that the consumers demand for beer is satisfied by managing the inventory in the retail store to the sufficient amount to meet the end customer requirements.<br>

### Objective ###
* **Keep your inventory steady** You need to keep your inventory steady to ensure you can deal with fluctuations in demand. Try to reach an inventory target of **250** by the end of the game.
* **Keep your cost down** Both excess inventory and backorders increase your cost. Keep your total cost below **$8300**

### Rules ###
The rules of the game are simple - the game is played in 24 rounds, in every round of the game you perform the following four steps:
* **Check deliveries** Check how many units of beer are being delivered to you from your wholesaler.
* **Check orders** Check how many units of beer your customers have ordered.
* **Deliver beer** Deliver as much beer as you can to satisfy demand in the beer game.
* **Maker order decision** Decide how many units of beer you need to order from your wholesaler to keep your inventory stocked up and to ensure you have enough beer to meet future demands.

### Constraints ###
There are various constraints in the game that you need to be aware of during the game.
* **Delays** Your demands for beer may not be fulfilled immediately - your supplier may also be out of stock and will then have to pass is own order up the supply chain.
* **Inventory Costs** If you order too many units of beer, your inventory costs will rise, because you will need more people to handle the beer and more storage space. Inventory cost you **$0.5**
* **Backorder Costs** If you order too few units of beer, you may not be able to supply your customer with enough beer. Backorders cost you **$1** per unit



## Environment Setup for Anaconda ##

### LINUX ###
download anaconda using command wget https://repo.continuum.io/archive/anaconda_file.sh
``` 
wget https://repo.continuum.io/archive/Anaconda3-4.3.0-Linux-x86_64.sh
```

after downloading anaconda we need to install it. for installing anaconda run comman anaconda_file.sh
for detailed instructions to install anaconda check the following link - https://docs.anaconda.com/anaconda/install/linux/
``` 
bash Anaconda3-4.3.0-Linux-x86_64.sh
```
### WINDOWS ###
download anaconda from website https://docs.anaconda.com/anaconda/install/windows/ and follow the instructions to install it.

___
now once anaconda is installed we need to setup a conda environment. for setting this up run command-
conda create --name myenv
``` 
conda create --name beer-game python==3.6.8
```

after creating environment, activate it using one of the following commands-
```
source activate beer-game
conda activate beer-game
```

run requirment.txt file to install necessary python modules
```
pip install -r requirements.txt
```

## Setup Selenium ##
download the chrome driver for selenium as per your chrome version from the link- https://sites.google.com/a/chromium.org/chromedriver/downloads


## Training the RL Agent ##
update configuration file **config.conf** for necessary paramenters of model and data otherwise leave as it is
* **csv_file_path** csv file path for the incoming orders data
* **io_column** column id in the csv file for the incoming orders column
* **number_of_weeks** this is the time duration for which the cost has to be optimized
* **time_delay** transportation time lag in weeks/days
after updating configuration file run the python script **main.py**
```
python main.py
```

## Playing Game ##
update configuration file **config.conf** for selenium chrome driver path in **chrome_driver**
run the play_beer_game.py file to start playing beer game.
```
python play_beer_game.py
```
