# xWinRate

## Introduction
xWinRate is a machine learning project designed to predict the likelihood of a player winning one encounter in Counter-Strike. 

### Why this measure?
In official matches, players' performances are generally evaluated using ambiguous features like K/D Ratio, which doesn't make a differentiation between each individual fight. Due to the simplicity of this measure, K/D Ratio cannot effectively reflect the true skill of a player: A player might die becasue of getting blind, because of low health, because of the weapon disadvantage ...

Thus, I was thinking about applying the Machine Learning Methods to provide a more objective measure of player skills by taking more factors into account. So that we can get more informative statistics, which can help the players and scouts to understand the players with less errors.

## Dataset

### What is a .dem file?
A `.dem` file is a data file format used by the game Counter-Strike to store recorded gameplay demos. These files contain detailed in-game information, capturing every action that happens during a match. This allows the game to "replay" the events exactly as they happened, essentially enabling users to watch a recording of the game from various perspectives.

### Data Source
hltv.org is a news website and forum which covers professional Counter Strike esports news, tournaments and statistics. The `.dem` files for all of the matches are available for downloading.

### Data used in this project
For this project, all demos are about matches took place on the map **de_dust2**. 

In this project, 95 matches (95 `.dem` files) are downloaded and used.

I intentionally picked one map **de_dust2** becasue the `xWinRate` measure will take player positions into account, and the outcomes of positioning at one particular x,y coordinate across maps are different.

### How is .dem file parsed to the usable format?
The data are first downloaded in the `.dem` format from the hltv.org (). Then these `.dem` files are feed into DemoParsers (https://github.com/LaihoE/demoparser), processed into the dataframe format where per tick in-game information are available in rows.

## Methodology & Models

### Training Data Preperation
Using the parser we can query a dataframe containing all player death events, the corresponding attackers, and the tick of this event. 
So the data would be prepared in such way:

- Ground truth label (y): the event itself, the id of the attacker corresponds to the True Label.
- Data to predict on (X): At the tick, that is 50 units prior to the death event, the information about [player health, player position, ... TODO: Fill in this part]

### Models & Performances
I've used Logistic Regression, Random Forest, Multiple Layer Perceptrons.

All of them reached around 70% accuracy when predicting the outcome of the encounter.


## Demonstrations
![xWinRateExample](demo_analysis\slides\xWinRate_01.jpg)