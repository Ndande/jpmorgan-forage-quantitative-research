# jpmorgan-forage-quantitative-research
## Task 1: Natural Gas Price Estimator

Builds a pricing function that estimates natural gas prices for any given date
based on 4 years of monthly historical data (Oct 2020 - Sep 2024).

## Features
- Price estimation for any historical or future date
- 24-month forward forecast using seasonal decomposition
- Weekend adjustment for reduced industrial demand
- Built with Meta's Prophet library

## How to Run
pip install pandas matplotlib prophet

python gas_pricing.py

## Output
- Interactive forecast chart showing historical data and 24-month projection
- Price estimates for any input date
