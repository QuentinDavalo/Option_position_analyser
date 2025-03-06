# Options Analysis Tool

A Python application to analyze strategies and visualize potential outcomes.

## What is this?

This tool helps you understand your options positions better. Simply enter the ticker of the stock you want to create a strategy on and let the program guide you.

## How does it work?

The application will ask you the different positions you want to take on each of the option tickers, including the direction of the trade and the size of it. You can add multiple trades, and the program will then aggregate them, calculating and plotting the payoff and PnL based on the datas that you will have entered. It will also calculate the aggregate greeks of the portfolio by using this formula:

Δ_portfolio = Σ(greek × Si × m).

We can simply add them because the option share the same Time to expiration and ticker.

## DISCLAIMER
1. We use the Black-Scholes model, which has known limitations, especially for American-style options.
2. The tool doesn't account for dividends, which can impact options on dividend-paying stocks.
3. The tool relies on Yahoo Finance for its real-time data. It can give inaccurate/unreliable prices depending on various factors, such as the availability of certain data.
4. We use the US 10 year risk-free rate as a benchmark for the calculation. Could be improved by implementing a system that align the risk-free rate based on the time to expiration of the option.
