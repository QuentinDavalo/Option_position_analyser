import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import math


def calculate_black_scholes_price(S, K, T, r, sigma, option_type):
    """
        Calculate theoritical price with BS model.

        Parameters:
        - S: VO
        - K: Strike price
        - T: Time to expiration (in years)
        - r: Rf rate
        - sigma: Vol
        - option_type: 'Call' or 'Put'
        """
    if T <= 0:
        if option_type == 'Call':
            return max(0, S - K)
        else:  # Put
            return max(0, K - S)

    def norm_cdf(x):
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    def norm_pdf(x):
        return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)

    # Compute d1 and d2
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    # Theoretical price
    if option_type == 'Call':
        price = S * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)
    else:
        price = K * math.exp(-r * T) * norm_cdf(-d2) - S * norm_cdf(-d1)

    return price


def calculate_black_scholes_greeks(S, K, T, r, sigma, option_type):
    """
        Calculate greeks with BS model.

        Parameters:
        - S: VO
        - K: Strike price
        - T: Time to expiration (in years)
        - r: Rf rate
        - sigma: Vol
        - option_type: 'Call' or 'Put'
        """
    if T <= 0:
        return {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0, 'rho': 0, 'price': 0}

    def norm_cdf(x):
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    def norm_pdf(x):
        return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)

    # Compute d1 and d2
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    # Option price and greeks according to Black-Scholes
    if option_type == 'Call':
        price = S * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)
        delta = norm_cdf(d1)
        rho = K * T * math.exp(-r * T) * norm_cdf(d2) * 0.01
        theta = (-S * norm_pdf(d1) * sigma / (2 * math.sqrt(T)) -
                 r * K * math.exp(-r * T) * norm_cdf(d2))
    else:
        price = K * math.exp(-r * T) * norm_cdf(-d2) - S * norm_cdf(-d1)
        delta = norm_cdf(d1) - 1
        rho = -K * T * math.exp(-r * T) * norm_cdf(-d2) * 0.01
        theta = (-S * norm_pdf(d1) * sigma / (2 * math.sqrt(T)) +
                 r * K * math.exp(-r * T) * norm_cdf(-d2))

    gamma = norm_pdf(d1) / (S * sigma * math.sqrt(T))

    vega = S * math.sqrt(T) * norm_pdf(d1) * 0.01

    # Convert theta to value per trading day
    theta = theta / 252

    return {
        'delta': delta,
        'gamma': gamma,
        'theta': theta,
        'vega': vega,
        'rho': rho,
        'price': price}


def calculate_payoff(portfolio, price, shares_per_contract):
    """
    Calculate the portfolio payoff at expiration for the actual price
    """
    payoff = 0

    for option in portfolio:
        strike = option['strike']
        option_type = option['type']
        position = option['position']
        size = option['size']

        if option_type == 'Call':
            option_payoff = max(0, price - strike) if position == 'Long' else -max(0, price - strike)
        else:  # Put
            option_payoff = max(0, strike - price) if position == 'Long' else -max(0, strike - price)

        payoff += option_payoff * size * shares_per_contract

    return payoff


def calculate_pnl(portfolio, price, shares_per_contract):
    """
    Calculate the portfolio PnL at expiration for the actual price.
    """
    payoff = calculate_payoff(portfolio, price, shares_per_contract)
    initial_cost = sum(
        option['price'] * option['size'] * shares_per_contract * (1 if option['position'] == 'Long' else -1) for option
        in portfolio)

    return payoff - initial_cost


def calculate_portfolio_value(portfolio, price, date_ratio, r, sigma, shares_per_contract):
    """Calculate the theoretical value of the portfolio at a given price and time.

    Parameters:
    - portfolio: List of options in the portfolio
    - price: Underlying asset price
    - date_ratio: Time remaining (0=today, 1=expiration)
    - r: Rf rate
    - sigma: Vol
    - shares_per_contract"""

    value = 0

    for option in portfolio:
        # Adjust the remaining time
        T_adjusted = option['expiration'] * date_ratio

        # Calculate the theoretical option value with Black-Scholes
        option_price = calculate_black_scholes_price(
            price, option['strike'], T_adjusted, r, sigma, option['type']
        )

        # Adjust according to position and size
        multiplier = 1 if option['position'] == 'Long' else -1
        value += option_price * option['size'] * shares_per_contract * multiplier

    return value


def calculate_portfolio_greeks(portfolio, shares_per_contract):
    """
    Calculate the total greeks of the portfolio.
    """
    portfolio_greeks = {
        'delta': 0,
        'gamma': 0,
        'theta': 0,
        'vega': 0,
        'rho': 0
    }

    for option in portfolio:
        for greek in portfolio_greeks:
            portfolio_greeks[greek] += option[greek] * option['size'] * shares_per_contract

    return portfolio_greeks


def display_options(options_df, option_type, current_price):
    """
    Display available options in a table format.
    """
    # Sort options by strike
    options_df = options_df.sort_values('strike')

    # Find the index of the option closest to the current price
    closest_idx = (options_df['strike'] - current_price).abs().idxmin()
    closest_position = options_df.index.get_loc(closest_idx)

    # Determine start and end indices to get ~20 options (10 on each side)
    start_idx = max(0, closest_position - 10)
    end_idx = min(len(options_df), closest_position + 11)  # +11 because option ATM is excluded

    # Select the filtered options
    filtered_options = options_df.iloc[start_idx:end_idx]

    # Display the table
    print(f"\n{option_type}s available:")
    print(f"{'#':<3} | {'Strike':<8} | {'Price':<8} | {'Volume':<8}")
    print("-" * 35)

    for i, (_, option) in enumerate(filtered_options.iterrows()):
        print(f"{i + 1:<3} | ${option['strike']:<7.2f} | ${option['lastPrice']:<7.2f} | {option['volume']:<8}")

    return filtered_options


def select_options(calls, puts, ticker_symbol, current_price, T, r, sigma):
    """
    Allow the user to select options for their portfolio.
    """
    portfolio = []

    # Filter and display available options
    filtered_calls = display_options(calls, "Call", current_price)
    filtered_puts = display_options(puts, "Put", current_price)

    # Option selection
    while True:
        option_type = input("\nEnter option type (Call/Put) or 'end' to finish: ").capitalize()

        if option_type.lower() == 'end':
            break

        if option_type not in ['Call', 'Put']:
            print("Invalid option type. Please enter 'Call' or 'Put'.")
            continue

        filtered_options = filtered_calls if option_type == 'Call' else filtered_puts
        all_options = calls if option_type == 'Call' else puts

        option_input = input(f"Enter the {option_type} option number (or 'x' to search by strike): ")

        if option_input.lower() == 'x':
            # Search by strike price
            try:
                target_strike = float(input("Enter the desired strike price: "))
                # Find the option with the closest strike
                closest_idx = (all_options['strike'] - target_strike).abs().idxmin()
                selected_option = all_options.loc[closest_idx].copy()
                print(
                    f"Closest option found: {option_type} Strike ${selected_option['strike']:.2f}, Price: ${selected_option['lastPrice']:.2f}")
                confirm = input("Do you want to use this option? (y/n): ").lower()
                if confirm != 'y':
                    continue
            except ValueError:
                print("Invalid strike price.")
                continue
        else:
            try:
                option_idx = int(option_input) - 1
                if option_idx < 0 or option_idx >= len(filtered_options):
                    print("Invalid option number.")
                    continue
                selected_option = filtered_options.iloc[option_idx].copy()
            except ValueError:
                print("Invalid input.")
                continue

        position_type = input("Enter position type (Long/Short): ").capitalize()

        if position_type not in ['Long', 'Short']:
            print("Invalid position type. Please enter 'Long' or 'Short'.")
            continue

        position_size = int(input("Enter the number of options: "))

        # Calculate greeks using the Black-Scholes model
        strike = selected_option['strike']

        # Get implied volatility if available in the data
        if 'impliedVolatility' in selected_option and not pd.isna(selected_option['impliedVolatility']):
            implied_vol = selected_option['impliedVolatility']
            print(f"Using implied volatility: {implied_vol:.2%}")
        else:
            implied_vol = sigma
            print(f"Using historical volatility: {implied_vol:.2%}")

        greeks = calculate_black_scholes_greeks(current_price, strike, T, r, implied_vol, option_type)

        # Adjust greeks based on position type (Long/Short)
        sign = 1 if position_type == 'Long' else -1
        for key in greeks:
            if key != 'price':  # Don't invert the price
                greeks[key] *= sign

        # Use the last price instead of bid/ask
        option_price = selected_option['lastPrice']


        # Add to our portfolio
        portfolio.append({
            'type': option_type,
            'ticker': ticker_symbol,
            'strike': strike,
            'position': position_type,
            'size': position_size,
            'price': option_price,
            'expiration': T,
            'delta': greeks['delta'],
            'gamma': greeks['gamma'],
            'theta': greeks['theta'],
            'vega': greeks['vega'],
            'rho': greeks['rho']})

        print(f"Option added to portfolio: {option_type} {ticker_symbol} {strike} {position_type} x{position_size}")
        print(f"Delta: {greeks['delta']:.4f}, Gamma: {greeks['gamma']:.4f}, Theta: {greeks['theta']:.4f}")
        print(f"Vega: {greeks['vega']:.4f}, Rho: {greeks['rho']:.4f}")

    return portfolio


def plot_payoff_and_pnl(portfolio, current_price, shares_per_contract):
    """
    Plot the portfolio payoff and PnL charts.
    """
    # Define a range of prices for the underlying asset
    price_range = np.linspace(current_price * 0.7, current_price * 1.3, 100)

    # Get global parameters
    # Assume all options have the same expiration date
    if portfolio:
        first_option = portfolio[0]
        T = first_option['expiration']
        r = 0.0425  # Risk-free rate (adjust if necessary)
        sigma = 0.2  # Volatility (adjust if necessary)
    else:
        T = 0
        r = 0.0425
        sigma = 0.2

    # Calculate payoff at expiration
    payoffs = []
    for price in price_range:
        payoff = calculate_payoff(portfolio, price, shares_per_contract)
        payoffs.append(payoff)

    # Calculate PnL at expiration (payoff minus initial cost)
    pnls = []
    for price in price_range:
        pnl = calculate_pnl(portfolio, price, shares_per_contract)
        pnls.append(pnl)

    # Calculate portfolio value at different times before expiration
    values_mid = []
    values_now = []

    for price in price_range:
        # Calculate value halfway to expiration
        value_mid = calculate_portfolio_value(portfolio, price, 0.5, r, sigma, shares_per_contract)
        values_mid.append(value_mid)

        # Calculate current theoretical value
        value_now = calculate_portfolio_value(portfolio, price, 1.0, r, sigma, shares_per_contract)
        values_now.append(value_now)

    print("\nGenerating payoff and PnL charts...")

    # Create charts
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

    # Payoff chart
    ax1.plot(price_range, payoffs, 'b-', linewidth=2, label="At Expiration")
    ax1.plot(price_range, values_mid, 'g--', linewidth=1.5, label="Mid-way to Expiration")
    ax1.plot(price_range, values_now, 'r:', linewidth=1.5, label="Current Theoretical Value")
    ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax1.axvline(x=current_price, color='green', linestyle='--', alpha=0.3, label="Current Price")
    ax1.set_title('Portfolio Value at Different Times', fontsize=14)
    ax1.set_xlabel('Underlying Price ($)', fontsize=12)
    ax1.set_ylabel('Value ($)', fontsize=12)
    ax1.grid(True)
    ax1.legend()

    # Add annotations
    max_payoff = max(payoffs)
    min_payoff = min(payoffs)
    ax1.text(price_range[0], max_payoff * 0.9, f'Max Payoff: ${max_payoff:.2f}',
             fontsize=10, bbox=dict(facecolor='white', alpha=0.5))

    # PnL chart
    ax2.plot(price_range, pnls, 'g-', linewidth=2)
    ax2.axhline(y=0, color='r', linestyle='-', alpha=0.3)
    ax2.axvline(x=current_price, color='g', linestyle='--', alpha=0.3)
    ax2.set_title('Portfolio PnL at Expiration', fontsize=14)
    ax2.set_xlabel('Underlying Price ($)', fontsize=12)
    ax2.set_ylabel('PnL ($)', fontsize=12)
    ax2.grid(True)

    # Add annotations for PnL
    max_pnl = max(pnls)
    min_pnl = min(pnls)
    breakeven_points = []
    for i in range(1, len(pnls)):
        if (pnls[i - 1] < 0 and pnls[i] >= 0) or (pnls[i - 1] >= 0 and pnls[i] < 0):
            # Linear estimation of breakeven point
            x1, x2 = price_range[i - 1], price_range[i]
            y1, y2 = pnls[i - 1], pnls[i]
            breakeven = x1 + (x2 - x1) * (0 - y1) / (y2 - y1)
            breakeven_points.append(breakeven)

    ax2.text(price_range[0], max_pnl * 0.9, f'Max PnL: ${max_pnl:.2f}',
             fontsize=10, bbox=dict(facecolor='white', alpha=0.5))

    if breakeven_points:
        for i, point in enumerate(breakeven_points):
            ax2.axvline(x=point, color='orange', linestyle='--', alpha=0.5)
            ax2.text(point, min_pnl * 0.2, f'BE: ${point:.2f}',
                     fontsize=8, rotation=90, va='bottom')

    plt.tight_layout()

    # Ask the user if they want to save the charts
    save_option = input("\nDo you want to save the charts? (y/n): ").lower()
    if save_option == 'y':
        filename = input("Filename (without extension): ") or "options_analysis"
        plt.savefig(f"{filename}.png", dpi=300, bbox_inches='tight')
        print(f"Charts saved to {filename}.png")

    # Display the charts
    print("Displaying charts... (close the window to continue)")
    plt.show()


def main():
    """
    Main program function.
    """
    # Ask for ticker
    ticker_symbol = input("Enter the stock ticker (US Stock and US index ETF only): ")

    # Ask for number of shares per contract
    shares_per_contract = int(input("Enter the number of shares per contract (default: 100): ") or "100")

    # Get real-time stock price
    ticker = yf.Ticker(ticker_symbol)
    current_price = ticker.history(period="1d")['Close'].iloc[-1]
    print(f"Current price of {ticker_symbol}: ${current_price:.2f}")

    # Ask for expiration date
    print("\nAvailable expiration dates:")
    expiration_dates = ticker.options
    for i, date in enumerate(expiration_dates):
        print(f"{i + 1}. {date}")

    date_idx = int(input("\nEnter the number corresponding to your desired expiration date: ")) - 1
    expiration_date = expiration_dates[date_idx]
    print(f"Selected expiration date: {expiration_date}")

    # Get the options chain
    options_chain = ticker.option_chain(expiration_date)
    calls = options_chain.calls
    puts = options_chain.puts

    # Check for the colums
    required_columns = ['strike', 'lastPrice', 'volume']
    missing_columns = [col for col in required_columns if col not in calls.columns]
    if missing_columns:
        print(f"Warning: The following columns are missing from the data: {missing_columns}")
        print("Using alternative columns if available.")

        # If no last price, average between the ask-bid
        if 'lastPrice' not in calls.columns and 'bid' in calls.columns and 'ask' in calls.columns:
            calls['lastPrice'] = (calls['bid'] + calls['ask']) / 2
            puts['lastPrice'] = (puts['bid'] + puts['ask']) / 2
            print("Price calculated as bid-ask average.")

        # If volume is not available, set to 0
        if 'volume' not in calls.columns:
            calls['volume'] = 0
            puts['volume'] = 0

    # Calculate days until expiration
    expiry_date = datetime.strptime(expiration_date, '%Y-%m-%d')
    days_to_expiry = (expiry_date - datetime.now()).days
    years_to_expiry = days_to_expiry / 365.0

    # Parameters for greek calculations
    risk_free_rate = 0.0425  # Averaged 10Y US Treasury rf rate

    # Calculate historical volatility
    print("Calculating historical volatility...")
    hist_data = ticker.history(period="1y")
    if len(hist_data) > 0:
        # Calculate daily logarithmic returns
        log_returns = np.log(hist_data['Close'] / hist_data['Close'].shift(1)).dropna()
        # Calculate standard deviation of returns
        daily_vol = log_returns.std()
        # Annualize volatility (√252 for trading days per year)
        volatility = daily_vol * np.sqrt(252)
        print(f"Annualized historical volatility: {volatility:.2%}")
    else:
        # Use default value if data is not available
        volatility = 0.2
        print("Unable to calculate historical volatility. Default value used: 20%")

    # 5-7. Option selection and parameters
    portfolio = select_options(calls, puts, ticker_symbol, current_price, years_to_expiry, risk_free_rate, volatility)

    # 8-10. Calculations and results display
    if not portfolio:
        print("Empty portfolio. No calculations to perform.")
        return

    # Display portfolio summary
    print("\nPortfolio Summary:")
    print(f"{'Type':<5} {'Ticker':<6} {'Strike':<8} {'Position':<8} {'Size':<6} {'Delta':<10} {'Gamma':<10} {'Theta':<10} {'Vega':<10} {'Rho':<10}")
    print("-" * 90)

    for option in portfolio:
        print(f"{option['type']:<5} {option['ticker']:<6} {option['strike']:<8.2f} {option['position']:<8} {option['size']:<6} {option['delta']:<10.4f} {option['gamma']:<10.4f} {option['theta']:<10.4f} {option['vega']:<10.4f} {option['rho']:<10.4f}")

    # Calculate portfolio greeks
    portfolio_greeks = calculate_portfolio_greeks(portfolio, shares_per_contract)
    print("\nPortfolio Greeks:")
    print(f"Delta: {portfolio_greeks['delta']:.4f}")
    print(f"Gamma: {portfolio_greeks['gamma']:.4f}")
    print(f"Theta: {portfolio_greeks['theta']:.4f} (per trading day)")
    print(f"Vega: {portfolio_greeks['vega']:.4f} (for 1% change in volatility)")
    print(f"Rho: {portfolio_greeks['rho']:.4f} (for 1% change in interest rate)")

    # Calculate and display payoff and PnL
    plot_payoff_and_pnl(portfolio, current_price, shares_per_contract)


if __name__ == "__main__":
    main()