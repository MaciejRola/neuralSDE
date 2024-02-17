import numpy as np
import pandas as pd


def generate_data(S0, T, frequency, file):
    """Generate a dataset of quarterly maturing options for given initial price and time horizon."""
    data = pd.DataFrame()
    frequency_dict = {
        'monthly': 1,
        'bimonthly': 2,
        'quarterly': 3,
        'semi-annually': 6,
        'annually': 12
    }
    freq = frequency_dict[frequency]
    maturities = np.arange(freq, T * 12 + freq, freq) / 12
    # strikes in 10% interval
    strikes = S0 * np.linspace(0.9, 1.1, num=21)
    for maturity in maturities:
        for strike in strikes:
            # we use only call options
            data = pd.concat((data, pd.DataFrame({"Expiration_date": [maturity], "Strike": [strike], "Call_or_Put": ["Call"]})))
    data.to_csv(file, index=False)
    return data


if __name__ == "__main__":
    generate_data(100, 2, 'quarterly', "Data/Options.csv")
