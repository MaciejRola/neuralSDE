import pandas as pd
import numpy as np


def generate_data(S0, T, file):
    data = pd.DataFrame()
    type = "Call"
    for maturity in np.linspace(0.25*T, T, 4):
        for strike in list(np.linspace(0.8 * S0, S0, num=100, endpoint=False)) + list(np.linspace(S0, 1.2 * S0, num=101, endpoint=True)):
            data = pd.concat((data, pd.DataFrame({"Expiration_date": [maturity], "Strike": [strike], "Call_or_Put": [type]})))
    data.to_csv(file, index=False)


generate_data(100, 2, "Data/Options.csv")
