import pandas as pd
def calculate_moving_average(data, window):
    return data.rolling(window=window).mean()