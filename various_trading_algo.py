import pywt
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import yfinance as yf
import talib as TA
from datetime import datetime
from datetime import date
from dateutil.relativedelta import relativedelta
#
from findiff import FinDiff
from scipy.signal import find_peaks


def projected_value(prev_all_bases, x_axis, y_axis, x_range, y_range, degree):
    if len(prev_all_bases) > 1:
        last_two_bases = prev_all_bases.iloc[-2:].index
        prev_base_index = last_two_bases[-1]
        x_base = x_axis.loc[last_two_bases]
        y_base = y_axis.loc[last_two_bases]
        weights = np.polyfit(x_base, y_base, degree)
        model = np.poly1d(weights)
        projection = model(x_axis)
        angle = np.arctan2((y_base[-1] - y_base[0]) / y_range,
                           (x_base[-1] - x_base[0])) / x_range * 180 / np.pi
        prev_high = y_axis.loc[prev_base_index]
        # print('{} {} {}'.format(projection, angle, prev_high))
        return model, projection[-1], angle, prev_high
    else:
        return None, None, None, None


def peak_trough_projection_no_leaking(data, TA):
    # prevent inforamtion from leaking to future
    temp = data[TA].copy()
    base_date = temp.index[0]
    x_axis_main = pd.Series(temp.index.map(lambda date: (
        pd.Timestamp(date) - base_date).days) + 1, index=temp.index)
    y_axis_main = temp
    #
    resistance = pd.Series(index=temp.index)
    resist_angle = pd.Series(index=temp.index)
    support = pd.Series(index=temp.index)
    support_angle = pd.Series(index=temp.index)
    prev_high = pd.Series(index=temp.index)
    prev_low = pd.Series(index=temp.index)
    #
    for current_date in temp.index:
        Close_uptonow = temp.loc[:current_date]
        Negative_uptonow = temp.loc[:current_date] * -1.0
        x_axis = x_axis_main.loc[:current_date]
        y_axis = y_axis_main.loc[:current_date]
        x_range = x_axis[-1] - x_axis[0] + 1
        y_range = y_axis.max() - y_axis.min()
        #
        # Find support and resistance using scipy
        peaks2, _ = find_peaks(Close_uptonow, prominence=1)
        peak = pd.Series(False, index=Close_uptonow.index)
        peak.iloc[peaks2] = True
        peak_only = peak[peak]
        #
        troughs2, _ = find_peaks(Negative_uptonow, prominence=1)
        trough = pd.Series(False, index=Negative_uptonow.index)
        trough.iloc[troughs2] = True
        trough_only = trough[trough]
        #
        degree = 1
        model_peak, resistance_c, resist_angle_c, prev_high_c = projected_value(
            peak_only, x_axis, y_axis, x_range, y_range, degree)
        model_trough, support_c, support_angle_c, prev_low_c = projected_value(
            trough_only, x_axis, y_axis, x_range, y_range, degree)
        #
        resistance.loc[current_date] = resistance_c
        resist_angle.loc[current_date] = resist_angle_c
        support.loc[current_date] = support_c
        support_angle.loc[current_date] = support_angle_c
        prev_high.loc[current_date] = prev_high_c
        prev_low.loc[current_date] = prev_low_c
    #
    data['resist_noleak'] = resistance
    data['resist_angle_noleak'] = resist_angle
    data['peak_noleak'] = peak
    data['support_noleak'] = support
    data['support_angle_noleak'] = support_angle
    data['trough_noleak'] = trough
    data['prev_high_noleak'] = prev_high
    data['prev_low_noleak'] = prev_low
    #
    return data, x_axis_main, model_peak, model_trough, peak_only, trough_only


def peak_trough_calculus_projection_no_leaking(data, TA):
    # prevent inforamtion from leaking to future
    temp = data[TA].copy()
    base_date = temp.index[0]
    x_axis_main = pd.Series(temp.index.map(lambda date: (
        pd.Timestamp(date) - base_date).days) + 1, index=temp.index)
    y_axis_main = temp
    #
    resistance = pd.Series(index=temp.index)
    resist_angle = pd.Series(index=temp.index)
    support = pd.Series(index=temp.index)
    support_angle = pd.Series(index=temp.index)
    prev_high = pd.Series(index=temp.index)
    prev_low = pd.Series(index=temp.index)
    #
    for current_date in temp.index[3:]:
        df_uptonow = pd.DataFrame(temp.loc[:current_date])
        # Negative_uptonow = temp.loc[:current_date] * -1.0
        x_axis = x_axis_main.loc[:current_date]
        y_axis = y_axis_main.loc[:current_date]
        x_range = x_axis[-1] - x_axis[0] + 1
        y_range = y_axis.max() - y_axis.min()
        #
        # Find support and resistance using scipy
        dx = 1  # 1 day interval
        d_dx = FinDiff(0, dx, 1)
        d2_dx2 = FinDiff(0, dx, 2)
        clarr = df_uptonow['Close'].values
        df_uptonow['mom'] = d_dx(clarr)
        df_uptonow['momacc'] = d2_dx2(clarr)
        #
        # shift(1) is previous day
        # df_uptonow['mom'] from +ve to -ve and df_uptonow['momacc'] is negatice
        peak1 = (df_uptonow['momacc'] < 0) & (df_uptonow['mom'] < 0) & (df_uptonow['mom'].shift(1) > 0) & \
                (df_uptonow['Close'] >= df_uptonow['Close'].shift(1))
        peak2 = (df_uptonow['momacc'].shift(1) < 0) & (df_uptonow['mom'] < 0) & (df_uptonow['mom'].shift(1) > 0) & \
                (df_uptonow['Close'].shift(1) >= df_uptonow['Close'])
        peak2 = peak2.shift(-1)
        peak3 = ((df_uptonow['momacc'] < 0) | ((df_uptonow['momacc'].shift(-1) < 0) & (df_uptonow['momacc'] > 0))) & \
                (df_uptonow['mom'] == 0) & (df_uptonow['mom'].shift(1) > 0) & \
                (df_uptonow['mom'].shift(-1) < 0)
        peak_sum = peak1 | peak2 | peak3
        # print(peak_sum.index)
        peak_clean = peak_sum[~(peak_sum & peak_sum.shift(-1))]
        # print(peak_clean.index)
        #
        peak = pd.Series(False, index=df_uptonow.index)
        # print(peak.index)
        peak = (peak | peak_clean)
        peak_only = peak[peak]
        #
        # shift(1) is previous day
        # df_uptonow['mom'] from -ve to +ve and df_uptonow['momacc'] is positive
        trough1 = (df_uptonow['momacc'] > 0) & (df_uptonow['mom'] > 0) & (df_uptonow['mom'].shift(1) < 0) & \
                  (df_uptonow['Close'] <= df_uptonow['Close'].shift(1))
        trough2 = (df_uptonow['momacc'].shift(1) > 0) & (df_uptonow['mom'] > 0) & (df_uptonow['mom'].shift(1) < 0) & \
                  (df_uptonow['Close'] >= df_uptonow['Close'].shift(1))
        trough2 = trough2.shift(-1)
        trough3 = ((df_uptonow['momacc'] > 0) | ((df_uptonow['momacc'].shift(-1) > 0) & (df_uptonow['momacc'] < 0))) & \
                  (df_uptonow['mom'] == 0) & (df_uptonow['mom'].shift(1) < 0) & \
                  (df_uptonow['mom'].shift(-1) > 0)
        trough3 = trough3.shift(1)
        trough_sum = trough1 | trough2 | trough3
        trough_clean = trough_sum[~(trough_sum & trough_sum.shift(-1))]
        #
        trough = pd.Series(False, index=df_uptonow.index)
        trough = (trough | trough_clean)
        trough_only = trough[trough]
        #
        degree = 1
        model_peak, resistance_c, resist_angle_c, prev_high_c = projected_value(
            peak_only, x_axis, y_axis, x_range, y_range, degree)
        model_trough, support_c, support_angle_c, prev_low_c = projected_value(
            trough_only, x_axis, y_axis, x_range, y_range, degree)
        #
        resistance.loc[current_date] = resistance_c
        resist_angle.loc[current_date] = resist_angle_c
        support.loc[current_date] = support_c
        support_angle.loc[current_date] = support_angle_c
        prev_high.loc[current_date] = prev_high_c
        prev_low.loc[current_date] = prev_low_c
    #
    data['resist_calculus'] = resistance
    data['resist_angle_calculus'] = resist_angle
    data['peak_calculus'] = peak
    data['support_calculus'] = support
    data['support_angle_calculus'] = support_angle
    data['trough_calculus'] = trough
    data['prev_high_calculus'] = prev_high
    data['prev_low_calculus'] = prev_low
    #
    return data, x_axis_main, model_peak, model_trough, peak_only, trough_only


def strategy(df, enter, exit, col_list, long, threshold, stoploss_needed=True):
    buy_date = []
    sell_date = []
    sell_cos_stoploss = []
    price_buy = []
    price_sell = []
    parm_list = []
    #
    last_sell = datetime.strptime('01-01-1900', '%m-%d-%Y')
    #
    for i in enter.index[enter == 1]:
        is_stoploss = False
        next_tradeday_exit = 0
        try:
            # df=data1 -> next day of the signal day
            date1 = df.index[df.index > i][0]
            stoploss_price = df.loc[date1, 'Open'] * threshold
            if date1 is not None:
                test = exit[exit == 1]  # test = ALL exit signal day
                try:
                    # The first exit signal equal or after to buy date
                    first_exit = test[test.index >= date1].index[0]
                except:
                    first_exit = None
                    next_tradeday_exit = None
                #
                if stoploss_needed:
                    # since buydate, all dates with Close < stoploss
                    stoploss_dates = df[df['Close'] <
                                        stoploss_price].loc[date1:].index
                    if len(stoploss_dates) > 0:
                        first_stoploss = stoploss_dates[0]
                    else:
                        first_stoploss = None
                    #
                    # Compare first_stoploss vs first_exit
                    if first_stoploss is None:
                        if first_exit is None:
                            next_tradeday_exit = None
                        else:
                            pass  # No change to first_exit
                    else:
                        if (first_exit is None) or (first_stoploss < first_exit):
                            first_exit = first_stoploss
                            is_stoploss = True
                if next_tradeday_exit is not None:
                    # the next trading date after the first exit signal
                    next_tradeday_exit = df.index[df.index > first_exit][0]
            else:
                next_tradeday_exit = None
            if i > last_sell:
                buy_date.append(date1)
                sell_date.append(next_tradeday_exit)
                sell_cos_stoploss.append(is_stoploss)
                if next_tradeday_exit is None:
                    # since the no more exit, set last_sell to today to prevent any more enter
                    last_sell = datetime.today()
                else:
                    last_sell = first_exit
        except:
            break

    for i in buy_date:
        if i is not None:
            price = df.loc[i]["Open"]
            price_buy.append(price)
            parm_list.append([df.loc[i, col] for col in col_list])
        else:
            price_buy.append(None)

    for i in sell_date:
        if i is not None:
            price = df.loc[i]["Open"]
            price_sell.append(price)
        else:
            price_sell.append(None)

    tran_dict = {"Date_Buy": buy_date,
                 "Date_Sell": sell_date,
                 "Price_Buy": price_buy,
                 "Price_Sell": price_sell,
                 "Sell_cos_stoploss": sell_cos_stoploss}
    for index, col in enumerate(col_list):
        tran_dict[col] = [l[index] for l in parm_list]

    d1 = pd.DataFrame(tran_dict)
    if long == 1:
        d1["Gain"] = d1["Price_Sell"] - d1["Price_Buy"]
        d1["Return"] = 100 * (d1["Price_Sell"] -
                              d1["Price_Buy"]) / d1["Price_Buy"]
        d1["Position"] = "long"
    else:
        d1["Gain"] = d1["Price_Buy"] - d1["Price_Sell"]
        d1["Return"] = 100 * (d1["Price_Buy"] -
                              d1["Price_Sell"]) / d1["Price_Buy"]
        d1["Position"] = "short"
    d1.index = d1["Date_Buy"]
    return d1


def annualised_sharpe(returns, N=245):
    return np.sqrt(N) * (returns.mean() / returns.std())


def get_kpi_run_strategy(df, input_dict):
    enter = input_dict['enter']
    exit = input_dict['exit']
    stoploss_needed = input_dict['stoploss_needed']
    k_enter = input_dict['k_enter']
    k_augm = input_dict['k_augm']
    k_exit = input_dict['k_exit']
    col_list = []
    long = 1
    threshold = 0.9
    df_cp = strategy(df, enter, exit, col_list, long,
                     threshold, stoploss_needed)
    GAIN = []
    for i in df.index:
        if i in list(df_cp.index):
            result = df_cp.loc[i]["Gain"]
        else:
            result = 0
        GAIN.append(result)
    GAIN_df = pd.Series(GAIN, index=df.index)
    algo_dict = {
        'stoploss_needed': stoploss_needed,
        'enter': k_enter,
        'augment': k_augm,
        'exit': k_exit,
        'kpi': GAIN_df.dropna().cumsum()[-1],
        'sharpe_ratio': annualised_sharpe(df_cp['Return'])
    }
    return algo_dict
