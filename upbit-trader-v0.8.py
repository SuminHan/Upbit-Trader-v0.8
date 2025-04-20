import os
import time
import json
import pyupbit
import argparse
import datetime
import pandas as pd
import pandas_ta as ta
import schedule
import numpy as np

# 설정값
BASIC_UNIT = 5500  # 기본 매매 단위 (원)
MAXIMUM_BUY_PER_COIN = 100000  # 코인당 최대 매수 금액

# 매매할 코인 목록
ticker_list = ['BTC', 'ETH', 'DOGE', 'TRUMP', 'XLM']

# 업비트 API 키 설정
access_key = os.getenv("UPBIT_ACCESS_KEY", "YOUR_ACCESS_KEY")
secret_key = os.getenv("UPBIT_SECRET_KEY", "YOUR_SECRET_KEY")
upbit = pyupbit.Upbit(access_key, secret_key)

# 전역 변수
last_bought = dict()
upbit_balances = {balance['currency']: balance for balance in upbit.get_balances()}


def refresh_balances():
    """잔고 정보를 새로고침하는 함수"""
    global upbit_balances
    upbit_balances = {balance['currency']: balance for balance in upbit.get_balances()}
    for ticker in ticker_list:
        if ticker not in upbit_balances:
            upbit_balances[ticker] = {'avg_buy_price': 0, 'balance': 0}

    
def get_balance_krw(ticker):
    """특정 코인의 KRW 기준 보유 금액을 반환하는 함수"""
    if ticker in upbit_balances:
        ticker_balance = upbit_balances[ticker]
        balance_krw = float(ticker_balance['avg_buy_price']) * float(ticker_balance['balance'])
        return balance_krw
    else:
        return 0

    
def get_current_price(ticker):
    """특정 코인의 현재 가격을 조회하는 함수"""
    krw_ticker = "KRW-" + ticker
    return float(pyupbit.get_current_price(krw_ticker))


def get_current_balance(ticker):
    """특정 코인의 현재 보유 수량을 반환하는 함수"""
    return float(upbit_balances[ticker]['balance'])

    
def buy_coin(ticker, won=BASIC_UNIT):
    """코인을 매수하는 함수"""
    global transaction_count, current_bought_price
    
    # 최대 매수 금액을 초과하는 경우 리턴
    if get_balance_krw(ticker) > MAXIMUM_BUY_PER_COIN - won:
        return
    
    # 1시간 이내에 이미 매수한 경우 리턴
    dtnow = datetime.datetime.now()
    if ticker in last_bought and last_bought[ticker] - dtnow < datetime.timedelta(hours=1):
        return
    
    krw_ticker = "KRW-" + ticker
    try:
        ret = upbit.buy_market_order(krw_ticker, won)
        last_bought[ticker] = dtnow
        print(f"---매수 주문 완료: {ret}")
    except Exception as e:
        print(f"---매수 주문 실패: {e}")
        

def sell_coin(ticker, won=BASIC_UNIT):
    """코인을 매도하는 함수"""
    global transaction_count, current_bought
    
    # 보유 금액이 부족한 경우 리턴
    if get_balance_krw(ticker) < won:
        return
    
    krw_ticker = "KRW-" + ticker
    try:
        amount = get_current_balance(ticker)
        ret = upbit.sell_market_order(krw_ticker, amount)
        print(f"---매도 주문 완료: {ret}")
    except Exception as e:
        print(f"---매도 주문 실패: {e}")

    
def get_trade_signal_multi_tf(ticker):
    """여러 시간 프레임을 고려하여 매매 신호를 생성하는 함수"""
    krw_ticker = "KRW-" + ticker
    df_hours = pyupbit.get_ohlcv(krw_ticker, "hours1", count=24*14)
    df_minutes = pyupbit.get_ohlcv(krw_ticker, "minute5", count=12*24*7)

    def add_indicators(df):
        """데이터프레임에 기술적 지표를 추가하는 내부 함수"""
        # RSI
        df['RSI_14'] = ta.rsi(df['close'], length=14)
        df['RSI_24'] = ta.rsi(df['close'], length=24)
        df['RSI_60'] = ta.rsi(df['close'], length=60)

        # MACD
        df['MACD'] = df['close'].ewm(span=12, adjust=False).mean() - df['close'].ewm(span=26, adjust=False).mean()
        df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Signal'] = np.where(df['MACD'] > df['Signal_Line'], 1, -1)  # 상승 교차(1), 하락 교차(-1)

        # Bollinger Bands
        df['Middle_Band'] = df['close'].rolling(window=20).mean()
        df['Upper_Band'] = df['Middle_Band'] + 2 * df['close'].rolling(window=20).std()
        df['Lower_Band'] = df['Middle_Band'] - 2 * df['close'].rolling(window=20).std()

        return df

    # 보조지표 적용
    df_hours = add_indicators(df_hours)
    df_minutes = add_indicators(df_minutes)

    # 최신 데이터 가져오기
    rsi_hour = df_hours['RSI_24'].iloc[-1]
    macd_signal_hour = df_hours['MACD_Signal'].iloc[-1]

    # RSI 상하한선 설정
    rsi_lower_bound_1 = np.percentile(df_hours['RSI_24'].iloc[24:].values, 30)
    rsi_upper_bound_1 = np.percentile(df_hours['RSI_24'].iloc[24:].values, 80)
    rsi_lower_bound_2 = np.percentile(df_minutes['RSI_60'].iloc[60:].values, 30)
    rsi_upper_bound_2 = np.percentile(df_minutes['RSI_60'].iloc[60:].values, 70)

    rsi_min = df_minutes['RSI_60'].iloc[-1]
    close_price_min = df_minutes['close'].iloc[-1]
    lower_band_min = df_minutes['Lower_Band'].iloc[-1]
    upper_band_min = df_minutes['Upper_Band'].iloc[-1]

    # 매수 조건 (BUY)
    if macd_signal_hour == 1 and rsi_hour <= rsi_lower_bound_1 and rsi_min <= rsi_lower_bound_2 and close_price_min <= lower_band_min:
        return 1

    # 매도 조건 (SELL)
    elif macd_signal_hour == -1 and rsi_hour >= rsi_upper_bound_1 and rsi_min >= rsi_upper_bound_2 and close_price_min >= upper_band_min:
        return -1

    # 보류 (HOLD)
    else:
        return 0


def make_decision_and_execute():
    """잔고를 업데이트하고 매매 결정을 내린 후 실행하는 함수"""
    refresh_balances()
    
    print('-'*120)
    print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'CASH',
                  '\t보유KRW: ', f"{float(upbit_balances['KRW']['balance']):.2f}", 'KRW')
    
    for ticker in ticker_list:
        try: 
            rsi_signal = get_trade_signal_multi_tf(ticker)
            decision_str = ''
            
            if rsi_signal > 0:
                buy_coin(ticker)
                decision_str = 'BUY'
            elif rsi_signal < 0:
                sell_coin(ticker)
                decision_str = 'SELL'
            else:
                decision_str = 'HOLD'
        
            current_price = get_current_price(ticker)
            ticker_balance = upbit_balances[ticker]
            current_avg_price = float(ticker_balance['avg_buy_price']) if ticker_balance['avg_buy_price'] != 0 else current_price
            
            print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), ticker, 
                  '\t보유KRW: ', f'{current_price*float(ticker_balance['balance']):10.2f}',
                  '\t판단: ', decision_str,
                  '\t비율: ', str(current_price/current_avg_price)[:6],
                  '\t현재 가격/매수 가격: ', str(current_price) , '/', str(current_avg_price)[:len(str(current_price))])
            
            # 평균 매수가 대비 현재가가 90% 미만일 경우 손절
            if current_price/current_avg_price < 0.9:
                sell_coin(ticker)
        except Exception as e:
            print('ERROR:', e)


if __name__ == "__main__":
    # 지속적으로 실행
    while True:
        make_decision_and_execute()
