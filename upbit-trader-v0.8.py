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


class CryptoTrader:
    """
    업비트 암호화폐 자동매매 클래스
    """
    def __init__(self, access_key, secret_key):
        """
        트레이더 초기화
        
        Args:
            access_key (str): 업비트 API access key
            secret_key (str): 업비트 API secret key
        """
        self.access_key = access_key
        self.secret_key = secret_key
        self.upbit = pyupbit.Upbit(access_key, secret_key)
        self.upbit_balances = {}
        self.refresh_balances()
        
        # 트레이딩 설정
        self.BASIC_UNIT = 5500
        self.MAXIMUM_BUY_PER_COIN = 100000
        self.ticker_list = ['BTC', 'ETH', 'DOGE', 'TRUMP', 'XLM']
        self.last_bought = {}
        
    def refresh_balances(self):
        """잔고 정보 갱신"""
        try:
            self.upbit_balances = {balance['currency']: balance for balance in self.upbit.get_balances()}
            
            # 각 티커에 대해 기본값 설정
            for ticker in self.ticker_list:
                if ticker not in self.upbit_balances:
                    self.upbit_balances[ticker] = {'avg_buy_price': 0, 'balance': 0}
        except Exception as e:
            print(f"잔고 정보 갱신 실패: {e}")
    
    def get_balance_krw(self, ticker):
        """
        특정 티커의 원화 평가액 반환
        
        Args:
            ticker (str): 코인 티커
            
        Returns:
            float: 원화 평가액
        """
        if ticker in self.upbit_balances:
            ticker_balance = self.upbit_balances[ticker]
            balance_krw = float(ticker_balance['avg_buy_price']) * float(ticker_balance['balance'])
            return balance_krw
        return 0.0
    
    def get_current_price(self, ticker):
        """
        현재 가격 조회
        
        Args:
            ticker (str): 코인 티커
            
        Returns:
            float: 현재 가격
        """
        krw_ticker = f"KRW-{ticker}"
        return float(pyupbit.get_current_price(krw_ticker))
    
    def get_current_balance(self, ticker):
        """
        현재 보유 수량 조회
        
        Args:
            ticker (str): 코인 티커
            
        Returns:
            float: 보유 수량
        """
        return float(self.upbit_balances[ticker]['balance'])
    
    def buy_coin(self, ticker, won=None):
        """
        코인 매수
        
        Args:
            ticker (str): 코인 티커
            won (float, optional): 매수 금액. Defaults to BASIC_UNIT.
        """
        if won is None:
            won = self.BASIC_UNIT
            
        # 최대 매수 금액 초과 체크
        if self.get_balance_krw(ticker) > self.MAXIMUM_BUY_PER_COIN - won:
            return
        
        # 1시간 내 추가 매수 제한
        dtnow = datetime.datetime.now()
        if ticker in self.last_bought and dtnow - self.last_bought[ticker] < datetime.timedelta(hours=1):
            return
        
        krw_ticker = f"KRW-{ticker}"
        try:
            ret = self.upbit.buy_market_order(krw_ticker, won)
            self.last_bought[ticker] = dtnow
            print(f"---매수 주문 완료: {ret}")
        except Exception as e:
            print(f"---매수 주문 실패: {e}")
    
    def sell_coin(self, ticker, won=None):
        """
        코인 매도
        
        Args:
            ticker (str): 코인 티커
            won (float, optional): 매도 금액. Defaults to BASIC_UNIT.
        """
        if won is None:
            won = self.BASIC_UNIT
            
        # 보유 금액이 매도 금액보다 적으면 반환
        if self.get_balance_krw(ticker) < won:
            return
        
        krw_ticker = f"KRW-{ticker}"
        try:
            amount = self.get_current_balance(ticker)
            ret = self.upbit.sell_market_order(krw_ticker, amount)
            print(f"---매도 주문 완료: {ret}")
        except Exception as e:
            print(f"---매도 주문 실패: {e}")
    
    def add_indicators(self, df):
        """
        DataFrame에 기술적 지표 추가
        
        Args:
            df (DataFrame): OHLCV 데이터프레임
            
        Returns:
            DataFrame: 지표가 추가된 데이터프레임
        """
        # RSI 계산
        df['RSI_14'] = ta.rsi(df['close'], length=14)
        df['RSI_24'] = ta.rsi(df['close'], length=24)
        df['RSI_60'] = ta.rsi(df['close'], length=60)

        # MACD 계산
        df['MACD'] = df['close'].ewm(span=12, adjust=False).mean() - df['close'].ewm(span=26, adjust=False).mean()
        df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Signal'] = np.where(df['MACD'] > df['Signal_Line'], 1, -1)

        # 볼린저 밴드 계산
        df['Middle_Band'] = df['close'].rolling(window=20).mean()
        df['Upper_Band'] = df['Middle_Band'] + 2 * df['close'].rolling(window=20).std()
        df['Lower_Band'] = df['Middle_Band'] - 2 * df['close'].rolling(window=20).std()

        return df
    
    def get_trade_signal_multi_tf(self, ticker):
        """
        멀티 타임프레임 기반 매매 신호 생성
        
        Args:
            ticker (str): 코인 티커
            
        Returns:
            int: 매매 신호 (1: 매수, 0: 관망, -1: 매도)
        """
        krw_ticker = f"KRW-{ticker}"
        
        # 시간대별 데이터 수집
        df_hours = pyupbit.get_ohlcv(krw_ticker, "hours1", count=24*14)
        df_minutes = pyupbit.get_ohlcv(krw_ticker, "minute5", count=12*24*7)

        # 보조지표 추가
        df_hours = self.add_indicators(df_hours)
        df_minutes = self.add_indicators(df_minutes)

        # 최신 데이터 추출
        rsi_hour = df_hours['RSI_24'].iloc[-1]
        macd_signal_hour = df_hours['MACD_Signal'].iloc[-1]

        # RSI 임계값 계산
        rsi_lower_bound_1 = np.percentile(df_hours['RSI_24'].iloc[24:].values, 30)
        rsi_upper_bound_1 = np.percentile(df_hours['RSI_24'].iloc[24:].values, 80)
        rsi_lower_bound_2 = np.percentile(df_minutes['RSI_60'].iloc[60:].values, 30)
        rsi_upper_bound_2 = np.percentile(df_minutes['RSI_60'].iloc[60:].values, 70)

        rsi_min = df_minutes['RSI_60'].iloc[-1]
        close_price_min = df_minutes['close'].iloc[-1]
        lower_band_min = df_minutes['Lower_Band'].iloc[-1]
        upper_band_min = df_minutes['Upper_Band'].iloc[-1]

        # 매수 조건
        if (macd_signal_hour == 1 and 
            rsi_hour <= rsi_lower_bound_1 and 
            rsi_min <= rsi_lower_bound_2 and 
            close_price_min <= lower_band_min):
            return 1

        # 매도 조건
        elif (macd_signal_hour == -1 and 
              rsi_hour >= rsi_upper_bound_1 and 
              rsi_min >= rsi_upper_bound_2 and 
              close_price_min >= upper_band_min):
            return -1

        # 관망
        else:
            return 0
    
    def make_decision_and_execute(self):
        """매매 결정 및 실행"""
        self.refresh_balances()
        
        print('-' * 120)
        print(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} CASH\t보유KRW: {float(self.upbit_balances['KRW']['balance']):.2f} KRW")
        
        for ticker in self.ticker_list:
            try: 
                # 매매 신호 생성
                rsi_signal = self.get_trade_signal_multi_tf(ticker)
                
                # 매매 실행
                if rsi_signal > 0:
                    self.buy_coin(ticker)
                    decision_str = 'BUY'
                elif rsi_signal < 0:
                    self.sell_coin(ticker)
                    decision_str = 'SELL'
                else:
                    decision_str = 'HOLD'
                
                # 현재 상태 출력
                current_price = self.get_current_price(ticker)
                ticker_balance = self.upbit_balances[ticker]
                current_avg_price = float(ticker_balance['avg_buy_price']) if ticker_balance['avg_buy_price'] != 0 else current_price
                
                print(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} {ticker}\t"
                      f"보유KRW: {current_price * float(ticker_balance['balance']):10.2f}\t"
                      f"판단: {decision_str}\t"
                      f"비율: {str(current_price/current_avg_price)[:6]}\t"
                      f"현재 가격/매수 가격: {current_price} / {str(current_avg_price)[:len(str(current_price))]}")
                
                # 손절매 (10% 손실 시)
                if current_price/current_avg_price < 0.9:
                    self.sell_coin(ticker)
                    
            except Exception as e:
                print(f'ERROR: {e}')


if __name__ == "__main__":
    # API 키 설정 (보안을 위해 환경 변수로 대체)
    ACCESS_KEY = os.getenv("UPBIT_ACCESS_KEY", "YOUR_ACCESS_KEY")
    SECRET_KEY = os.getenv("UPBIT_SECRET_KEY", "YOUR_SECRET_KEY")
    
    # 트레이더 인스턴스 생성
    trader = CryptoTrader(ACCESS_KEY, SECRET_KEY)
    
    # 매매 실행
    while True:
        trader.make_decision_and_execute()
