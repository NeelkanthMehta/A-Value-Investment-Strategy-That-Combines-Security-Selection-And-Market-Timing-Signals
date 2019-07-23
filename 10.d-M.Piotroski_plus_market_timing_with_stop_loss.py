"""
THIS IS COMBINATION OF PIOTROSKI ALGOROTHM THAT SCREENS VALUE SECURITIES AND TECHINCAL INDICATORS OF RSI, MFI and ADX
1. Piotroski F-Score is a no. between 0 - 9 which is used to assess strength of company's financial position[1]
   Modified Piotroski is a no. between 0 - 10, used to assess strength of company's financial position[2]
   The score is calculated based on 9 criteria divided into 3 groups:
    
   Profitability:
       Return on Assets (1 point if it is positive in the current year, 0 otherwise);
       Free Cash Flow to Assets (1 point if it is positive in the current year, 0 otherwise);
       Change in Return of Assets (ROA) (1 point if ROA is higher in the current year compared to the previous one, 0 otherwise);
       Accruals (1 point if Operating Cash Flow/Total Assets is higher than ROA in the current year, 0 otherwise);
       Change in Free Cash Flow to Assets (1 point if it is positive in the current year, 0 otherwise);
   
   Leverage, Liquidity and Source of Funds
       Change in Leverage (long-term) ratio (1 point if the ratio is lower this year compared to the previous one, 0 otherwise);
       Change in Current ratio (1 point if it is higher in the current year compared to the previous one, 0 otherwise);
       Change in the net number of shares outstanding (1 point if no new shares were issued during the last year);
   
   Operating Efficiency
       Change in Gross Margin (1 point if it is higher in the current year compared to the previous one, 0 otherwise);
       Change in Asset Turnover ratio (1 point if it is higher in the current year compared to the previous one, 0 otherwise);

2. Relative Strength Index (RSI):
   Classified as a momentum Oscillator, RSI charts the current and historical strength or weakness of a stock or market based on the closing prices of a recent trading period[3]
   
   a. n-period Smoothed/ Modified Moving Average --> SMMA
   b. RS = SMMA(U, n)/ SMMA(U, n)
   c. RSI = 100 - (100/ (1 + RS))

3. Money Flow Index (MFI):
   Again an Oscilator, is used to show the money flow (an approximation of the dollar value of a day's trading) over several days [4].
   Steps to calculate are:
   a. typical price = (high + low + close)/ 3
   b. money flow = typical price x volume
      Positive money flow: days when money flow is higher than previous day's money flow
      Negative money flow: days when money flow is lower than previous day's money flow
   c. money ratio = Positive money flow/ Negative money flow
   d. MFI = 100 x (Positive money flow/ (Positive money flow + Negative money flow))

4. Average Directional Movement Index (ADX):
   Indicates strength in current series of prices of a financial instrument. It is a combination of two other positive directional indicator positive/ negative Directional Indicator (or +/ -DI)[5].
   UpMove = h_t - h_(t-1)
   DownMove = l_{t-1} - l_t
   if UpMove > DownMove & UpMove > 0, then +DM = UpMove, else +DM = 0
   if DownMove > UpMove & DownMove > , then -DM = DownMove, else -DM = 0
   
   After selecting number of periods, +DI and -DI are:
       +DI = 100 x SMMA(+DM)/ Average True Range
       -DI = 100 x SMMA(-DM)/ Average True Range

ref:
    1. https://en.wikipedia.org/wiki/Piotroski_F-Score
    2. https://www.aaii.com/journal/article/simple-methods-to-improve-the-piotroski-f-score
    3. https://en.wikipedia.org/wiki/Relative_strength_index
    4. https://en.wikipedia.org/wiki/Money_flow_index
    5. https://en.wikipedia.org/wiki/Average_directional_movement_index
"""

# Importing all the necessary libraries and modules
import numpy as np
import pandas as pd
import talib
import quantopian.algorithm as algo
import quantopian.optimize as opt
from sklearn import preprocessing
from scipy.stats.mstats import winsorize
from quantopian.pipeline import factors
from quantopian.pipeline import filters
from quantopian.pipeline import Pipeline
from quantopian.pipeline.data.builtin import USEquityPricing
from quantopian.pipeline.filters import QTradableStocksUS
from quantopian.pipeline.experimental import risk_loading_pipeline
from quantopian.pipeline.data import Fundamentals
import quantopian.pipeline.factors as Factors

# Defining universal variables
WIN_LIMIT = 0.0
MAX_GROSS_LEVERAGE = 1.0
NUM_LONG_POSITIONS = 25
NUM_SHORT_POSITIONS= 25
# MAX_LONG_POSITION__SIZE = 0.5/ NUM_LONG_POSITIONS
# MAX_SHORT_POSITION_SIZE = -0.5/ NUM_SHORT_POSITIONS

# defining winsorizing function
def preprocess(a):
    a = np.nan_to_num(a - np.nanmean(a))
    a = winsorize(a, limits=[WIN_LIMIT, WIN_LIMIT])
    return preprocessing.scale(a)


# Piotroski 9-pt criteria
class Piotroski(factors.CustomFactor):
    inputs = [
        Fundamentals.roa,
        Fundamentals.free_cash_flow,
        Fundamentals.total_assets,
        Fundamentals.cash_flow_from_continuing_operating_activities,
        Fundamentals.long_term_debt_equity_ratio,
        Fundamentals.current_ratio,
        Fundamentals.shares_outstanding,
        Fundamentals.gross_margin,
        Fundamentals.assets_turnover
    ]
    
    window_length = 252
    
    def compute(self, today, assets, out, roa, cash_flow, total_assets, cash_flow_from_ops, long_term_debt_ratio, current_ratio, shares_outstanding, gross_margin, assets_turnover):
        
        profit = (
            (roa[-1] > 0).astype(int) +
            ((cash_flow[-1]/ total_assets[-1]) > 0).astype(int) +
            ((cash_flow[-1]/ total_assets[-1]) > (cash_flow[0]/ total_assets[0])).astype(int) +
            (roa[-1] > roa[0]).astype(int) +
            (cash_flow_from_ops[-1] > roa[-1]).astype(int)
        )
        
        leverage = (
            (long_term_debt_ratio[-1] < long_term_debt_ratio[0]).astype(int) +
            (shares_outstanding[-1] <= shares_outstanding[0]).astype(int) +
            (current_ratio[-1] > current_ratio[0]).astype(int)
        )
        
        operating = (
            (gross_margin[-1] > gross_margin[0]).astype(int) +
            (assets_turnover[-1] > assets_turnover[0]).astype(int)
        )
        
        out[:] = profit + leverage + operating


# Money Flow Index (MFI) indicator
class MFI(Factors.CustomFactor):
    """
    Money Flow Index
    Volume Indicator
    **Default Inputs:**  USEquityPricing.high, USEquityPricing.low, USEquityPricing.close, USEquityPricing.volume
    **Default Window Length:** 15 (14 + 1-day for difference in prices)
    http://www.fmlabs.com/reference/default.htm?url=MoneyFlowIndex.htm
    """

    inputs = [USEquityPricing.high, USEquityPricing.low, USEquityPricing.close, USEquityPricing.volume]
    window_length = 15

    def compute(self, today, assets, out, high, low, close, vol):

        # calculate typical price
        typical_price = (high + low + close) / 3.

        # calculate money flow of typical price
        money_flow = typical_price * vol

        # get differences in daily typical prices
        tprice_diff = (typical_price - np.roll(typical_price, 1, axis=0))[1:]

        # create masked arrays for positive and negative money flow
        pos_money_flow = np.ma.masked_array(money_flow[1:], tprice_diff < 0, fill_value = 0.)
        neg_money_flow = np.ma.masked_array(money_flow[1:], tprice_diff > 0, fill_value = 0.)

        # calculate money ratio
        money_ratio = np.sum(pos_money_flow, axis=0) / np.sum(neg_money_flow, axis=0)

        # MFI
        out[:] = 100. - (100. / (1. + money_ratio))


# Initializing algorithms and routines therein
def initialize(context):
    """
    Called once at the start of the algorithm.
    """
    set_benchmark(symbol('SPY'))
    set_commission(commission.PerTrade(cost=0.0))
    set_slippage(slippage.VolumeShareSlippage(volume_limit=1, price_impact=0))
    set_long_only()
    # set_max_leverage(1.1)
    
    # Rebalance every day, 1 hour after market open.
    algo.schedule_function(rebalance, algo.date_rules.every_day(), algo.time_rules.market_open(hours=1),)

    # Record tracking variables at the end of each day.
    algo.schedule_function(record_vars, algo.date_rules.every_day(), algo.time_rules.market_close(),)
    
    # Create our dynamic stock selector.
    algo.attach_pipeline(make_pipeline(), 'piotroski')
    algo.attach_pipeline(risk_loading_pipeline(), 'risk_factors')


# Defining a Quantopian pipeline
def make_pipeline():
    """
    A function to create our dynamic stock selector (pipeline). Documentation
    on pipeline can be found here:
    https://www.quantopian.com/help#pipeline-title
    """

    # Base universe set to the QTradableStocksUS
    base_universe = QTradableStocksUS()

    # For Piotroski, we need stocks with Book-to-Market >= 1
    book = Fundamentals.book_value_per_share.latest
    market = Fundamentals.market_cap.latest/ Fundamentals.shares_outstanding.latest
    book_to_market = book/ market
    p_universe = book_to_market >= 1
    
    # CustomFactor of Piotroski F-Score
    f_score = Piotroski()
    
    # Filtering top and bottom 25 stocks
    longs  = f_score.eq(9) or f_score.eq(8)  #f_score.top(25, mask=base_universe)
    shorts = f_score.eq(0) #f_score.bottom(25, mask=base_universe)
    
    universe = base_universe & p_universe & ( longs| shorts)

    pipe = Pipeline(
        columns={
            'longs': longs,
            'shorts': shorts,
            'f_score': f_score,
        },
        screen=universe
    )
    return pipe


def before_trading_start(context, data):
    """
    Called every day before market open.
    """
    try:
        # generates output
        context.output = algo.pipeline_output('piotroski')
        
        # loads risk factors
        context.risk_loadings = algo.pipeline_output('risk_factors')
        
        # records position
        record(cash=context.portfolio.cash, asset=context.portfolio.portfolio_value, ) 

        # These are the securities that we are interested in trading each day.
        context.security_list = context.output.index.tolist()
    except Exception as e:
        print(str(e))


def rebalance(context, data):
    """
    Execute orders according to our schedule_function() timing.
    """
    #: Obtaining Prices
    # prices = data.history(context.security_list, 'price', 252, '1d')
    
    #: Compute for portfolio weights
    long_secs = context.output[context.output['longs']].index
    long_weight = (1.0/ max(len(long_secs), 1))
    
    short_secs = context.output[context.output['shorts']].index
    short_weight = 0 #(0.0/ max(len(short_secs), 1))
    
    # Open our long position.
    for security in long_secs:
        try:
            RSI_daily = factors.RSI(inputs = [USEquityPricing.close], window_length = 14)
            
            MFI_Daily = MFI()
            
            # Defining ADX
            period = 30
            H = data.history(security,'high', 2*period,'1d') #data.history should be changed right?
            L = data.history(security,'low', 2*period,'1d')
            C = data.history(security,'price', 2*period,'1d')
            ta_ADX = talib.ADX(H, L, C, period)  
            ta_nDI = talib.MINUS_DI(H, L, C, period)  
            ta_pDI = talib.PLUS_DI(H, L, C, period)  
            ADX = ta_ADX[-1]
            nDI = ta_nDI[-1]
            pDI = ta_pDI[-1]

            # defining Long and short positions position criteria
            if data.can_trade(security) and (RSI_daily < 30) and (MFI_Daily < 20) or ((ADX > 20) & (pDI > nDI)):
                log.info("Going long on stock %s"%(security.symbol))
                order_target_percent(security, long_weight)
            
            if data.can_trade(security) and (RSI_daily > 70) and (MFI_Daily > 80) or ((ADX > 25) & (pDI < nDI)):
                log.info("Going short on stock %s"%(security.symbol))
                order_target_percent(security, 0)
        except:
            pass
   
    # Closing the position
    for security in context.portfolio.positions:
        if data.can_trade(security) and security not in long_secs and security not in short_secs:
            order_target_percent(security, 0)


def record_vars(context, data):
    """
    Plot variables at the end of each day.
    """
    long_count = 0
    short_count = 0
    
    for position in context.portfolio.positions.itervalues():
        if position.amount > 0:
            long_count += 1
        if position.amount < 0:
            short_count += 1


def handle_data(context, data):
    """
    This inbuilt function is called every minute.
    Here, stop loss is defined, so that it runs every minute
    
    The logic is that if a security's price is 10% lower than the price it was purchased, a stoplimit  order is triggred. 
    # """
    context.security_list = context.output.index.tolist()
    for security in context.security_list:
        current_price = data.current(security, 'price')
        position = context.portfolio.positions[security].amount
        price_position = context.portfolio.positions[security].cost_basis
        if (position > 0) and (current_price < price_position * 0.9):
            order_target_percent(security, 0, style=LimitOrder(current_price))
            log.info('Sell with stop loss hit' + str(security.symbol))
