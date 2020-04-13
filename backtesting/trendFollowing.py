### Quantiacs Trend Following Trading System Example
# import necessary Packages below:
import numpy

def myTradingSystem(DATE, OPEN, HIGH, LOW, CLOSE, VOL, exposure, equity, settings):
    ''' This system uses trend following techniques to allocate capital into the desired equities'''

    nMarkets=CLOSE.shape[1]

    periodLong=200 #last 200 days
    periodShort=40 #last 40 days

    smaLong=numpy.nansum(CLOSE[-periodLong:,:],axis=0)/periodLong
    smaRecent=numpy.nansum(CLOSE[-periodShort:,:],axis=0)/periodShort

    settings['smaLong'] = smaLong #save data into settings field for long period data
    settings['smaRecent'] = smaRecent #save data into settings field for short period data

    # longEquity= numpy.array(smaRecent > smaLong)
    longEquity= smaRecent > smaLong
    shortEquity= ~longEquity

    pos=numpy.zeros(nMarkets)
    pos[longEquity]=1
    pos[shortEquity]=-1

    #pos = [0,1,-1,]#normalized transfer (+ is buy, - is sell)

    return pos, settings


def mySettings():
    ''' Define your trading system settings here '''

    settings= {}

    # S&P 100 stocks
    # settings['markets']=['CASH','AAPL','ABBV','ABT','ACN','AEP','AIG','ALL',
    # 'AMGN','AMZN','APA','APC','AXP','BA','BAC','BAX','BK','BMY','BRKB','C',
    # 'CAT','CL','CMCSA','COF','COP','COST','CSCO','CVS','CVX','DD','DIS','DOW',
    # 'DVN','EBAY','EMC','EMR','EXC','F','FB','FCX','FDX','FOXA','GD','GE',
    # 'GILD','GM','GOOGL','GS','HAL','HD','HON','HPQ','IBM','INTC','JNJ','JPM',
    # 'KO','LLY','LMT','LOW','MA','MCD','MDLZ','MDT','MET','MMM','MO','MON',
    # 'MRK','MS','MSFT','NKE','NOV','NSC','ORCL','OXY','PEP','PFE','PG','PM',
    # 'QCOM','RTN','SBUX','SLB','SO','SPG','T','TGT','TWX','TXN','UNH','UNP',
    # 'UPS','USB','UTX','V','VZ','WAG','WFC','WMT','XOM']

    # Futures Contracts

    settings['markets']  = ['CASH','F_AD', 'F_BO', 'F_BP', 'F_C', 'F_CC', 'F_CD',
    'F_CL', 'F_CT', 'F_DX', 'F_EC', 'F_ED', 'F_ES', 'F_FC','F_FV', 'F_GC',
    'F_HG', 'F_HO', 'F_JY', 'F_KC', 'F_LB', 'F_LC', 'F_LN', 'F_MD', 'F_MP',
    'F_NG', 'F_NQ', 'F_NR', 'F_O', 'F_OJ', 'F_PA', 'F_PL', 'F_RB', 'F_RU',
    'F_S','F_SB', 'F_SF', 'F_SI', 'F_SM', 'F_TU', 'F_TY', 'F_US','F_W', 'F_XX',
    'F_YM']
    settings['beginInSample'] = '20120506' #start date
    settings['endInSample'] = '20190410' #end date
    settings['lookback']= 504 #how many data points prior to last n days
    settings['budget']= 10**6 #$1M budget
    settings['slippage']= 0.05 #brokerage commisions, market slippage, inflation, etc.

    return settings

# Evaluate trading system defined in current file.
if __name__ == '__main__':
    import quantiacsToolbox
    results = quantiacsToolbox.runts('trendFollowing.py')
