function plotStockData(ticker)

% load data
load(strcat("models/", ticker, ".mat")) % test and train data
load(strcat("models/", ticker, "_predict.mat")) % model predictions


% histogram of percentage gains
figure(1)
A_percentageGains = (A - 1) * 100;
histogram(A_percentageGains, "BinMethod", "sqrt", "FaceColor", "#0066ff")
title("Percentage gained on " + ticker + " on a given day")
xlabel("Percentage gained (%)")
ylabel("Frequency")

% histogram of estimated percentage gains
figure(2)
D_percentageEstimates = abs(D) * 100
histogram(D_percentageEstimates, "BinMethod", "sqrt", "FaceColor", "#00ff66")
title("Prediction error on " + ticker + " on a given day")
xlabel("Percentage error (%)")
ylabel("Frequency")

% line plot of open/close prices and prediction
figure(3)
% openClosePrices = flipud(stockData(1:11,:))
% the last 11 days of the stock, oldest first
flippedStockData = flipud(stockData)
candle(flippedStockData, "b")
title(ticker + " training and testing stock data")
datetick("x")

figure(4)
candle(flippedStockData(end-10:end, :), "r")

predictedPrice = G(end-1) * flippedStockData(end)

annotation("new estimated price: $" + predictedPrice)
datetick("x")
end