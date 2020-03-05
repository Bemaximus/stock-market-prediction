function plotStockData(ticker)

flatten = @(arr) reshape(arr, 1, numel(arr))

% load data
load(strcat("../models/", ticker, ".mat")) % test and train data
load(strcat("../models/", ticker, "_predict.mat")) % model predictions


% histogram of percentage gains
figure(1)
A_percentageGains = (A - 1) * 100;
histogram(A_percentageGains, "BinMethod", "sqrt", "FaceColor", "green")
title("Percentage gained on " + ticker + " on a given day")
xlabel("Percentage gained (%)")
ylabel("Frequency")

% histogram of estimated percentage gains
figure(2)
D_percentageEstimates = abs(D) * 100;
histogram(D_percentageEstimates, "BinMethod", "sqrt", "FaceColor", "red")
title("Prediction error on " + ticker + " on a given day")
xlabel("Percentage error (%)")
ylabel("Frequency")

% line plot of open/close prices and prediction
figure(3)
% openClosePrices = flipud(stockData(1:11,:))
% the last 11 days of the stock, oldest first
flippedStockData = flipud(stockData);
candle(flippedStockData, "b")
title(ticker + " training and testing stock data")
datetick("x")
title(ticker + " stock trends over time")
xlabel("Date")
ylabel("Market Price (USD)")

figure(4)
last11daysMatrix = flippedStockData(end-14:end-4, :);
prepCandle(ticker, last11daysMatrix);

figure(5)
last11daysMatrix = flippedStockData(end-10:end, :);
prepCandle(ticker, last11daysMatrix);

hold off

end

function prepCandle(ticker, last11daysMatrix)

flatten = @(arr) reshape(arr, 1, numel(arr))

last10daysMatrix = last11daysMatrix(1:end-1, :);
multiplyFactor = last10daysMatrix(1,1);
last10daysMatrix = last10daysMatrix / multiplyFactor;
OpenPrices = last10daysMatrix(:,1);
HighPrices = last10daysMatrix(:,2);
LowPrices = last10daysMatrix(:,3);
ClosePrices = last10daysMatrix(:,4);

% colorOptions = 'rg';  % the two-vector from which to choose
% colors = {colorOptions( (ClosePrices >= OpenPrices) + 1 ))}  % select based on condition

% candle(HighPrices, LowPrices, ClosePrices, OpenPrices, colors)
bullCheck = OpenPrices <= ClosePrices;
c1 = candle(last10daysMatrix .* bullCheck * 100, 'g')
hold on
c2 = candle(last10daysMatrix .* ~bullCheck * 100, 'r')

title("Ten days of stock data for " + ticker + " with a prediction")
xlabel("Day")
ylabel("Percentage Change (%)")

todayStock = last11daysMatrix(end, :) / multiplyFactor;
todayOpening = todayStock(1);
todayClosing = todayStock(4);

load("../models/" + ticker + "_predict.mat", "m");

prevData = flatten(last10daysMatrix)
prevData(end + 1) = todayOpening
estPercGain = prevData * m
estClosing = todayOpening * estPercGain;

todayOpening = todayOpening * 100;
todayClosing = todayClosing * 100;
estClosing = estClosing * 100;

sc1 = scatter(11, todayOpening, 'bs', 'filled', "DisplayName", "Today's Opening Price")
sc2 = scatter(11, todayClosing, 'co', 'filled', "DisplayName", "Today's Closing Price")
sc3 = scatter(11,   estClosing, 'md', 'filled', "DisplayName", "Estimated Closing Price")

% text(11, todayOpening, "Today's Opening Price   ", "HorizontalAlignment", "right");
% text(11, todayClosing, "Today's Closing Price   ", "HorizontalAlignment", "right");
% text(11, estClosing, "Estimated Closing Price   ", "HorizontalAlignment", "right");

% legend(sc1, "Today's Opening Price")
% legend(sc2, "Today's Closing Price")
% legend(sc3, "Estimated Closing Price")
% legend("1", "2", "3")
legend([sc1 sc2 sc3], "Location", "best")

portfolioGain = (estPercGain - 1) * (todayClosing / todayOpening - 1) + 1

end