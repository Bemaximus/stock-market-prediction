
%{
	KAZDAQ
%}

ticker = "DRIP"
todayOpening = 5.02; % input today's opening price

% https://eodhistoricaldata.com/api/eod/AAPL.US?from=2017-01-05&to=2017-02-10&api_token=OeAFFmMliFG5orCUuwAKQ8l4WWFQ67YX&period=d

format compact

[Y C T Q] = readData(ticker);

% load(strcat(ticker, ".mat"))
% This would include Y, C, T, and Q

[P A] = predictMarket(Y,C,T,Q, ticker);

% [status cmdout] = system("curl https://eodhistoricaldata.com/api/eod/AAPL.US?from=2017-01-05&to=2017-02-10&api_token=OeAFFmMliFG5orCUuwAKQ8l4WWFQ67YX&period=d")

% Predict today's price

% Load the stock data
load(strcat("models/", ticker, ".mat"), "stockData");
last10days = stockData(1:10, :);

flatten = @(arr) reshape(arr, numel(arr), 1);

last10days = flatten(last10days);
last10days(end + 1) = todayOpening;
last10days = last10days / last10days(1)

% Load the model
load(strcat("models/", ticker, "_predict.mat"), "m");

estimatedIncrease = transpose(last10days) * m

P
disp("hi")
