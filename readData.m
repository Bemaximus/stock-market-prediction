%{
	Read data from a csv and convert it into test and train data
%}

function [Y C T Q] = readData(ticker, varargin)
	flatten = @(arr) reshape(arr, numel(arr), 1);

	% Open up the csv of all JNUG data

	% Oldest data comes first
	% stockData = flipud(readmatrix(strcat("../../data/", ticker, ".csv")));
	stockData = readmatrix(strcat("../../data/", ticker, ".csv"));

	if size(stockData, 2) == 7
		stockData = stockData(:,2:5);
	end

	trainStockData = stockData(1:end-250,:);

	testStockData = stockData(end-249:end,:);

	Y = zeros(size(trainStockData, 1) - 11, size(trainStockData, 2) * 10 + 1);
	C = zeros(size(trainStockData, 1) - 11, 1);

	for i = 11:size(trainStockData, 1)
		j = i - 10;
		
		% Last 10 days of data
		last10Days = flatten(trainStockData(j:i-1,:));

		% This is the first opening price
		firstData = last10Days(1);
		
		% This is info for the day we're testing
		newOpenPrice = trainStockData(i,1);
		newClosePrice = trainStockData(i,4);

		% This is the percentage increase in a day
		% What we're looking for
		percentageIncrease = newClosePrice / newOpenPrice;

		% Combine the data for previous days and current opening price
		last10Days(end + 1) = newOpenPrice;
		
		% Only capture percentage data, remove dollars
		last10Days = last10Days / firstData;

		% Row j is this day's analysis data
		Y(j, :) = last10Days;
		C(j) = percentageIncrease;
	end

	T = zeros(size(testStockData, 1) - 11, size(testStockData, 2) * 10 + 1);
	Q = zeros(size(testStockData, 1) - 11, 1);

	for i = 11:size(testStockData, 1)
		j = i - 10;
		
		% Last 10 days of data
		last10Days = flatten(testStockData(j:i-1,:));

		% This is the first opening price
		firstData = last10Days(1);
		
		% This is info for the day we're testing
		newOpenPrice = testStockData(i,1);
		newClosePrice = testStockData(i,4);

		% This is the percentage increase in a day
		% What we're looking for
		percentageIncrease = newClosePrice / newOpenPrice;

		% Combine the data for previous days and current opening price
		last10Days(end + 1) = newOpenPrice;

		% Only capture percentage data, remove dollars
		last10Days = last10Days / firstData;

		% Row j is this day's analysis data
		T(j,:) = last10Days;
		Q(j) = percentageIncrease;
	end
	Y
	C
	T
	Q
	save(strcat("./models/", ticker,".mat"), "Y", "C", "T", "Q", "stockData")
end