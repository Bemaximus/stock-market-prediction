%{
	Read data from a csv and convert it into test and train data
%}


flatten = @(arr) reshape(arr, numel(arr), 1);

% Open up the csv of all JNUG data

% Oldest data comes first
jnugData = flipud(readmatrix('JNUG_Data.csv'));

jnugSize = size(jnugData)
clear jnugSize

trainJnugData = jnugData(1:end-200,:);

testJnugData = jnugData(end-199:end,:);

Y = zeros(size(trainJnugData, 1) - 11, size(trainJnugData, 2) * 10 + 1);
C = zeros(size(trainJnugData, 1) - 11, 1);

for i = 11:size(trainJnugData, 1)
	j = i - 10;
	
	% Last 10 days of data
	last10Days = flatten(trainJnugData(j:i-1,:));

	% This is the first opening price
	firstData = last10Days(1);
	
	% This is info for the day we're testing
	newOpenPrice = trainJnugData(i,1);
	newClosePrice = trainJnugData(i,4);

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

T = zeros(size(testJnugData, 1) - 11, size(testJnugData, 2) * 10 + 1);
Q = zeros(size(testJnugData, 1) - 11, 1);

for i = 11:size(testJnugData, 1)
	j = i - 10;
	
	% Last 10 days of data
	last10Days = flatten(testJnugData(j:i-1,:))

	% This is the first opening price
	firstData = last10Days(1);
	
	% This is info for the day we're testing
	newOpenPrice = testJnugData(i,1)
	newClosePrice = testJnugData(i,4)

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

save("jnug.mat", "Y", "C", "T", "Q", "jnugData")