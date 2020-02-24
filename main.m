
%{
	KAZDAQ
%}

ticker = "PCG";

% https://eodhistoricaldata.com/api/eod/AAPL.US?from=2017-01-05&to=2017-02-10&api_token=OeAFFmMliFG5orCUuwAKQ8l4WWFQ67YX&period=d

[Y C T Q] = readData(ticker);

% load(strcat(ticker, ".mat"))
% This would include Y, C, T, and Q

[P A] = predictMarket(Y,C,T,Q, ticker)

% [status cmdout] = system("curl https://eodhistoricaldata.com/api/eod/AAPL.US?from=2017-01-05&to=2017-02-10&api_token=OeAFFmMliFG5orCUuwAKQ8l4WWFQ67YX&period=d")

disp(P)