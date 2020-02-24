
%{
	KAZDAQ
%}


% https://eodhistoricaldata.com/api/eod/AAPL.US?from=2017-01-05&to=2017-02-10&api_token=OeAFFmMliFG5orCUuwAKQ8l4WWFQ67YX&period=d

load("jnug.mat")
% This would include Y, C, T, and Q

[P A] = predictMarket(Y,C,T,Q)
