
function [P A] = predictMarket(Y, C, T, Q, ticker)
	
	% Make a value constrained to the region [minVal maxVal]
	constrain = @(minVal, maxVal, val) max(min(maxVal, val), minVal)


	% Solve for m in the equation Y * m = C
	% Get the transpose of Y to get Yt * Y * m = Yt * C

	Yt = transpose(Y);
	YtY = Yt * Y;

	YtC = Yt * C;

	m = YtY \ YtC;

	% Solve for estimated percentage increases, G, given test data T
	% D is the absolute percentage measured by G - 1

	G = T * m;
	D = G - 1;

	% Determine the percentage of the portfolio invested
	% (negative for selling short)

	J = constrain(-1, 1, 20 * D);

	% Determine the percentage gain in the portfolio on each day
	% Given the actual percentage gain A

	A = J .* (Q - 1);

	% Calculate the overall percentage change in the portfolio
	% Multiply all percentage increases and decreases in A

	A = A + 1
	P = prod(A);

	disp(P)
	save(strcat("../models/", ticker, "_predict.mat"))
end