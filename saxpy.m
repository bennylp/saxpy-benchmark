N = 2 ^ 26
XVAL = 2.0;
YVAL = 1.0;
AVAL = 3.0;

x = ones(N,1) * XVAL;
y = ones(N,1) * YVAL;

tic;
y += x * AVAL;
toc

answer = YVAL + AVAL * XVAL;
error = sum(abs(y - answer))
