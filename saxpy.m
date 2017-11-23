N = 2 ^ 26
XVAL = 2.5;
YVAL = 1.3;
AVAL = 3.7;

x = ones(N,1) * XVAL;
y = ones(N,1) * YVAL;

tic;
y += x * AVAL;
elapsed = toc;
elapsed = elapsed * 1000

answer = YVAL + AVAL * XVAL;
error = sum(abs(y - answer))
