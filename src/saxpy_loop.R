N <- 2 ^ 26

# Random numbers
XVAL <- 10 * runif(1)
YVAL <- 10 * runif(1)
AVAL <- 10 * runif(1)

x <- array(XVAL, dim=c(N))
y <- array(YVAL, dim=c(N))
cat("N:", N, "\n")

t0 <- proc.time()
for(i in 1:N) {
  y[i] <- y[i] + x[i] * AVAL
}
t1 = proc.time()
diff <- t1 - t0
cat("Elapsed:", diff*1000, " ms\n")

answer <- YVAL + AVAL * XVAL
error <- sum(abs(y - answer))
cat("Error:", error, "\n")
