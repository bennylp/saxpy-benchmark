library(data.table)

N <- 2 ^ 26
cat("N:", N, "\n")

# Random numbers
XVAL <- 10 * runif(1)
YVAL <- 10 * runif(1)
AVAL <- 10 * runif(1)

m <- as.data.table(matrix(0, ncol = 2, nrow = N))
m[,1] = YVAL
m[,2] = XVAL


t0 <- Sys.time()
m[,1] = m[,1] + AVAL * m[,2]
diff <- Sys.time() - t0

cat("Elapsed:", diff*1000, " ms\n")

answer <- YVAL + AVAL * XVAL
error <- sum(abs(m[,1] - answer))
cat("Error:", error, "\n")
