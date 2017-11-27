saxpy <- function() {
    N <- 2 ^ 26
    cat("N:", N, "\n")
    
    # Random numbers
    XVAL <- 10 * runif(1)
    YVAL <- 10 * runif(1)
    AVAL <- 10 * runif(1)
    
    x <- array(XVAL, dim=c(N))
    y <- array(YVAL, dim=c(N))
    
    t0 <- Sys.time()
    for(i in 1:N) {
      y[i] <- y[i] + x[i] * AVAL
    }
    diff <- Sys.time() - t0
    cat("Elapsed:", diff*1000, " ms\n")
    
    answer <- YVAL + AVAL * XVAL
    error <- sum(abs(y - answer))
    cat("Error:", error, "\n")
}

require(compiler)
saxpy_compiled <- cmpfun(saxpy)
saxpy_compiled()
