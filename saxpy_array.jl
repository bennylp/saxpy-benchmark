function saxpy(N)
    println("N: $N")
    
    const XVAL = Float32(10 * rand())
    const YVAL = Float32(10 * rand())
    const AVAL = Float32(10 * rand())
    
    x = zeros(Float32,N) + XVAL
    y = zeros(Float32,N) + YVAL

    time = @elapsed y += AVAL * x
    
    time = time * 1000
    println("Elapsed: $time ms")
    
    const TRUEVAL = YVAL + AVAL * XVAL
    err = sum(y - TRUEVAL)
    println("Error: $err")
    
end

saxpy(1 << 26)

