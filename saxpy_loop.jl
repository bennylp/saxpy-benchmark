function saxpy( a, x, y )
    @simd for i=1:length(x)
        @inbounds y[i] += a*x[i]
    end
end

function flog( N )
    println("N: $N")
    
    const XVAL = Float32(10 * rand())
    const YVAL = Float32(10 * rand())
    const AVAL = Float32(10 * rand())
    
    x = zeros(Float32,N) + XVAL
    y = zeros(Float32,N) + YVAL    
    
    time = @elapsed saxpy(AVAL,x,y)
    
    time = time * 1000
    println("Elapsed: $time ms")
    
    err = 0.0
    for i in 1:N
        err += abs( y[i] - (YVAL + AVAL * x[i]) )
    end
    println("Error: $err")
end

flog(1 << 26)
