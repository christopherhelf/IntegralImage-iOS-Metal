# Integral-Image in Metal for iOS

This repository contains an iOS project written in Swift 3.0 where I've tried to recreate a [parallel prefix sum algorithm](http://http.developer.nvidia.com/GPUGems3/gpugems3_ch39.html) written in Metal for educational purposes and then compare against validity and performance of the [MPSImageIntegral](https://developer.apple.com/reference/metalperformanceshaders/mpsimageintegral) class provided by Apple. Given a first attempt it's not even that bad (around ~70 FPS on a 720p image with my implementation as compared to ~260 FPS with MPSImageIntegral on a iPhone 6S). I've tried to optimize if by fixing the algorithm to a specific block size, unrolling all loops and using `float4` values in each thread in order to save global memory bandwith. 

The `TestClass`file gives an idea on how to use the `IntegralImage` class. It's basically instantiating the class via

    IntegralImage(device: MTLDevice, library: MTLLibrary, width: Int, height: Int, inclusive: Bool)
    
where `inclusive` indicates whether the computed sums at `(x,y)` should include `I(x,y)`. 

Feel free to experiment and check out the source. The repository is not maintained and I don't guarantee validity (even though all tests are passing just fine).