//
//  TestClass.swift
//  IntegralImage
//
//  Created by Christopher Helf on 29.06.16.
//  Copyright © 2016 Christopher Helf. All rights reserved.
//

import Foundation
import Metal
import MetalPerformanceShaders

class TestClass {
    
    var device: MTLDevice! = nil
    var library : MTLLibrary! = nil
    var commandQueue : MTLCommandQueue! = nil

    init() {
        device = MTLCreateSystemDefaultDevice()!;
        library = device.newDefaultLibrary()!;
        commandQueue = device.makeCommandQueue()
    }
    
    func testSmallTextureSum() -> Bool {
        let n = 32
        let width = n
        let height = n
        let (ii, input, output) = createTestSetup(width, height)
        let sum = TestClass.getBufferForFloat(device: device)
        
        let commandBuffer = commandQueue.makeCommandBuffer()
        ii.encodeToCommandBuffer(commandBuffer, sourceTexture: input, destinationTexture: output)
        ii.getBoxIntegral(commandBuffer, integralImage: output, row: 0, col: 0, rows: n, cols: n, output: sum)
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        let vals = TestClass.textureToArray(texture: output)
        return (vals[n*n-1]==Float(n*n)) && (TestClass.floatBufferToFloat(sum) == Float(n*n))
    }
    
    func testMPS() -> Bool {
        print("testMPS")
        
        let mps = MPSImageIntegral(device: device)
        mps.offset = MPSOffset(x: 0, y: 0, z: 0)
        let n = 1000
        let (_, input, output) = createTestSetup(1280, 720)
        
        var elapsedGPU : UInt64 = 0
        for _ in 0..<n {
            let commandBuffer = commandQueue.makeCommandBuffer()
            mps.encode(commandBuffer: commandBuffer, sourceTexture: input, destinationTexture: output)
            let _t1 = mach_absolute_time()
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
            let _t2 = mach_absolute_time()
            elapsedGPU += _t2-_t1
        }
        var timeBaseInfo = mach_timebase_info_data_t()
        mach_timebase_info(&timeBaseInfo)
        
        let elapsedNanoGPU = elapsedGPU * UInt64(timeBaseInfo.numer) / UInt64(timeBaseInfo.denom);
        let nanoSecondsGPU = Float(elapsedNanoGPU)/Float(n)
        print("Nano Seconds MPS (GPU): \(nanoSecondsGPU)")
        let milliSecondsGPU = nanoSecondsGPU*Float(1e-6)
        print("Milli Seconds MPS (GPU): \(milliSecondsGPU)")
        print("Theoretical FPS MPS: \(1/(milliSecondsGPU/1000))·")
        return true

    }
    
    func compareImplAgainstMPSWithBounds() -> Bool {
        
        let width = 1280
        let height = 720
        
        let mps = MPSImageIntegral(device: device)
        mps.offset = MPSOffset(x: 0, y: 0, z: 0)
        let input = TestClass.createRandomTexture(device: device, width: width, height: height)
        let (ii, _, output1) = createTestSetup(width, height)
        let (_, _, output2) = createTestSetup(width, height)
        
        let commandBuffer = commandQueue.makeCommandBuffer()
        mps.encode(commandBuffer: commandBuffer, sourceTexture: input, destinationTexture: output1)
        ii.encodeToCommandBuffer(commandBuffer, sourceTexture: input, destinationTexture: output2)
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        
        let vals1 = TestClass.textureToArray(texture: output1)
        let vals2 = TestClass.textureToArray(texture: output2)
        
        for i in 0..<vals1.count {
            assertWithinBounds(val1: vals1[i], val2: vals2[i], bound: 5.0)
        }
        
        return true;
        
    }
    
    func compareImplAgainstMPS() -> Bool {
        
        let width = 1280
        let height = 720
        
        let mps = MPSImageIntegral(device: device)
        mps.offset = MPSOffset(x: 0, y: 0, z: 0)
        let input = TestClass.createTestTexture(device: device, val: 1.0, width: width, height: height)
        let (ii, _, output1) = createTestSetup(width, height)
        let (_, _, output2) = createTestSetup(width, height)
        
        let commandBuffer = commandQueue.makeCommandBuffer()
        mps.encode(commandBuffer: commandBuffer, sourceTexture: input, destinationTexture: output1)
        ii.encodeToCommandBuffer(commandBuffer, sourceTexture: input, destinationTexture: output2)
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        
        let vals1 = TestClass.textureToArray(texture: output1)
        let vals2 = TestClass.textureToArray(texture: output2)
        
        for i in 0..<vals1.count {
            assertWithinBounds(val1: vals1[i], val2: vals2[i], bound: 0.0001)
        }
        
        return true;
        
    }
    
    func assertWithinBounds(val1: Float, val2: Float, bound: Float) {
        assert(val1 >= val2 - bound && val1 <= val2 + bound)
    }
    
    
    func testTimes720p() -> Bool {
        print("testTimes720p")
        
        let n = 1000
        let (ii, input, output) = createTestSetup(1280, 720)
        
        var elapsedGPU : UInt64 = 0
        for _ in 0..<n {
            let commandBuffer = commandQueue.makeCommandBuffer()
            ii.encodeToCommandBuffer(commandBuffer, sourceTexture: input, destinationTexture: output)
            let _t1 = mach_absolute_time()
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
            let _t2 = mach_absolute_time()
            elapsedGPU += _t2-_t1
        }
        var timeBaseInfo = mach_timebase_info_data_t()
        mach_timebase_info(&timeBaseInfo)
        
        let elapsedNanoGPU = elapsedGPU * UInt64(timeBaseInfo.numer) / UInt64(timeBaseInfo.denom);
        let nanoSecondsGPU = Float(elapsedNanoGPU)/Float(n)
        print("Nano Seconds 720p (GPU): \(nanoSecondsGPU)")
        let milliSecondsGPU = nanoSecondsGPU*Float(1e-6)
        print("Milli Seconds 720p (GPU): \(milliSecondsGPU)")
        print("Theoretical FPS 720p: \(1/(milliSecondsGPU/1000))·")
        return true
    }
    
    func testTimes1080p() -> Bool {
        print("testTimes1080p")
        
        let n = 1000
        let (ii, input, output) = createTestSetup(1920, 1080)
        
        var elapsedGPU : UInt64 = 0
        for _ in 0..<n {
            let commandBuffer = commandQueue.makeCommandBuffer()
            ii.encodeToCommandBuffer(commandBuffer, sourceTexture: input, destinationTexture: output)
            let _t1 = mach_absolute_time()
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
            let _t2 = mach_absolute_time()
            elapsedGPU += _t2-_t1
        }
        var timeBaseInfo = mach_timebase_info_data_t()
        mach_timebase_info(&timeBaseInfo)
        
        let elapsedNanoGPU = elapsedGPU * UInt64(timeBaseInfo.numer) / UInt64(timeBaseInfo.denom);
        let nanoSecondsGPU = Float(elapsedNanoGPU)/Float(n)
        print("Nano Seconds 1080p (GPU): \(nanoSecondsGPU)")
        let milliSecondsGPU = nanoSecondsGPU*Float(1e-6)
        print("Milli Seconds 1080p (GPU): \(milliSecondsGPU)")
        print("Theoretical FPS 1080p: \(1/(milliSecondsGPU/1000))·")
        return true
    }
    
    private func createTestSetup(_ width: Int, _ height: Int, val: Float = 1.0, inclusive: Bool = true) -> (IntegralImage, MTLTexture, MTLTexture) {
        let ii = IntegralImage(device: device, library: library, width: width, height: height, inclusive: true)
        let input = TestClass.createTestTexture(device: device, val: val, width: width, height: height, useIncrease: false)
        let output = TestClass.createTestTexture(device: device, val: 0.0, width: width, height: height)
        return (ii, input, output)
    }
    
    class func createRandomTexture(device: MTLDevice, width: Int, height: Int) -> MTLTexture {
        var test = [Float](repeatElement(0.0, count: width*height))
        for i in 0..<width*height {
            test[i] = Float(Float(arc4random()) / Float(UINT32_MAX))
        }
        return TestClass.createTexture(device: device, format: .r32Float, width: width, height: height, bytes: test)
    }
    
    class func printTexture(texture: MTLTexture, displayBlockSize: Bool = true, blockSize: Int = 64) {
        let bytesPerRow = texture.width*MemoryLayout<Float>.size
        let region = MTLRegionMake2D(0, 0, texture.width, texture.height)
        var vals = [Float](repeatElement(0.0, count: texture.width*texture.height))
        texture.getBytes(&vals, bytesPerRow: bytesPerRow, from: region, mipmapLevel: 0)
        for y in 0..<texture.height {
            var rowStr = "";
            var blockCnt = 0;
            for x in 0..<texture.width {
                rowStr += "\(vals[y*texture.width+x])"
                
                if x < texture.width-1 {
                    rowStr += ","
                }
                if (((x+1)%blockSize==0 && x>0) || x == texture.width-1) && displayBlockSize {
                    rowStr += "||| (row: \(y), block: \(blockCnt))\n"
                    blockCnt+=1;
                }
            }
            print(rowStr)
        }
    }
    
    class func createTestTexture(device: MTLDevice, val: Float, width: Int, height: Int, useIncrease: Bool = false) -> MTLTexture {
        
        var test = [Float](repeatElement(val, count: width*height))
        if useIncrease {
            var cnt = 1;
            for i in 0..<test.count {
                if(i%width==0) {
                    cnt = 1
                };
                test[i]=Float(cnt)*val
                cnt+=1;
            }
        }
        
        return createTexture(device: device, format: .r32Float, width: width, height: height, bytes: test)
    }
    
    class func createTexture(device: MTLDevice, format: MTLPixelFormat, width: Int, height: Int, bytes: [Float]) -> MTLTexture {
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: format, width: width, height: height, mipmapped: false)
        descriptor.resourceOptions = MTLResourceOptions.storageModeShared
        descriptor.storageMode = MTLStorageMode.shared
        descriptor.usage = [MTLTextureUsage.renderTarget, MTLTextureUsage.shaderRead, MTLTextureUsage.shaderWrite]
        let t = device.makeTexture(descriptor: descriptor)
        t.replace(region: MTLRegionMake2D(0, 0, width, height), mipmapLevel: 0, withBytes: bytes, bytesPerRow: width*4)
        return t
    }
    
    class func textureToArray(texture: MTLTexture) -> [Float] {
        let bytesPerRow = texture.width*MemoryLayout<Float>.size
        let region = MTLRegionMake2D(0, 0, texture.width, texture.height)
        var vals = [Float](repeatElement(0.0, count: texture.width*texture.height))
        texture.getBytes(&vals, bytesPerRow: bytesPerRow, from: region, mipmapLevel: 0)
        return vals;
    }
    
    class func getBufferForFloat(device: MTLDevice) -> MTLBuffer {
        return device.makeBuffer(length: MemoryLayout<Float>.size, options: MTLResourceOptions.storageModeShared)
    }
    
    class func floatBufferToFloat(_ buffer: MTLBuffer) -> Float {
        let data = NSData(bytesNoCopy: buffer.contents(),
                          length: MemoryLayout<Float>.size, freeWhenDone: false)
        var rtn : Float = -1.0
        data.getBytes(&rtn, length:MemoryLayout<Float>.size)
        return rtn
    }
    
    
}
