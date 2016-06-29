//
//  IntegralImageTests.swift
//  IntegralImageTests
//
//  Created by Christopher Helf on 28.06.16.
//  Copyright © 2016 Christopher Helf. All rights reserved.
//

import XCTest
import Metal
import MetalPerformanceShaders

@testable import IntegralImage

class IntegralImageTests: XCTestCase {
    
    var device: MTLDevice! = nil
    var library : MTLLibrary! = nil
    var commandQueue : MTLCommandQueue! = nil
    
    override func setUp() {
        super.setUp()
        device = MTLCreateSystemDefaultDevice()!;
        library = device.newDefaultLibrary()!;
        commandQueue = device.newCommandQueue()
    }
    
    func testSmallTexture() {
        
        let width = 16
        let height = 16
        
        let ii = IntegralImage(device: device, library: library, width: width, height: height)
        let input = createTestTexture(device: device, val: 1.0, width: width, height: height)
        let output = createTestTexture(device: device, val: 0.0, width: width, height: height)
        
        let commandBuffer = commandQueue.commandBuffer()
        ii.encodeToCommandBuffer(commandBuffer, sourceTexture: input, destinationTexture: output)
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        
        print(textureToArray(texture: output))
        
        XCTAssertTrue(1==1)
        
    }
    
    func testPerformanceExample() {
        // This is an example of a performance test case.
        
    }
    
    
    
}
