//
//  IntegralImage.swift
//  IntegralImage
//
//  Created by Christopher Helf on 28.06.16.
//  Copyright © 2016 Christopher Helf. All rights reserved.
//

import Foundation
import Metal
import MetalPerformanceShaders

class IntegralImage : MPSKernel {
    
    // Some general variables needed
    private var width : Int
    private var height: Int
    private var library : MTLLibrary
    
    // The inclusive variable, automatically adjusts the respective buffer
    var inclusive : Bool! {
        didSet {
            self.inclusiveBuffer = device.newBuffer(withBytes: &inclusive, length: sizeof(Bool), options: [])
        }
    }
    var inclusiveBuffer: MTLBuffer! = nil
    
    // Fixed blocksize
    private let blockSize : Int = 64
    
    // The pipelines we use
    private var scanPipeline : MTLComputePipelineState! = nil
    private var fixupPipeline : MTLComputePipelineState! = nil
    private var boxintegralPipeline : MTLComputePipelineState! = nil
    private var transposePass : MPSImageTranspose! = nil
    
    // Our intermediary textures
    private var aux : MTLTexture! = nil
    private var fakeAux : MTLTexture! = nil
    private var auxScanned : MTLTexture! = nil
    private var intermediary : MTLTexture! = nil
    private var out : MTLTexture! = nil
    private var input_t : MTLTexture! = nil
    private var aux_t : MTLTexture! = nil
    private var auxScanned_t : MTLTexture! = nil
    private var intermediary_t : MTLTexture! = nil
    private var out_t : MTLTexture! = nil
    
    
    /**
     Constructor
     takes the MTLLibrary already as argument for the sake of simplicity, and
     width and height as dimensions of the be processed input texture as allocating
     intermediary textures is expensive, thus this should not be done too often
     however, upon encoding to the commandbuffer another check if performed and textures
     are reallocated with the correct sizes if necessary. the inclusive parameter specifies
     whether the resulting integral image should contain I(x,y) at (x,y)
     */
    init(device: MTLDevice, library: MTLLibrary, width: Int, height: Int, inclusive: Bool=false) {
        
        // Store width and height
        self.width = width
        self.height = height
        self.library = library
        
        super.init(device: device)
        
        // Set inclusion
        self.setInclusive(inclusive: inclusive)
        
        // Setup
        setupPipelines()
        setupTransposePass()
        createIntermediaryTextures()
    }
    
    // Convenience function
    func setInclusive(inclusive: Bool) {
        self.inclusive = inclusive
    }
    
    // Sets the pipelines for encoding
    func setupPipelines() {
        scanPipeline = self.getPipeline(kernel: "ii_scan")
        fixupPipeline = self.getPipeline(kernel: "ii_fixup")
        boxintegralPipeline = self.getPipeline(kernel: "ii_boxintegral")
    }
    
    // We use the MPSImageTranspose shader
    func setupTransposePass() {
        transposePass = MPSImageTranspose(device: device)
    }
    
    // Creates all necessary intermediary textures
    func createIntermediaryTextures() {
        
        let auxBlockSize = width % self.blockSize == 0 ? max(width/self.blockSize,1) : width/self.blockSize+1
        let auxTBlockSize = height % self.blockSize == 0 ? max(height/self.blockSize,1) : height/self.blockSize+1
        
        // There must be better way to do this without that many allocations?
        fakeAux = self.createIntermediaryTexture(format: .r32Float, width: 1, height: 1)
        aux = self.createIntermediaryTexture(format: .r32Float, width: auxBlockSize, height: height)
        auxScanned = self.createIntermediaryTexture(format: .r32Float, width: auxBlockSize, height: height)
        intermediary = self.createIntermediaryTexture(format: .r32Float, width: width, height: height)
        out = self.createIntermediaryTexture(format: .r32Float, width: width, height: height)
        input_t = self.createIntermediaryTexture(format: .r32Float, width: height, height: width)
        aux_t = self.createIntermediaryTexture(format: .r32Float, width: auxTBlockSize, height: width)
        auxScanned_t = self.createIntermediaryTexture(format: .r32Float, width: auxTBlockSize, height: width)
        intermediary_t = self.createIntermediaryTexture(format: .r32Float, width: height, height: width)
        out_t = self.createIntermediaryTexture(format: .r32Float, width: height, height: width)
    }
    
    // Encodes a scan pass, i.e. scan sums within blocks
    // if aux is nil, we use a dummy aux texture with width and height 1
    private func encodeScan(commandBuffer: MTLCommandBuffer, input: MTLTexture, aux: MTLTexture?, output: MTLTexture) {
        
        // Get the total number of blocks
        let totalBlocks = input.width % blockSize == 0 ? max(input.width/blockSize,1) : input.width/blockSize+1
        
        // Get the required number of blocks
        let requiredBlocks = totalBlocks % 4 == 0 ? totalBlocks/4 : totalBlocks/4+1
        
        // We are scanning per row only, i.e. multiple one-row threadgroups
        let scanGrid = MTLSizeMake(requiredBlocks, input.height, 1)
        let scanBlock = MTLSizeMake(blockSize, 1, 1)
        
        let enc = commandBuffer.computeCommandEncoder()
        enc.pushDebugGroup("integral_image_scan")
        enc.setComputePipelineState(scanPipeline)
        
        // Set the input
        enc.setTexture(input, at: 0)
        
        // Set the auxiliary texture (nil if we dont need it)
        if let _aux = aux {
            enc.setTexture(_aux, at: 1)
        } else {
            enc.setTexture(fakeAux, at: 1)
        }
        
        // Set the output
        enc.setTexture(output, at: 2)
        enc.setBuffer(self.inclusiveBuffer, offset: 0, at: 0)
        
        // Create the threadgroup memory (times 4 because we use float4s)
        enc.setThreadgroupMemoryLength(blockSize * sizeof(Float) * 4, at: 0)
        
        enc.dispatchThreadgroups(scanGrid, threadsPerThreadgroup: scanBlock)
        enc.popDebugGroup()
        enc.endEncoding()
    }
    
    // Encodes the pass that writes auxiliary sums to the rows
    private func encodeFixup(commandBuffer: MTLCommandBuffer, input: MTLTexture, aux: MTLTexture, output: MTLTexture) {
        let blocks = input.width % blockSize == 0 ? max(input.width/blockSize,1) : input.width/blockSize+1
        let scanBlock = MTLSizeMake(blockSize, 1, 1)
        let scanGrid = MTLSizeMake(blocks, input.height, 1)
        let enc = commandBuffer.computeCommandEncoder()
        enc.pushDebugGroup("integral_image_fixup")
        enc.setComputePipelineState(fixupPipeline)
        enc.setTexture(input, at: 0)
        enc.setTexture(aux, at: 1)
        enc.setTexture(output, at: 2)
        enc.dispatchThreadgroups(scanGrid, threadsPerThreadgroup: scanBlock)
        enc.popDebugGroup()
        enc.endEncoding()
    }
   
    
    // Encodes the calculation of the integral image of sourceTexture to the commandBuffer, resulting
    // in the integral image in destinationTexture
    func encodeToCommandBuffer(_ commandBuffer: MTLCommandBuffer,
                               sourceTexture: MTLTexture,
                               destinationTexture: MTLTexture) {
        
        // We need a grayscale texture in any case
        assert(sourceTexture.pixelFormat == MTLPixelFormat.r32Float, "Source texture must be a grayscale r32Float texture")
        
        // Check whether the dimensions are the same
        assert(sourceTexture.width == destinationTexture.width, "Source texture must be the same size as destination texture")
        
        // Check whether the dimensions are the same
        assert(sourceTexture.height == destinationTexture.height, "Source texture must be the same size as destination texture")
        
        // We need at least two rows, otherwise we could get mixed up with our dummy auxiliary array
        assert(sourceTexture.height > 1, "Source texture must be at least of height 2")
        
        // We use a fixed blockSize of 64 and only one auxiliary array which is scanned only once, 
        // hence this last scan pass of the aux array must fit into a single block
        assert(sourceTexture.height <= Int(pow(Float(blockSize), 2.0)) && sourceTexture.width <= Int(pow(Float(blockSize), 2.0)))
        
        // Check whether width and height are still valid and recreate intermediary textures if necessary
        if (sourceTexture.width != width || sourceTexture.height != height) {
            createIntermediaryTextures()
        }
        
        self.encodeScan(commandBuffer: commandBuffer, input: sourceTexture, aux: aux, output: intermediary)
        self.encodeScan(commandBuffer: commandBuffer, input: aux, aux: nil, output: auxScanned)
        self.encodeFixup(commandBuffer: commandBuffer, input: intermediary, aux: auxScanned, output: out)
        self.transposePass.encode(to: commandBuffer, sourceTexture: out, destinationTexture: input_t)
        self.encodeScan(commandBuffer: commandBuffer, input: input_t, aux: aux_t, output: intermediary_t)
        self.encodeScan(commandBuffer: commandBuffer, input: aux_t, aux: nil, output: auxScanned_t)
        self.encodeFixup(commandBuffer: commandBuffer, input: intermediary_t, aux: auxScanned_t, output: out_t)
        self.transposePass.encode(to: commandBuffer, sourceTexture: out_t, destinationTexture: destinationTexture)
    }
    
    // Convenience function for testing
    func getBoxIntegral(_ commandBuffer: MTLCommandBuffer, integralImage: MTLTexture, row: Int, col: Int, rows: Int, cols: Int, output: MTLBuffer) {
        let enc = commandBuffer.computeCommandEncoder()
        enc.setComputePipelineState(boxintegralPipeline)
        enc.setTexture(integralImage, at: 0)
        enc.setBuffer(getBufferFromInt(row), offset: 0, at: 0)
        enc.setBuffer(getBufferFromInt(col), offset: 0, at: 1)
        enc.setBuffer(getBufferFromInt(rows), offset: 0, at: 2)
        enc.setBuffer(getBufferFromInt(cols), offset: 0, at: 3)
        enc.setBuffer(output, offset: 0, at: 4)
        enc.dispatchThreadgroups(MTLSize(width: 1,height: 1,depth: 1), threadsPerThreadgroup: MTLSize(width: 1,height: 1,depth: 1))
        enc.endEncoding()
    }
    
    // Helper functions
    
    // Creates a MTLTexture from arguments
    private func createIntermediaryTexture(format: MTLPixelFormat, width: Int, height: Int) -> MTLTexture {
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(with: format, width: width, height: height, mipmapped: false)
        descriptor.resourceOptions = MTLResourceOptions.storageModeShared
        descriptor.storageMode = MTLStorageMode.shared
        return device.newTexture(with: descriptor)
    }
    
    // Creates a compute pipeline from a kernel name
    private func getPipeline(kernel: String) -> MTLComputePipelineState {
        let kernelFunction = library.newFunction(withName: kernel)
        do {
            let pipeline = try device.newComputePipelineState(with: kernelFunction!)
            return pipeline
        }
        catch {
            fatalError("MMPSIntegralImage failed for kernel: \(kernel)")
        }
    }
    
    // Returns a buffer from an integer
    private func getBufferFromInt(_ val: Int) -> MTLBuffer {
        var _v = val
        return device.newBuffer(withBytes: &_v, length: sizeof(Int), options: .storageModeShared)
    }
    
    
    
}
