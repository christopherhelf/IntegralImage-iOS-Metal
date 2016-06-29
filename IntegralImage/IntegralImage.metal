//
//  IntegralImage.metal
//  IntegralImage
//
//  Created by Christopher Helf on 28.06.16.
//  Copyright © 2016 Christopher Helf. All rights reserved.
//

#include <metal_stdlib>
#include <metal_compute>
using namespace metal;

// Our scan kernel function
kernel void ii_scan(
                                     /* Our grayscale input texture */
                                     texture2d<float, access::read> input [[texture(0)]],
                                     /* The auxiliary texture holding the sums per threadgroup,
                                      specified as array here in order to make it optional */
                                     texture2d<float, access::write> aux [[texture(1)]],
                                     /* The output texture containing summed up values per block/row */
                                     texture2d<float, access::write> output [[texture(2)]],
                                     /* Our temporary threadgroup memory */
                                     threadgroup float4 *temp [[ threadgroup(0) ]],
                                     /* Whether we want an inclusive scan */
                                     constant bool &inclusive [[buffer(0)]],
                                     /* The current threads block id */
                                     ushort2 blockIdx [[threadgroup_position_in_grid]],
                                     /* The current threads position within a block */
                                     ushort2 threadIdx [[ thread_position_in_threadgroup ]]
                                     )
{
    // Another threadgroup memory in order to store sums of blocks (for inclusion)
    threadgroup float4 sums;
    
    // We will work with a fixed blocksize so we can unroll
    // the loops in reduction and down-sweep
    int BLOCKSIZE = 64;
    
    // Get current positions in the respective row
    int tdx = threadIdx.x;
    int bdx = blockIdx.x;
    int bdy = blockIdx.y;
    
    // Whether we have an auxiliary array
    bool hasAux = aux.get_width() > 1 || aux.get_height() > 1;
    
    // Get the actual width of the texuture
    uint width = uint(input.get_width());
    uint auxWidth = 0;
    
    // If we have an auxiliary array, get its width
    if(hasAux) {
        auxWidth = uint(aux.get_width());
    }
    
    // Calculate the block offsets, as well will dispatch 1/4 of the blocks only
    // in horizontal direction and process four values at the same time
    int blockOffset1 = (bdx*4)*BLOCKSIZE;
    int blockOffset2 = blockOffset1+BLOCKSIZE;
    int blockOffset3 = blockOffset2+BLOCKSIZE;
    int blockOffset4 = blockOffset3+BLOCKSIZE;
    
    // Calculate the actual position from where to read values
    uint2 pos1 = uint2(blockOffset1+tdx,bdy);
    uint2 pos2 = uint2(blockOffset2+tdx,bdy);
    uint2 pos3 = uint2(blockOffset3+tdx,bdy);
    uint2 pos4 = uint2(blockOffset4+tdx,bdy);
    
    // Load four values into shared memory, check
    // however that we are still within the texture
    // otherwise pad with zeros
    temp[tdx] = float4(
                       pos1.x < width ? input.read(pos1).r : 0.0,
                       pos2.x < width ? input.read(pos2).r : 0.0,
                       pos3.x < width ? input.read(pos3).r : 0.0,
                       pos4.x < width ? input.read(pos4).r : 0.0
                       );
    
    
    // Wait until all threads have read into shared memory
    threadgroup_barrier( mem_flags::mem_threadgroup );
    
    // Reduction - unrolled loop 32, 16, 8, 4, 2, 1
    int offset = 1;
    int d, ai, bi;
    
    d = 32;
    threadgroup_barrier( mem_flags::mem_threadgroup );
    if (tdx < d) {
        ai = offset*(2*tdx+1)-1;
        bi = offset*(2*tdx+2)-1;
        temp[bi] += temp[ai];
    }
    offset *= 2;
    
    d = 16;
    threadgroup_barrier( mem_flags::mem_threadgroup );
    if (tdx < d) {
        ai = offset*(2*tdx+1)-1;
        bi = offset*(2*tdx+2)-1;
        temp[bi] += temp[ai];
    }
    offset *= 2;
    
    d = 8;
    threadgroup_barrier( mem_flags::mem_threadgroup );
    if (tdx < d) {
        ai = offset*(2*tdx+1)-1;
        bi = offset*(2*tdx+2)-1;
        temp[bi] += temp[ai];
    }
    offset *= 2;
    
    d = 4;
    threadgroup_barrier( mem_flags::mem_threadgroup );
    if (tdx < d) {
        ai = offset*(2*tdx+1)-1;
        bi = offset*(2*tdx+2)-1;
        temp[bi] += temp[ai];
    }
    offset *= 2;
    
    d = 2;
    threadgroup_barrier( mem_flags::mem_threadgroup );
    if (tdx < d) {
        ai = offset*(2*tdx+1)-1;
        bi = offset*(2*tdx+2)-1;
        temp[bi] += temp[ai];
    }
    offset *= 2;
    
    d = 1;
    threadgroup_barrier( mem_flags::mem_threadgroup );
    if (tdx < d) {
        ai = offset*(2*tdx+1)-1;
        bi = offset*(2*tdx+2)-1;
        temp[bi] += temp[ai];
    }
    offset *= 2;
        
    // We use four threads to store auxiliary sums (TODO)
    if (tdx==0) {
        
        // Store the sum
        sums = temp[BLOCKSIZE-1];
        
        // Store auxiliary sums if necessary
        if(hasAux) {
            
            // Get the positions
            uint2 auxPos1 = uint2(bdx*4, bdy);
            uint2 auxPos2 = uint2(bdx*4+1, bdy);
            uint2 auxPos3 = uint2(bdx*4+2, bdy);
            uint2 auxPos4 = uint2(bdx*4+3, bdy);
            
            // Write them if we are not exceeding
            if (auxPos1[0] < auxWidth) aux.write(sums[0],auxPos1);
            if (auxPos2[0] < auxWidth) aux.write(sums[1],auxPos2);
            if (auxPos3[0] < auxWidth) aux.write(sums[2],auxPos3);
            if (auxPos4[0] < auxWidth) aux.write(sums[3],auxPos4);
            
        }
        
        // Reset last element
        // Only the first thread clears the memory
        temp[BLOCKSIZE-1] = float4(0.0);// clear the last element
        
    }
    
    // Down-sweep phase
    // traverse down tree & build scan, unrolled loop 1,2,4,8,16,32
    float4 t;
    d=1;
    offset >>= 1;
    threadgroup_barrier( mem_flags::mem_threadgroup );
    if(tdx < d) {
        ai = offset*(2*tdx+1)-1;
        bi = offset*(2*tdx+2)-1;
        t = temp[ai];
        temp[ai] = temp[bi];
        temp[bi] += t;
    }
    
    d=2;
    offset >>= 1;
    threadgroup_barrier( mem_flags::mem_threadgroup );
    if(tdx < d) {
        ai = offset*(2*tdx+1)-1;
        bi = offset*(2*tdx+2)-1;
        t = temp[ai];
        temp[ai] = temp[bi];
        temp[bi] += t;
    }
    
    d=4;
    offset >>= 1;
    threadgroup_barrier( mem_flags::mem_threadgroup );
    if(tdx < d) {
        ai = offset*(2*tdx+1)-1;
        bi = offset*(2*tdx+2)-1;
        t = temp[ai];
        temp[ai] = temp[bi];
        temp[bi] += t;
    }
    
    d=8;
    offset >>= 1;
    threadgroup_barrier( mem_flags::mem_threadgroup );
    if(tdx < d) {
        ai = offset*(2*tdx+1)-1;
        bi = offset*(2*tdx+2)-1;
        t = temp[ai];
        temp[ai] = temp[bi];
        temp[bi] += t;
    }
    
    d=16;
    offset >>= 1;
    threadgroup_barrier( mem_flags::mem_threadgroup );
    if(tdx < d) {
        ai = offset*(2*tdx+1)-1;
        bi = offset*(2*tdx+2)-1;
        t = temp[ai];
        temp[ai] = temp[bi];
        temp[bi] += t;
    }
    
    d=32;
    offset >>= 1;
    threadgroup_barrier( mem_flags::mem_threadgroup );
    if(tdx < d) {
        ai = offset*(2*tdx+1)-1;
        bi = offset*(2*tdx+2)-1;
        t = temp[ai];
        temp[ai] = temp[bi];
        temp[bi] += t;
    }
    
    threadgroup_barrier( mem_flags::mem_threadgroup );
    
    // Shift array one to the left, and store sums in the last column for inclusive array
    if (inclusive && hasAux) {
        if (tdx < BLOCKSIZE -1) {
            if (pos1.x < width) output.write(temp[tdx+1][0], pos1);
            if (pos2.x < width) output.write(temp[tdx+1][1], pos2);
            if (pos3.x < width) output.write(temp[tdx+1][2], pos3);
            if (pos4.x < width) output.write(temp[tdx+1][3], pos4);
        } else {
            if (pos1.x < width) output.write(sums[0], pos1);
            if (pos2.x < width) output.write(sums[1], pos2);
            if (pos3.x < width) output.write(sums[2], pos3);
            if (pos4.x < width) output.write(sums[3], pos4);
        }
        
    } else {
        // write results to device memory if we are not exceeding
        if (pos1.x < width) output.write(temp[tdx][0], pos1);
        if (pos2.x < width) output.write(temp[tdx][1], pos2);
        if (pos3.x < width) output.write(temp[tdx][2], pos3);
        if (pos4.x < width) output.write(temp[tdx][3], pos4);
    }
    
    
}

// Simple kernel function to add sums from the auxiliary array
kernel void ii_fixup(
                                  /* the input texture from the scan */
                                  texture2d<float, access::read> input [[texture(0)]],
                                  /* the auxiliary texture containing the cumulated sums */
                                  texture2d<float, access::read> aux [[texture(1)]],
                                  /* the output texture */
                                  texture2d<float, access::write> output [[texture(2)]],
                                  /* The current threads block id */
                                  uint2 blockIdx [[threadgroup_position_in_grid]],
                                  /* the current threads global positon in the texture */
                                  uint2 globalIdx [[thread_position_in_grid]]
                                  )
{
    // Get the width
    uint width = output.get_width();
    
    // Get the input value
    float val = input.read(globalIdx).r;
    
    // Sum up the values
    if(globalIdx.x<width) output.write(val+aux.read(blockIdx).r,globalIdx);
}

// Simple function, which can be used in any subsequent encoders after the integral image has been
// calculated
float ii_boxintegral_f(texture2d<float, access::read> integralImage, int row, int col, int rows, int cols);
float ii_boxintegral_f(texture2d<float, access::read> integralImage, int row, int col, int rows, int cols) {
    int width = integralImage.get_width();
    int height = integralImage.get_height();
    int r1 = min(row-1, height-1);
    int c1 = min(col-1, width-1);
    int r2 = min(row+rows, height-1);
    int c2 = min(col+cols, width-1);
    float A(0.0f), B(0.0f), C(0.0f), D(0.0f);
    if (r1 >= 0 && c1 >= 0) A = integralImage.read(uint2(c1,r1)).r;
    if (r1 >= 0 && c2 >= 0) B = integralImage.read(uint2(c2,r1)).r;
    if (r2 >= 0 && c1 >= 0) C = integralImage.read(uint2(c1,r2)).r;
    if (r2 >= 0 && c2 >= 0) D = integralImage.read(uint2(c2,r2)).r;
    return max(0.f, A-B-C+D);
}

// Simple kernel function to calculate the sum of pixels using the integral image
kernel void ii_boxintegral(
                           texture2d<float, access::read> integralImage [[texture(0)]],
                           constant int &row [[buffer(0)]],
                           constant int &col [[buffer(1)]],
                           constant int &rows [[buffer(2)]],
                           constant int &cols [[buffer(3)]],
                           device float &result [[buffer(4)]]
                           ) {
    result = ii_boxintegral_f(integralImage, row, col, rows, cols);
}


