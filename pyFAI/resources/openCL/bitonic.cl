/*############################################################################
# Sort elements within a vector by Matthew Scarpino,
# Taken from his book "OpenCL in Action",
# November 2011 ISBN 9781617290176
# Original license for the code: "public domain"
#
# Originally this code is public domain. The MIT license has been added
# by J. Kieffer (jerome.kieffer@esrf.eu) to provide a disclaimer.
# J. Kieffer does not claim authorship of this code developed by .
#
# Copyright (c) 2011 Matthew Scarpino
# Copyright (c) 2018 Jerome Kieffer, ESRF
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
#############################################################################*/

// Updated: J. Kieffer 15/02/2018

static float8 vector_swap(float4 input1,
                          float4 input2,
                          int dir, 
                          int4 comp)
{
    int4 add3 = (int4)(4, 5, 6, 7);
    float4 temp = input1;
    comp = ((input1 < input2) ^ dir) * 4 + add3;
    input1 = shuffle2(input1, input2, as_uint4(comp));
    input2 = shuffle2(input2, temp, as_uint4(comp));
    return (float8)(input1, input2);
}


static float4 vector_sort(float4 input, int dir)
{
    uint4 mask1 = (uint4)(1, 0, 3, 2);
    uint4 mask2 = (uint4)(2, 3, 0, 1);
    int4 add1 = (int4)(1, 1, 3, 3);
    int4 add2 = (int4)(2, 3, 2, 3);

    int4 comp = (input < shuffle(input, mask2)) ^ dir;
    float4 tmp = shuffle(input, as_uint4(comp * 2 + add2));
    comp = (tmp < shuffle(tmp, mask1)) ^ dir;
    return shuffle(tmp, as_uint4(comp + add1));
}


/*
 * Functions to be called from an actual kernel.
 * 
 * Workgroup colaborative function which sorts a vector of 8 element per thread
 * needs 4 element of shared memory per thread.
 */

static float8 local_sort(uint local_id, uint group_id, uint local_size,
                         float8 input, local float4 *l_data)
{
    float4 input1, input2, temp;
    float8 output;

    int dir;
    uint id, size, stride;
    int4 comp;

    uint4 mask1 = (uint4)(1, 0, 3, 2);
    uint4 mask2 = (uint4)(2, 3, 0, 1);
    uint4 mask3 = (uint4)(3, 2, 1, 0);

    int4 add1 = (int4)(1, 1, 3, 3);
    int4 add2 = (int4)(2, 3, 2, 3);
    int4 add3 = (int4)(1, 2, 2, 3);

    // retrieve input data
    input1 = input.lo;
    input2 = input.hi;

    // Find global address
    id = local_id * 2;

    /* Sort input 1 - ascending */
    comp = input1 < shuffle(input1, mask1);
    input1 = shuffle(input1, as_uint4(comp + add1));
    comp = input1 < shuffle(input1, mask2);
    input1 = shuffle(input1, as_uint4(comp * 2 + add2));
    comp = input1 < shuffle(input1, mask3);
    input1 = shuffle(input1, as_uint4(comp + add3));

    /* Sort input 2 - descending */
    comp = input2 > shuffle(input2, mask1);
    input2 = shuffle(input2, as_uint4(comp + add1));
    comp = input2 > shuffle(input2, mask2);
    input2 = shuffle(input2, as_uint4(comp * 2 + add2));
    comp = input2 > shuffle(input2, mask3);
    input2 = shuffle(input2, as_uint4(comp + add3));

    /* Swap corresponding elements of input 1 and 2 */
    add3 = (int4)(4, 5, 6, 7);
    dir = - (int) (local_id % 2);
    temp = input1;
    comp = ((input1 < input2) ^ dir) * 4 + add3;
    input1 = shuffle2(input1, input2, as_uint4(comp));
    input2 = shuffle2(input2, temp, as_uint4(comp));

    /* Sort data and store in local memory */
    l_data[id]   = vector_sort(input1, dir);
    l_data[id+1] = vector_sort(input2, dir);
    barrier(CLK_LOCAL_MEM_FENCE);
    
    /* Create bitonic set */
    for(size = 2; size < local_size; size <<= 1)
    {
      dir = - (int) (local_id/size & 1) ;

      for(stride = size; stride > 1; stride >>= 1)
      {
         barrier(CLK_LOCAL_MEM_FENCE);
         id = local_id + (local_id/stride)*stride;
         output = vector_swap(l_data[id], l_data[id + stride], dir, comp);
         l_data[id] = output.lo;
         l_data[id + stride] = output.hi;
      }

      barrier(CLK_LOCAL_MEM_FENCE);
      id = local_id * 2;
      input1 = l_data[id];
      input2 = l_data[id+1];
      temp = input1;
      comp = ((input1 < input2) ^ dir) * 4 + add3;
      input1 = shuffle2(input1, input2, as_uint4(comp));
      input2 = shuffle2(input2, temp, as_uint4(comp));
      l_data[id] = vector_sort(input1, dir);
      l_data[id+1] = vector_sort(input2, dir);
      barrier(CLK_LOCAL_MEM_FENCE);
    }

    /* Perform bitonic merge */
    dir = - (int) (group_id % 2);
    for(stride = local_size; stride > 1; stride >>= 1) 
    {
      barrier(CLK_LOCAL_MEM_FENCE);
      id = local_id + (local_id/stride)*stride;
      output = vector_swap(l_data[id], l_data[id + stride], dir, comp);
      l_data[id] = output.lo;
      l_data[id + stride]  = output.hi;

    }
    barrier(CLK_LOCAL_MEM_FENCE);

    /* Perform final sort */
    id = local_id * 2;
    input1 = l_data[id]; 
    input2 = l_data[id+1];
    temp = input1;
    comp = ((input1 < input2) ^ dir) * 4 + add3;
    input1 = shuffle2(input1, input2, as_uint4(comp));
    input2 = shuffle2(input2, temp, as_uint4(comp));
    input1 = vector_sort(input1, dir);
    input2 = vector_sort(input2, dir);

    // setup output and return it
    output = (float8)(input1, input2);
    return  output;
}
//////////////
// Kernels
//////////////

/*
 * bsort_all: Perform a sort within a workgroup. 
 * 
 * 1D kernel. each workgroup sorts 8x its size
 * dim0: wg=number_of_element/8
 */ 


kernel void bsort_all(global float4 *g_data,
                      local float4 *l_data)
{
    float4 input1, input2;
    float8 input, output;
    uint id, global_start;
    // Find global address
    id = get_local_id(0) * 2;
    global_start = get_group_id(0) * get_local_size(0) * 2 + id;

    input1 = g_data[global_start];
    input2 = g_data[global_start+1];
    input = (float8)(input1, input2);
    output = local_sort(get_local_id(0), get_group_id(0), get_local_size(0),
                          input, l_data);
    input1 = (float4) (output.s0, output.s1, output.s2, output.s3);
    input2 = (float4) (output.s4, output.s5, output.s6, output.s7);
    g_data[global_start] = input1;
    g_data[global_start+1] = input2;
}

/* 
 * Kernel: bsort_horizontal
 * 
 * Perform the sort along the horizontal axis of a 2D image
 * 
 * dim0 = y: wg=1
 * dim1 = x: wg=number_of_element/8
 */
kernel void bsort_horizontal(global float *g_data,
                             local float4 *l_data) 
{
    float8 input, output;
    uint id, global_start, offset;

    // Find global address
    offset = get_global_size(1)*get_global_id(0)*8;
    id = get_local_id(1) * 8;
    global_start = offset + get_group_id(1) * get_local_size(1) * 8 + id;

    input = (float8)(g_data[global_start    ],
                     g_data[global_start + 1],
                     g_data[global_start + 2],
                     g_data[global_start + 3],
                     g_data[global_start + 4],
                     g_data[global_start + 5],
                     g_data[global_start + 6],
                     g_data[global_start + 7]);

    output = local_sort(get_local_id(1), get_group_id(1), get_local_size(1),
                        input, l_data);

    g_data[global_start    ] = output.s0;
    g_data[global_start + 1] = output.s1;
    g_data[global_start + 2] = output.s2;
    g_data[global_start + 3] = output.s3;
    g_data[global_start + 4] = output.s4;
    g_data[global_start + 5] = output.s5;
    g_data[global_start + 6] = output.s6;
    g_data[global_start + 7] = output.s7;
}


/* 
 * Kernel: bsort_vertical
 * 
 * Perform the sort along the vertical axis of a 2D image
 * 
 * dim0 = y: wg=number_of_element/8
 * dim1 = x: wg=1
 * 
 * check if transposing +bsort_horizontal is not more efficient ?
 */

kernel void bsort_vertical(global float *g_data,
                           local float4 *l_data) 
{
    // we need to read 8 float position along the vertical axis
    float8 input, output;
    uint id, global_start, padding;

    // Find global address
    padding = get_global_size(1);
    id = get_local_id(0) * 8 * padding + get_global_id(1);
    global_start = get_group_id(0) * get_local_size(0) * 8 * padding + id;

    input = (float8)(g_data[global_start            ],
                     g_data[global_start + padding  ],
                     g_data[global_start + 2*padding],
                     g_data[global_start + 3*padding],
                     g_data[global_start + 4*padding],
                     g_data[global_start + 5*padding],
                     g_data[global_start + 6*padding],
                     g_data[global_start + 7*padding]);

      output = local_sort(get_local_id(0), get_group_id(0), get_local_size(0),
                          input, l_data);
      g_data[global_start             ] = output.s0;
      g_data[global_start + padding   ] = output.s1;
      g_data[global_start + 2*padding ] = output.s2;
      g_data[global_start + 3*padding ] = output.s3;
      g_data[global_start + 4*padding ] = output.s4;
      g_data[global_start + 5*padding ] = output.s5;
      g_data[global_start + 6*padding ] = output.s6;
      g_data[global_start + 7*padding ] = output.s7;
}

