// RUN: %clang_cc1 %s -triple spir -O0 -emit-llvm-bc -o %t.bc
// RUN: llvm-spirv %t.bc -o %t.spv
// RUN: llvm-spirv %t.spv -to-text -o - | FileCheck %s
// RUN: spirv-val %t.spv

// CHECK: TypeImage [[TypeImage:[0-9]+]]
// CHECK: TypeSampler [[TypeSampler:[0-9]+]]
// CHECK: TypePointer [[TypeImagePtr:[0-9]+]] {{[0-9]+}} [[TypeImage]]
// CHECK: TypePointer [[TypeSamplerPtr:[0-9]+]] {{[0-9]+}} [[TypeSampler]]

// CHECK: FunctionParameter [[TypeImage]] [[srcimg:[0-9]+]]
// CHECK: FunctionParameter [[TypeSampler]] [[sampler:[0-9]+]]

// CHECK: Variable [[TypeImagePtr]] [[srcimg_addr:[0-9]+]]
// CHECK: Variable [[TypeSamplerPtr]] [[sampler_addr:[0-9]+]]

// CHECK: Store [[srcimg_addr]] [[srcimg]]
// CHECK: Store [[sampler_addr]] [[sampler]]

// CHECK: Load {{[0-9]+}} [[srcimg_val:[0-9]+]] [[srcimg_addr]]
// CHECK: Load {{[0-9]+}} [[sampler_val:[0-9]+]] [[sampler_addr]]

// CHECK: SampledImage {{[0-9]+}} {{[0-9]+}} [[srcimg_val]] [[sampler_val]]
// CHECK-NEXT: ImageSampleExplicitLod

// CHECK: Load {{[0-9]+}} [[srcimg_val:[0-9]+]] [[srcimg_addr]]
// CHECK: ImageQuerySizeLod {{[0-9]+}} {{[0-9]+}} [[srcimg_val]]

// Excerpt from opencl-c-base.h
typedef float float4 __attribute__((ext_vector_type(4)));
typedef int int2 __attribute__((ext_vector_type(2)));
typedef __SIZE_TYPE__ size_t;

// Excerpt from opencl-c.h to speed up compilation.
#define __ovld __attribute__((overloadable))
#define __purefn __attribute__((pure))
#define __cnfn __attribute__((const))
size_t __ovld __cnfn get_global_id(unsigned int dimindx);
int __ovld __cnfn get_image_width(read_only image2d_t image);
float4 __purefn __ovld read_imagef(read_only image2d_t image, sampler_t sampler, int2 coord);


__kernel void test_fn(image2d_t srcimg, sampler_t sampler, global float4 *results) {
  int tid_x = get_global_id(0);
  int tid_y = get_global_id(1);
  results[tid_x + tid_y * get_image_width(srcimg)] = read_imagef(srcimg, sampler, (int2){tid_x, tid_y});
}
