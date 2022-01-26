// RUN: %clang_cc1 -triple spir -cl-std=CL2.0 %s -fdeclare-opencl-builtins -finclude-default-header -emit-llvm-bc -o %t.bc
// RUN: llvm-spirv %t.bc -spirv-text -o %t.txt
// RUN: FileCheck < %t.txt %s --check-prefix=CHECK-SPIRV
// RUN: llvm-spirv %t.bc -o %t.spv
// RUN: spirv-val %t.spv
// RUN: llvm-spirv -r %t.spv -o %t.rev.bc
// RUN: llvm-dis < %t.rev.bc | FileCheck %s --check-prefix=CHECK-LLVM
// RUN: llvm-spirv -r --spirv-target-env=SPV-IR %t.spv -o %t.rev.bc
// RUN: llvm-dis < %t.rev.bc | FileCheck %s --check-prefix=CHECK-SPV-IR
// RUN: llvm-spirv %t.rev.bc -spirv-text -o %t.rev.spt
// RUN: FileCheck < %t.rev.spt %s --check-prefix=CHECK-SPIRV

constant sampler_t constSampl = CLK_FILTER_LINEAR;

__kernel
void sample_kernel_float(image2d_t input, float2 coords, global float4 *results, sampler_t argSampl) {
  *results = read_imagef(input, constSampl, coords);
  *results = read_imagef(input, argSampl, coords);
  *results = read_imagef(input, CLK_FILTER_NEAREST|CLK_ADDRESS_REPEAT, coords);
}

__kernel
void sample_kernel_int(image2d_t input, float2 coords, global int4 *results, sampler_t argSampl) {
  *results = read_imagei(input, constSampl, coords);
  *results = read_imagei(input, argSampl, coords);
  *results = read_imagei(input, CLK_FILTER_NEAREST|CLK_ADDRESS_REPEAT, coords);
}

// CHECK-SPIRV: Capability LiteralSampler
// CHECK-SPIRV: EntryPoint 6 [[sample_kernel_float:[0-9]+]] "sample_kernel_float"
// CHECK-SPIRV: EntryPoint 6 [[sample_kernel_int:[0-9]+]] "sample_kernel_int"

// CHECK-SPIRV: TypeSampler [[TypeSampler:[0-9]+]]
// CHECK-SPIRV: TypeSampledImage [[SampledImageTy:[0-9]+]]
// CHECK-SPIRV: ConstantSampler [[TypeSampler]] [[ConstSampler1:[0-9]+]] 0 0 1
// CHECK-SPIRV: ConstantSampler [[TypeSampler]] [[ConstSampler2:[0-9]+]] 3 0 0
// CHECK-SPIRV: ConstantSampler [[TypeSampler]] [[ConstSampler3:[0-9]+]] 0 0 1
// CHECK-SPIRV: ConstantSampler [[TypeSampler]] [[ConstSampler4:[0-9]+]] 3 0 0

// CHECK-SPIRV: Function {{.*}} [[sample_kernel_float]]
// CHECK-SPIRV: FunctionParameter {{.*}} [[InputImage:[0-9]+]]
// CHECK-SPIRV: FunctionParameter [[TypeSampler]] [[argSampl:[0-9]+]]
// CHECK-LLVM: define spir_kernel void @sample_kernel_float(%opencl.image2d_ro_t addrspace(1)* %input, <2 x float> %coords, <4 x float> addrspace(1)* nocapture %results, %opencl.sampler_t addrspace(2)* %argSampl)

// CHECK-SPIRV: SampledImage [[SampledImageTy]] [[SampledImage1:[0-9]+]] [[InputImage]] [[ConstSampler1]]
// CHECK-SPIRV: ImageSampleExplicitLod {{.*}} [[SampledImage1]]
// CHECK-LLVM:  call spir_func <4 x float> @_Z11read_imagef14ocl_image2d_ro11ocl_samplerDv2_f(%opencl.image2d_ro_t addrspace(1)* %input, %opencl.sampler_t addrspace(2)* %0, <2 x float> %coords)
// CHECK-SPV-IR: call spir_func %spirv.SampledImage._void_1_0_0_0_0_0_0 addrspace(1)* @_Z20__spirv_SampledImagePU3AS133__spirv_Image__void_1_0_0_0_0_0_0PU3AS215__spirv_Sampler(%spirv.Image._void_1_0_0_0_0_0_0 addrspace(1)* %input, %spirv.Sampler addrspace(2)* %0)
// CHECK-SPV-IR: call spir_func <4 x float> @_Z38__spirv_ImageSampleExplicitLod_Rfloat4PU3AS140__spirv_SampledImage__void_1_0_0_0_0_0_0Dv2_fif(%spirv.SampledImage._void_1_0_0_0_0_0_0 addrspace(1)* %TempSampledImage, <2 x float> %coords, i32 2, float 0.000000e+00)

// CHECK-SPIRV: SampledImage [[SampledImageTy]] [[SampledImage2:[0-9]+]] [[InputImage]] [[argSampl]]
// CHECK-SPIRV: ImageSampleExplicitLod {{.*}} [[SampledImage2]]
// CHECK-LLVM:   call spir_func <4 x float> @_Z11read_imagef14ocl_image2d_ro11ocl_samplerDv2_f(%opencl.image2d_ro_t addrspace(1)* %input, %opencl.sampler_t addrspace(2)* %argSampl, <2 x float> %coords)
// CHECK-SPV-IR: call spir_func %spirv.SampledImage._void_1_0_0_0_0_0_0 addrspace(1)* @_Z20__spirv_SampledImagePU3AS133__spirv_Image__void_1_0_0_0_0_0_0PU3AS215__spirv_Sampler(%spirv.Image._void_1_0_0_0_0_0_0 addrspace(1)* %input, %spirv.Sampler addrspace(2)* %argSampl)
// CHECK-SPV-IR: call spir_func <4 x float> @_Z38__spirv_ImageSampleExplicitLod_Rfloat4PU3AS140__spirv_SampledImage__void_1_0_0_0_0_0_0Dv2_fif(%spirv.SampledImage._void_1_0_0_0_0_0_0 addrspace(1)* %TempSampledImage4, <2 x float> %coords, i32 2, float 0.000000e+00)

// CHECK-SPIRV: SampledImage [[SampledImageTy]] [[SampledImage3:[0-9]+]] [[InputImage]] [[ConstSampler2]]
// CHECK-SPIRV: ImageSampleExplicitLod {{.*}} [[SampledImage3]]
// CHECK-LLVM: call spir_func <4 x float> @_Z11read_imagef14ocl_image2d_ro11ocl_samplerDv2_f(%opencl.image2d_ro_t addrspace(1)* %input, %opencl.sampler_t addrspace(2)* %{{[0-9]+}}, <2 x float> %coords)
// CHECK-SPV-IR: call spir_func %spirv.SampledImage._void_1_0_0_0_0_0_0 addrspace(1)* @_Z20__spirv_SampledImagePU3AS133__spirv_Image__void_1_0_0_0_0_0_0PU3AS215__spirv_Sampler(%spirv.Image._void_1_0_0_0_0_0_0 addrspace(1)* %input, %spirv.Sampler addrspace(2)* %1)
// CHECK-SPV-IR: call spir_func <4 x float> @_Z38__spirv_ImageSampleExplicitLod_Rfloat4PU3AS140__spirv_SampledImage__void_1_0_0_0_0_0_0Dv2_fif(%spirv.SampledImage._void_1_0_0_0_0_0_0 addrspace(1)* %TempSampledImage6, <2 x float> %coords, i32 2, float 0.000000e+00)

// CHECK-SPIRV: Function {{.*}} [[sample_kernel_int]]
// CHECK-SPIRV: FunctionParameter {{.*}} [[InputImage:[0-9]+]]
// CHECK-SPIRV: FunctionParameter [[TypeSampler]] [[argSampl:[0-9]+]]
// CHECK-LLVM: define spir_kernel void @sample_kernel_int(%opencl.image2d_ro_t addrspace(1)* %input, <2 x float> %coords, <4 x i32> addrspace(1)* nocapture %results, %opencl.sampler_t addrspace(2)* %argSampl)

// CHECK-SPIRV: SampledImage [[SampledImageTy]] [[SampledImage4:[0-9]+]] [[InputImage]] [[ConstSampler3]]
// CHECK-SPIRV: ImageSampleExplicitLod {{.*}} [[SampledImage4]]
// CHECK-LLVM: call spir_func <4 x i32> @_Z11read_imagei14ocl_image2d_ro11ocl_samplerDv2_f(%opencl.image2d_ro_t addrspace(1)* %input, %opencl.sampler_t addrspace(2)* %0, <2 x float> %coords)
// CHECK-SPV-IR: call spir_func %spirv.SampledImage._void_1_0_0_0_0_0_0 addrspace(1)* @_Z20__spirv_SampledImagePU3AS133__spirv_Image__void_1_0_0_0_0_0_0PU3AS215__spirv_Sampler(%spirv.Image._void_1_0_0_0_0_0_0 addrspace(1)* %input, %spirv.Sampler addrspace(2)* %0)
// CHECK-SPV-IR: call spir_func <4 x i32> @_Z36__spirv_ImageSampleExplicitLod_Rint4PU3AS140__spirv_SampledImage__void_1_0_0_0_0_0_0Dv2_fif(%spirv.SampledImage._void_1_0_0_0_0_0_0 addrspace(1)* %TempSampledImage, <2 x float> %coords, i32 2, float 0.000000e+00)

// CHECK-SPIRV: SampledImage [[SampledImageTy]] [[SampledImage5:[0-9]+]] [[InputImage]] [[argSampl]]
// CHECK-SPIRV: ImageSampleExplicitLod {{.*}} [[SampledImage5]]
// CHECK-LLVM: call spir_func <4 x i32> @_Z11read_imagei14ocl_image2d_ro11ocl_samplerDv2_f(%opencl.image2d_ro_t addrspace(1)* %input, %opencl.sampler_t addrspace(2)* %argSampl, <2 x float> %coords)
// CHECK-SPV-IR: call spir_func %spirv.SampledImage._void_1_0_0_0_0_0_0 addrspace(1)* @_Z20__spirv_SampledImagePU3AS133__spirv_Image__void_1_0_0_0_0_0_0PU3AS215__spirv_Sampler(%spirv.Image._void_1_0_0_0_0_0_0 addrspace(1)* %input, %spirv.Sampler addrspace(2)* %argSampl)
// CHECK-SPV-IR: call spir_func <4 x i32> @_Z36__spirv_ImageSampleExplicitLod_Rint4PU3AS140__spirv_SampledImage__void_1_0_0_0_0_0_0Dv2_fif(%spirv.SampledImage._void_1_0_0_0_0_0_0 addrspace(1)* %TempSampledImage4, <2 x float> %coords, i32 2, float 0.000000e+00)

// CHECK-SPIRV: SampledImage [[SampledImageTy]] [[SampledImage6:[0-9]+]] [[InputImage]] [[ConstSampler4]]
// CHECK-SPIRV: ImageSampleExplicitLod {{.*}} [[SampledImage6]]
// CHECK-LLVM: call spir_func <4 x i32> @_Z11read_imagei14ocl_image2d_ro11ocl_samplerDv2_f(%opencl.image2d_ro_t addrspace(1)* %input, %opencl.sampler_t addrspace(2)* %1, <2 x float> %coords)
// CHECK-SPV-IR: call spir_func %spirv.SampledImage._void_1_0_0_0_0_0_0 addrspace(1)* @_Z20__spirv_SampledImagePU3AS133__spirv_Image__void_1_0_0_0_0_0_0PU3AS215__spirv_Sampler(%spirv.Image._void_1_0_0_0_0_0_0 addrspace(1)* %input, %spirv.Sampler addrspace(2)* %1)
// CHECK-SPV-IR: call spir_func <4 x i32> @_Z36__spirv_ImageSampleExplicitLod_Rint4PU3AS140__spirv_SampledImage__void_1_0_0_0_0_0_0Dv2_fif(%spirv.SampledImage._void_1_0_0_0_0_0_0 addrspace(1)* %TempSampledImage6, <2 x float> %coords, i32 2, float 0.000000e+00)