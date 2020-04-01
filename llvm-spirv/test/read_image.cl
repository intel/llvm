// RUN: %clang_cc1 -triple spir64 -finclude-default-header -O0 -cl-std=CL2.0 -emit-llvm-bc %s -o %t.bc
// RUN: llvm-spirv %t.bc -o %t.spv
// RUN: spirv-val %t.spv
// RUN: llvm-spirv %t.spv -to-text -o - | FileCheck %s --check-prefix=CHECK-SPIRV
// RUN: llvm-spirv -s %t.bc -o %t1.bc
// RUN: llvm-dis %t1.bc -o - | FileCheck %s --check-prefix=CHECK-LLVM

// CHECK-SPIRV: TypeInt [[IntTy:[0-9]+]]
// CHECK-SPIRV: TypeVector [[IVecTy:[0-9]+]] [[IntTy]]
// CHECK-SPIRV: TypeFloat [[FloatTy:[0-9]+]]
// CHECK-SPIRV: TypeVector [[FVecTy:[0-9]+]] [[FloatTy]]
// CHECK-SPIRV: ImageRead [[IVecTy]]
// CHECK-SPIRV: ImageRead [[FVecTy]]

// CHECK-LLVM: call spir_func <4 x i32> @_Z24__spirv_ImageRead_Ruint414ocl_image3d_roDv4_i
// CHECK-LLVM: call spir_func <4 x float> @_Z25__spirv_ImageRead_Rfloat414ocl_image3d_roDv4_i

__kernel void kernelA(__read_only image3d_t input) {
  uint4 c = read_imageui(input, (int4)(0, 0, 0, 0));
}

__kernel void kernelB(__read_only image3d_t input) {
  float4 f = read_imagef(input, (int4)(0, 0, 0, 0));
}
