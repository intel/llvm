// RUN: %clang_cc1 -triple spir64 -fdeclare-opencl-builtins -finclude-default-header -O0 -cl-std=CL2.0 -emit-llvm-bc %s -o %t.bc
// RUN: llvm-spirv %t.bc -o %t.spv
// RUN: spirv-val %t.spv
// RUN: llvm-spirv %t.spv -to-text -o - | FileCheck %s --check-prefix=CHECK-SPIRV
// RUN: llvm-spirv -r %t.spv -o %t.rev.bc --spirv-target-env=SPV-IR
// RUN: llvm-dis < %t.rev.bc | FileCheck %s --check-prefix=CHECK-SPV-LLVM
// RUN: llvm-spirv %t.rev.bc -o %t.rev.spv
// RUN: spirv-val %t.rev.spv
// RUN: llvm-spirv %t.rev.spv -to-text -o - | FileCheck %s --check-prefix=CHECK-SPIRV

// CHECK-SPIRV: TypeInt [[IntTy:[0-9]+]]
// CHECK-SPIRV: TypeVector [[IVecTy:[0-9]+]] [[IntTy]]
// CHECK-SPIRV: TypeFloat [[FloatTy:[0-9]+]]
// CHECK-SPIRV: TypeVector [[FVecTy:[0-9]+]] [[FloatTy]]
// CHECK-SPIRV: ImageRead [[IVecTy]]
// CHECK-SPIRV: ImageRead [[FVecTy]]

// CHECK-SPV-LLVM: call spir_func <4 x i32> @_Z24__spirv_ImageRead_Ruint4PU3AS133__spirv_Image__void_2_0_0_0_0_0_0Dv4_i(%spirv.Image._void_2_0_0_0_0_0_0 addrspace(1)*
// CHECK-SPV-LLVM: call spir_func <4 x float> @_Z25__spirv_ImageRead_Rfloat4PU3AS133__spirv_Image__void_2_0_0_0_0_0_0Dv4_i(%spirv.Image._void_2_0_0_0_0_0_0 addrspace(1)*

__kernel void kernelA(__read_only image3d_t input) {
  uint4 c = read_imageui(input, (int4)(0, 0, 0, 0));
}

__kernel void kernelB(__read_only image3d_t input) {
  float4 f = read_imagef(input, (int4)(0, 0, 0, 0));
}
