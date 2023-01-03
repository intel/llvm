// RUN: %clang_cc1 -triple spir64 -fdeclare-opencl-builtins -finclude-default-header -O0 -cl-std=CL2.0 -emit-llvm-bc %s -o %t.bc
// Added -opaque-pointers.
// FIXME: Align with the community code when project is ready to enable opaque
// pointers by default
// RUN: llvm-spirv -opaque-pointers --spirv-max-version=1.3 %t.bc -o %t.spv
// RUN: spirv-val %t.spv
// Added -opaque-pointers.
// FIXME: Align with the community code when project is ready to enable opaque
// pointers by default
// RUN: llvm-spirv -opaque-pointers %t.spv -to-text -o - | FileCheck %s --check-prefix=CHECK-SPIRV
// Added -opaque-pointers.
// FIXME: Align with the community code when project is ready to enable opaque
// pointers by default
// RUN: llvm-spirv -r %t.spv -o %t.rev.bc --spirv-target-env=SPV-IR
// Added -opaque-pointers.
// FIXME: Align with the community code when project is ready to enable opaque
// pointers by default
// RUN: llvm-dis -opaque-pointers -opaque-pointers=0 < %t.rev.bc | FileCheck %s --check-prefix=CHECK-SPV-LLVM
// Added -opaque-pointers.
// FIXME: Align with the community code when project is ready to enable opaque
// pointers by default
// RUN: llvm-spirv -opaque-pointers --spirv-max-version=1.3 %t.rev.bc -o %t.rev.spv
// RUN: spirv-val %t.rev.spv
// Added -opaque-pointers.
// FIXME: Align with the community code when project is ready to enable opaque
// pointers by default
// RUN: llvm-spirv -opaque-pointers --spirv-max-version=1.3 %t.rev.spv -to-text -o - | FileCheck %s --check-prefix=CHECK-SPIRV

// CHECK-SPIRV: TypeInt [[IntTy:[0-9]+]] 32
// CHECK-SPIRV: TypeVector [[IVecTy:[0-9]+]] [[IntTy]]
// CHECK-SPIRV: TypeFloat [[FloatTy:[0-9]+]]
// CHECK-SPIRV: TypeVector [[FVecTy:[0-9]+]] [[FloatTy]]
// CHECK-SPIRV: ImageRead [[IVecTy]]
// CHECK-SPIRV: ImageRead [[FVecTy]]

// CHECK-SPV-LLVM: call spir_func <4 x i32> @_Z23__spirv_ImageRead_Rint4PU3AS133__spirv_Image__void_2_0_0_0_0_0_0Dv4_i(%spirv.Image._void_2_0_0_0_0_0_0 addrspace(1)*
// CHECK-SPV-LLVM: call spir_func <4 x float> @_Z25__spirv_ImageRead_Rfloat4PU3AS133__spirv_Image__void_2_0_0_0_0_0_0Dv4_i(%spirv.Image._void_2_0_0_0_0_0_0 addrspace(1)*

__kernel void kernelA(__read_only image3d_t input) {
  uint4 c = read_imageui(input, (int4)(0, 0, 0, 0));
}

__kernel void kernelB(__read_only image3d_t input) {
  float4 f = read_imagef(input, (int4)(0, 0, 0, 0));
}
