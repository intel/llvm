; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -spirv-ext=+SPV_INTEL_joint_matrix -o %t.spv
; RUN: llvm-spirv %t.spv -to-text -o - | FileCheck %s
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

; Ensure that ::sycl::_V1.*{half|bfloat16} are parsed as SYCL types.

; CHECK-DAG: TypeFloat [[#HalfTy:]] 16
; CHECK-DAG: TypeInt [[#BFloat16Ty:]] 16

%"class.sycl::_V1::anything::half" = type { half }
%"class.sycl::_V1::anything::bfloat16" = type { i16 }

%"struct.__spv::__spirv_JointMatrixINTEL.half" = type { [2 x [2 x [1 x [4 x %"class.sycl::_V1::anything::half"]]]]* }
%"struct.__spv::__spirv_JointMatrixINTEL.bfloat16" = type { [2 x [2 x [1 x [4 x %"class.sycl::_V1::anything::bfloat16"]]]]* }

define spir_func void @foo(%"struct.__spv::__spirv_JointMatrixINTEL.half" *) {
  ret void
}

define spir_func void @bar(%"struct.__spv::__spirv_JointMatrixINTEL.bfloat16" *) {
  ret void
}
