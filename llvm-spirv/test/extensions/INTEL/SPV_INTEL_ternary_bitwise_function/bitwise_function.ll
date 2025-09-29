; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc --spirv-ext=+SPV_INTEL_ternary_bitwise_function -o %t.spv
; RUN: llvm-spirv %t.spv --to-text -o %t.spt
; RUN: FileCheck < %t.spt %s --check-prefix=CHECK-SPIRV

; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis < %t.rev.bc | FileCheck %s --check-prefix=CHECK-LLVM

; RUN: not llvm-spirv %t.bc 2>&1 | FileCheck %s --check-prefix=CHECK-ERROR
; CHECK-ERROR: RequiresExtension: Feature requires the following SPIR-V extension:
; CHECK-ERROR-NEXT: SPV_INTEL_ternary_bitwise_function

; CHECK-SPIRV-NOT: Name [[#]] "_Z28__spirv_BitwiseFunctionINTELiiij"
; CHECK-SPIRV-NOT: Name [[#]] "_Z28__spirv_BitwiseFunctionINTELDv4_iS_S_j"

; CHECK-SPIRV-DAG: Capability TernaryBitwiseFunctionINTEL
; CHECK-SPIRV-DAG: Extension "SPV_INTEL_ternary_bitwise_function"

; CHECK-SPIRV-DAG: TypeInt [[#TYPEINT:]] 32 0
; CHECK-SPIRV-DAG: TypeVector [[#TYPEINTVEC4:]] [[#TYPEINT]] 4
; CHECK-SPIRV-DAG: Constant [[#TYPEINT]] [[#ScalarLUT:]] 24
; CHECK-SPIRV-DAG: Constant [[#TYPEINT]] [[#VecLUT:]] 42

; CHECK-SPIRV: Load [[#TYPEINT]] [[#ScalarA:]]
; CHECK-SPIRV: Load [[#TYPEINT]] [[#ScalarB:]]
; CHECK-SPIRV: Load [[#TYPEINT]] [[#ScalarC:]]
; CHECK-SPIRV: BitwiseFunctionINTEL [[#TYPEINT]] {{.*}} [[#ScalarA]] [[#ScalarB]] [[#ScalarC]] [[#ScalarLUT]]
; CHECK-SPIRV: Load [[#TYPEINTVEC4]] [[#VecA:]]
; CHECK-SPIRV: Load [[#TYPEINTVEC4]] [[#VecB:]]
; CHECK-SPIRV: Load [[#TYPEINTVEC4]] [[#VecC:]]
; CHECK-SPIRV: BitwiseFunctionINTEL [[#TYPEINTVEC4]] {{.*}} [[#VecA]] [[#VecB]] [[#VecC]] [[#VecLUT]]

; CHECK-LLVM: %[[ScalarA:.*]] = load i32, ptr
; CHECK-LLVM: %[[ScalarB:.*]] = load i32, ptr
; CHECK-LLVM: %[[ScalarC:.*]] = load i32, ptr
; CHECK-LLVM: call spir_func i32 @_Z28__spirv_BitwiseFunctionINTELiiii(i32 %[[ScalarA]], i32 %[[ScalarB]], i32 %[[ScalarC]], i32 24)
; CHECK-LLVM: %[[VecA:.*]] = load <4 x i32>, ptr
; CHECK-LLVM: %[[VecB:.*]] = load <4 x i32>, ptr
; CHECK-LLVM: %[[VecC:.*]] = load <4 x i32>, ptr
; CHECK-LLVM: call spir_func <4 x i32> @_Z28__spirv_BitwiseFunctionINTELDv4_iS_S_i(<4 x i32> %[[VecA]], <4 x i32> %[[VecB]], <4 x i32> %[[VecC]], i32 42)

target datalayout = "e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir"

; Function Attrs: nounwind readnone
define spir_kernel void @fooScalar() {
entry:
  %argA = alloca i32
  %argB = alloca i32
  %argC = alloca i32
  %A = load i32, ptr %argA
  %B = load i32, ptr %argB
  %C = load i32, ptr %argC
  %res = call spir_func i32 @_Z28__spirv_BitwiseFunctionINTELiiii(i32 %A, i32 %B, i32 %C, i32 24)
  ret void
}

; Function Attrs: nounwind readnone
define spir_kernel void @fooVec() {
entry:
  %argA = alloca <4 x i32>
  %argB = alloca <4 x i32>
  %argC = alloca <4 x i32>
  %A = load <4 x i32>, ptr %argA
  %B = load <4 x i32>, ptr %argB
  %C = load <4 x i32>, ptr %argC
  %res = call spir_func <4 x i32> @_Z28__spirv_BitwiseFunctionINTELDv4_iS_S_i(<4 x i32> %A, <4 x i32> %B, <4 x i32> %C, i32 42)
  ret void
}

declare dso_local spir_func i32 @_Z28__spirv_BitwiseFunctionINTELiiii(i32, i32, i32, i32)
declare dso_local spir_func <4 x i32> @_Z28__spirv_BitwiseFunctionINTELDv4_iS_S_i(<4 x i32>, <4 x i32>, <4 x i32>, i32)

!llvm.module.flags = !{!0}
!opencl.spir.version = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 1, i32 2}
