; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -spirv-text -o %t.txt
; RUN: FileCheck < %t.txt %s --check-prefix=CHECK-SPIRV
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: spirv-val %t.spv
; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis < %t.rev.bc | FileCheck %s --check-prefix=CHECK-LLVM

; CHECK-SPIRV: TypeBool [[bool:[0-9]+]]
; CHECK-SPIRV: TypeVector [[bool2:[0-9]+]] [[bool]] 2

target datalayout = "e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir-unknown-unknown"

; CHECK-SPIRV-LABEL: 5 Function
; CHECK-SPIRV-NEXT: FunctionParameter {{[0-9]+}} [[A:[0-9]+]]
; CHECK-SPIRV-NEXT: FunctionParameter {{[0-9]+}} [[B:[0-9]+]]
; CHECK-SPIRV: 5 FUnordEqual [[bool2]] {{[0-9]+}} [[A]] [[B]]
; CHECK-SPIRV: FunctionEnd

; CHECK-LLVM-LABEL: @testFUnordEqual
; CHECK-LLVM: fcmp ueq <2 x float> %a, %b

; Function Attrs: nounwind
define spir_kernel void @testFUnordEqual(<2 x float> %a, <2 x float> %b) #0 !kernel_arg_addr_space !2 !kernel_arg_access_qual !3 !kernel_arg_type !4 !kernel_arg_type_qual !5 !kernel_arg_base_type !4 {
entry:
  %0 = fcmp ueq <2 x float> %a, %b
  ret void
}

; CHECK-SPIRV-LABEL: 5 Function
; CHECK-SPIRV-NEXT: FunctionParameter {{[0-9]+}} [[A:[0-9]+]]
; CHECK-SPIRV-NEXT: FunctionParameter {{[0-9]+}} [[B:[0-9]+]]
; CHECK-SPIRV: 5 FUnordGreaterThan [[bool2]] {{[0-9]+}} [[A]] [[B]]
; CHECK-SPIRV: FunctionEnd

; CHECK-LLVM-LABEL: @testFUnordGreaterThan
; CHECK-LLVM: fcmp ugt <2 x float> %a, %b

; Function Attrs: nounwind
define spir_kernel void @testFUnordGreaterThan(<2 x float> %a, <2 x float> %b) #0 !kernel_arg_addr_space !2 !kernel_arg_access_qual !3 !kernel_arg_type !4 !kernel_arg_type_qual !5 !kernel_arg_base_type !4 {
entry:
  %0 = fcmp ugt <2 x float> %a, %b
  ret void
}

; CHECK-SPIRV-LABEL: 5 Function
; CHECK-SPIRV-NEXT: FunctionParameter {{[0-9]+}} [[A:[0-9]+]]
; CHECK-SPIRV-NEXT: FunctionParameter {{[0-9]+}} [[B:[0-9]+]]
; CHECK-SPIRV: 5 FUnordGreaterThanEqual [[bool2]] {{[0-9]+}} [[A]] [[B]]
; CHECK-SPIRV: FunctionEnd

; CHECK-LLVM-LABEL: @testFUnordGreaterThanEqual
; CHECK-LLVM: fcmp uge <2 x float> %a, %b

; Function Attrs: nounwind
define spir_kernel void @testFUnordGreaterThanEqual(<2 x float> %a, <2 x float> %b) #0 !kernel_arg_addr_space !2 !kernel_arg_access_qual !3 !kernel_arg_type !4 !kernel_arg_type_qual !5 !kernel_arg_base_type !4 {
entry:
  %0 = fcmp uge <2 x float> %a, %b
  ret void
}

; CHECK-SPIRV-LABEL: 5 Function
; CHECK-SPIRV-NEXT: FunctionParameter {{[0-9]+}} [[A:[0-9]+]]
; CHECK-SPIRV-NEXT: FunctionParameter {{[0-9]+}} [[B:[0-9]+]]
; CHECK-SPIRV: 5 FUnordLessThan [[bool2]] {{[0-9]+}} [[A]] [[B]]
; CHECK-SPIRV: FunctionEnd

; CHECK-LLVM-LABEL: @testFUnordLessThan
; CHECK-LLVM: fcmp ult <2 x float> %a, %b

; Function Attrs: nounwind
define spir_kernel void @testFUnordLessThan(<2 x float> %a, <2 x float> %b) #0 !kernel_arg_addr_space !2 !kernel_arg_access_qual !3 !kernel_arg_type !4 !kernel_arg_type_qual !5 !kernel_arg_base_type !4 {
entry:
  %0 = fcmp ult <2 x float> %a, %b
  ret void
}

; CHECK-SPIRV-LABEL: 5 Function
; CHECK-SPIRV-NEXT: FunctionParameter {{[0-9]+}} [[A:[0-9]+]]
; CHECK-SPIRV-NEXT: FunctionParameter {{[0-9]+}} [[B:[0-9]+]]
; CHECK-SPIRV: 5 FUnordLessThanEqual [[bool2]] {{[0-9]+}} [[A]] [[B]]
; CHECK-SPIRV: FunctionEnd

; CHECK-LLVM-LABEL: @testFUnordLessThanEqual
; CHECK-LLVM: fcmp ule <2 x float> %a, %b

; Function Attrs: nounwind
define spir_kernel void @testFUnordLessThanEqual(<2 x float> %a, <2 x float> %b) #0 !kernel_arg_addr_space !2 !kernel_arg_access_qual !3 !kernel_arg_type !4 !kernel_arg_type_qual !5 !kernel_arg_base_type !4 {
entry:
  %0 = fcmp ule <2 x float> %a, %b
  ret void
}

attributes #0 = { nounwind }

!opencl.enable.FP_CONTRACT = !{}
!opencl.spir.version = !{!0}
!opencl.ocl.version = !{!0}
!opencl.used.extensions = !{!1}
!opencl.used.optional.core.features = !{!1}

!0 = !{i32 2, i32 0}
!1 = !{}
!2 = !{i32 0, i32 0}
!3 = !{!"none", !"none"}
!4 = !{!"float2", !"float2"}
!5 = !{!"", !""}
