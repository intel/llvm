; RUN: llvm-spirv %s -o %t.spt -spirv-text -spirv-ext=+SPV_INTEL_function_pointers
; RUN: FileCheck < %t.spt %s --check-prefix=CHECK-SPIRV

; RUN: llvm-spirv %t.spt -o %t.spv -to-binary
; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis %t.rev.bc -o %t.rev.ll
; RUN: FileCheck < %t.rev.ll %s --check-prefix CHECK-LLVM

; RUN: llvm-spirv %s -o %t.u.spt -spirv-text -spirv-ext=+SPV_INTEL_function_pointers,+SPV_KHR_untyped_pointers
; RUN: FileCheck < %t.u.spt %s --check-prefixes=CHECK-SPIRV,CHECK-SPIRV-UNTYPED

; RUN: llvm-spirv %t.u.spt -o %t.u.spv -to-binary
; RUN: llvm-spirv -r %t.u.spv -o %t.ru.bc
; RUN: llvm-dis %t.ru.bc -o %t.ru.ll
; RUN: FileCheck < %t.ru.ll %s --check-prefix CHECK-LLVM-UNTYPED

; RUN: %if spirv-backend %{ llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_INTEL_function_pointers -filetype=obj %s -o %t.llc.spv %}
; RUN: %if spirv-backend %{ llvm-spirv -r %t.llc.spv -o %t.llc.rev.bc %}
; RUN: %if spirv-backend %{ llvm-dis %t.llc.rev.bc -o %t.llc.rev.ll %}
; RUN: %if spirv-backend %{ FileCheck %s --check-prefixes=CHECK-LLVM-LLC < %t.llc.rev.ll %}

; RUN: %if spirv-backend %{ llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_INTEL_function_pointers,+SPV_KHR_untyped_pointers -filetype=obj %s -o %t.llc.u.spv %}
; RUN: %if spirv-backend %{ llvm-spirv -r %t.llc.u.spv -o %t.llc.u.rev.bc %}
; RUN: %if spirv-backend %{ llvm-dis %t.llc.u.rev.bc -o %t.llc.u.rev.ll %}
; RUN: %if spirv-backend %{ FileCheck %s --check-prefixes=CHECK-LLVM-LLC < %t.llc.u.rev.ll %}

; CHECK-SPIRV-UNTYPED: Capability UntypedPointersKHR
; CHECK-SPIRV: Capability FunctionPointersINTEL
; CHECK-SPIRV: Extension "SPV_INTEL_function_pointers"
; CHECK-SPIRV-UNTYPED: Extension "SPV_KHR_untyped_pointers"

; CHECK-SPIRV: Decorate [[#TargetId:]] ArgumentAttributeINTEL 0 5
; CHECK-SPIRV: Decorate [[#TargetId]] ArgumentAttributeINTEL 0 4
; CHECK-SPIRV: Decorate [[#TargetId]] ArgumentAttributeINTEL 0 2
; CHECK-SPIRV-UNTYPED: TypeUntypedPointerKHR [[#PtrTy:]]
; CHECK-SPIRV-UNTYPED: ConstantFunctionPointerINTEL [[#PtrTy]] [[#FnPtr:]] [[#Inc:]]
; CHECK-SPIRV-UNTYPED: Function [[#]] [[#Inc]]
; CHECK-SPIRV-UNTYPED: UntypedVariableKHR [[#PtrTy]] [[#Var:]]
; CHECK-SPIRV-UNTYPED: Select [[#PtrTy]] [[#Sel:]] [[#]] [[#FnPtr]] [[#]]
; CHECK-SPIRV-UNTYPED: Bitcast [[#]] [[#ByvalArg:]] [[#Var]]
; CHECK-SPIRV: FunctionPointerCallINTEL
; CHECK-SPIRV-SAME: [[#TargetId]]
; CHECK-SPIRV-UNTYPED-SAME: [[#Sel]] [[#ByvalArg]]

; CHECK-LLVM: call spir_func void %cond.i.i(ptr noalias byval(%multi_ptr) captures(none) %agg.tmp.i.i)

; CHECK-LLVM-UNTYPED: %[[AGG:.*]] = alloca %multi_ptr
; CHECK-LLVM-UNTYPED: %[[BC:.*]] = bitcast ptr %[[AGG]] to ptr
; CHECK-LLVM-UNTYPED: call spir_func void %cond.i.i(ptr noalias byval(%multi_ptr) captures(none) %[[BC]])

; CHECK-LLVM-LLC: %[[SEL:.*]] = select i1 %{{.*}}, ptr @inc_function, ptr %{{.*}}
; CHECK-LLVM-LLC: %[[FP:.*]] = bitcast ptr %[[SEL]] to ptr
; CHECK-LLVM-LLC: call spir_func void %[[FP]](ptr %{{.*}})

; ModuleID = 'sycl_test.cpp'
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

%"multi_ptr" = type { ptr }
%"range" = type { %"array" }
%"array" = type { [1 x i64] }
%wrapper_class = type { ptr addrspace(1) }
%wrapper_class.0 = type { ptr addrspace(1) }

$RoundedRangeKernel = comdat any

; Function Attrs: nounwind
define spir_func void @inc_function(ptr byval(%"multi_ptr") noalias captures(none) %ptr) #0 {
entry:
  ret void
}


; Function Attrs: convergent norecurse
define weak_odr dso_local spir_kernel void @RoundedRangeKernel(ptr byval(%"range") align 8 %_arg_NumWorkItems, i1 zeroext %_arg_, ptr byval(%wrapper_class) align 8 %_arg_1, ptr byval(%wrapper_class.0) align 8 %_arg_2) local_unnamed_addr #0 comdat !kernel_arg_buffer_location !6 {
entry:
  %agg.tmp.i.i = alloca %"multi_ptr", align 8
  %cond.i.i = select i1 %_arg_, ptr @inc_function, ptr null
  call spir_func void %cond.i.i(ptr nonnull byval(%"multi_ptr") align 8 noalias captures(none) %agg.tmp.i.i) #1, !callees !7
  ret void
}

attributes #0 = { convergent norecurse "denormal-fp-math"="preserve-sign,preserve-sign" "denormal-fp-math-f32"="ieee,ieee" "frame-pointer"="all" "min-legal-vector-width"="0" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "sycl-module-id"="sycl_test.cpp" "uniform-work-group-size"="true" "unsafe-fp-math"="true" }
attributes #1 = { convergent }

!llvm.module.flags = !{!0, !1}
!opencl.spir.version = !{!2}
!spirv.Source = !{!3}
!opencl.used.extensions = !{!4}
!opencl.used.optional.core.features = !{!4}
!opencl.compiler.options = !{!4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 7, !"frame-pointer", i32 2}
!2 = !{i32 1, i32 2}
!3 = !{i32 4, i32 100000}
!4 = !{}
!5 = !{!"Compiler"}
!6 = !{i32 -1, i32 -1, i32 -1, i32 -1}
!7 = !{ptr @inc_function}
