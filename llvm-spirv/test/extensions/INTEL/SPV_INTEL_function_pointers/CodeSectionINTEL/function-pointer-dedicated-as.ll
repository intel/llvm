; This test checks how a function pointer in a dedicated addr space would be
; translated with and without -spirv-emit-function-ptr-addr-space option.
; Expected behaviour:
; No option is passed to the forward translation stage - no CodeSectionINTEL storage class in SPIR-V
; The option is passed to the forward translation stage - CodeSectionINTEL storage class is generated
; No option is passed to the reverse translation stage - function pointers are in private address space
; The option is passed to the reverse translation stage - function pointers are in addrspace(9)
;
; Overall IR generation is tested elsewhere, here checks are very simple

; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -spirv-text --spirv-ext=+SPV_INTEL_function_pointers -spirv-emit-function-ptr-addr-space -o %t.spt
; RUN: FileCheck < %t.spt %s --check-prefixes=CHECK-SPIRV,CHECK-SPIRV-AS
; RUN: llvm-spirv %t.bc --spirv-ext=+SPV_INTEL_function_pointers -spirv-emit-function-ptr-addr-space -o %t.spv
; RUN: llvm-spirv -r %t.spv -o %t.r.bc
; RUN: llvm-dis %t.r.bc -o %t.r.ll
; RUN: FileCheck < %t.r.ll %s --check-prefix=CHECK-LLVM-NO-AS

; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -spirv-text --spirv-ext=+SPV_INTEL_function_pointers -spirv-emit-function-ptr-addr-space -o %t.spt
; RUN: FileCheck < %t.spt %s --check-prefixes=CHECK-SPIRV,CHECK-SPIRV-AS
; RUN: llvm-spirv %t.bc --spirv-ext=+SPV_INTEL_function_pointers -spirv-emit-function-ptr-addr-space -o %t.spv
; RUN: llvm-spirv -r -spirv-emit-function-ptr-addr-space %t.spv -o %t.r.bc
; RUN: llvm-dis %t.r.bc -o %t.r.ll
; RUN: FileCheck < %t.r.ll %s --check-prefix=CHECK-LLVM-AS

; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -spirv-text --spirv-ext=+SPV_INTEL_function_pointers -o %t.spt
; RUN: FileCheck < %t.spt %s --check-prefix=CHECK-SPIRV-NO-AS
; RUN: llvm-spirv %t.bc --spirv-ext=+SPV_INTEL_function_pointers -o %t.spv
; RUN: llvm-spirv -r %t.spv -o %t.r.bc
; RUN: llvm-dis %t.r.bc -o %t.r.ll
; RUN: FileCheck < %t.r.ll %s --check-prefix=CHECK-LLVM-NO-AS

; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -spirv-text --spirv-ext=+SPV_INTEL_function_pointers -o %t.spt
; RUN: FileCheck < %t.spt %s --check-prefix=CHECK-SPIRV-NO-AS
; RUN: llvm-spirv %t.bc --spirv-ext=+SPV_INTEL_function_pointers -o %t.spv
; RUN: llvm-spirv -r -spirv-emit-function-ptr-addr-space %t.spv -o %t.r.bc
; RUN: llvm-dis %t.r.bc -o %t.r.ll
; RUN: FileCheck < %t.r.ll %s --check-prefix=CHECK-LLVM-AS

; RUN: llvm-spirv %s -spirv-text --spirv-ext=+SPV_INTEL_function_pointers,+SPV_KHR_untyped_pointers -spirv-emit-function-ptr-addr-space -o %t.u.spt
; RUN: FileCheck < %t.u.spt %s --check-prefixes=CHECK-SPIRV,CHECK-SPIRV-UNTYPED
; RUN: llvm-spirv %t.bc --spirv-ext=+SPV_INTEL_function_pointers,+SPV_KHR_untyped_pointers -spirv-emit-function-ptr-addr-space -o %t.u.spv
; RUN: llvm-spirv -r -spirv-emit-function-ptr-addr-space %t.u.spv -o %t.ru.bc
; RUN: llvm-dis %t.ru.bc -o %t.ru.ll
; RUN: FileCheck < %t.ru.ll %s --check-prefix=CHECK-LLVM-AS

; RUN: %if spirv-backend %{ llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_INTEL_function_pointers -filetype=obj %s -o %t.llc.spv %}
; RUN: %if spirv-backend %{ llvm-spirv -r -spirv-emit-function-ptr-addr-space %t.llc.spv -o %t.llc.rev.bc %}
; RUN: %if spirv-backend %{ llvm-dis %t.llc.rev.bc -o %t.llc.rev.ll %}
; RUN: %if spirv-backend %{ FileCheck %s --check-prefixes=CHECK-LLVM-LLC,CHECK-LLVM-LLC-TYPED < %t.llc.rev.ll %}

; RUN: %if spirv-backend %{ llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_INTEL_function_pointers,+SPV_KHR_untyped_pointers -filetype=obj %s -o %t.llc.u.spv %}
; RUN: %if spirv-backend %{ llvm-spirv -r -spirv-emit-function-ptr-addr-space %t.llc.u.spv -o %t.llc.u.rev.bc %}
; RUN: %if spirv-backend %{ llvm-dis %t.llc.u.rev.bc -o %t.llc.u.rev.ll %}
; RUN: %if spirv-backend %{ FileCheck %s --check-prefixes=CHECK-LLVM-LLC,CHECK-LLVM-LLC-UNTYPED < %t.llc.u.rev.ll %}

; CHECK-SPIRV-UNTYPED: Capability UntypedPointersKHR
; CHECK-SPIRV: Capability FunctionPointersINTEL
; CHECK-SPIRV: Extension "SPV_INTEL_function_pointers"
; CHECK-SPIRV-UNTYPED: Extension "SPV_KHR_untyped_pointers"
; CHECK-SPIRV-AS-DAG: TypePointer [[#PtrCodeTy:]] 5605 [[#]]
; CHECK-SPIRV-AS-DAG: TypePointer [[#PtrPrivTy:]] 7 [[#PtrCodeTy]]
; CHECK-SPIRV-UNTYPED-DAG: TypeUntypedPointerKHR [[#PtrCodeTy:]] 5605
; CHECK-SPIRV-UNTYPED-DAG: TypeUntypedPointerKHR [[#PtrPrivTy:]] 7
; CHECK-SPIRV-DAG: ConstantFunctionPointerINTEL [[#PtrCodeTy]] [[#FunPtr:]] [[#Foo:]]
; CHECK-SPIRV: Function [[#]] [[#Foo]]
; CHECK-SPIRV-AS: Variable [[#PtrPrivTy]] [[#Var:]] 7
; CHECK-SPIRV-UNTYPED: UntypedVariableKHR [[#PtrPrivTy]] [[#Var:]] 7 [[#PtrCodeTy]]
; CHECK-SPIRV: Store [[#Var]] [[#FunPtr]]
; CHECK-SPIRV: Load [[#PtrCodeTy]] [[#Load:]] [[#Var]]
; CHECK-SPIRV: FunctionPointerCallINTEL [[#]] [[#]] [[#Load]] [[#]]

; CHECK-SPIRV-NO-AS-NOT: TypePointer [[#]] 5605 [[#]]

; CHECK-LLVM-AS:  define spir_func i32 @foo(i32 %{{.*}}) addrspace(9)

; CHECK-LLVM-NO-AS-NOT: addrspace(9)

; CHECK-LLVM-LLC: %fp = alloca ptr addrspace(9)
; CHECK-LLVM-LLC-TYPED: %[[B1:.*]] = bitcast ptr %fp to ptr
; CHECK-LLVM-LLC-TYPED: store ptr addrspace(9) @foo, ptr %[[B1]]
; CHECK-LLVM-LLC-TYPED: %[[B2:.*]] = bitcast ptr %fp to ptr
; CHECK-LLVM-LLC-TYPED: %[[LD:.*]] = load ptr addrspace(9), ptr %[[B2]]
; CHECK-LLVM-LLC-TYPED: call spir_func addrspace(9) i32 %[[LD]](i32 %{{.*}})
; CHECK-LLVM-LLC-UNTYPED: store ptr addrspace(9) @foo, ptr %fp
; CHECK-LLVM-LLC-UNTYPED: %[[LD:.*]] = load ptr addrspace(9), ptr %fp
; CHECK-LLVM-LLC-UNTYPED: call spir_func addrspace(9) i32 %[[LD]](i32 %{{.*}})

; ModuleID = 'function-pointer-dedicated-as.bc'
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-G1"
target triple = "spir64-unknown-unknown"

; Function Attrs: noinline nounwind
define spir_func i32 @foo(i32 %arg) addrspace(9) #0 {
entry:
  %arg.addr = alloca i32, align 4
  store i32 %arg, ptr %arg.addr, align 4
  %0 = load i32, ptr %arg.addr, align 4
  %add = add nsw i32 %0, 10
  ret i32 %add
}

; Function Attrs: noinline nounwind
define spir_kernel void @test(ptr addrspace(1) %data, i32 %input) #0 !kernel_arg_addr_space !3 !kernel_arg_access_qual !6 !kernel_arg_type !7 !kernel_arg_type_qual !8 !kernel_arg_base_type !7 !spirv.ParameterDecorations !9 {
entry:
  %data.addr = alloca ptr addrspace(1), align 8
  %input.addr = alloca i32, align 4
  %fp = alloca ptr addrspace(9), align 8
  store ptr addrspace(1) %data, ptr %data.addr, align 8
  store i32 %input, ptr %input.addr, align 4
  store ptr addrspace(9) @foo, ptr %fp, align 8
  %0 = load ptr addrspace(9), ptr %fp, align 8
  %1 = load i32, ptr %input.addr, align 4
  %call = call spir_func addrspace(9) i32 %0(i32 %1)
  %2 = load ptr addrspace(1), ptr %data.addr, align 8
  store i32 %call, ptr addrspace(1) %2, align 4
  ret void
}

attributes #0 = { noinline nounwind }

!spirv.MemoryModel = !{!0}
!spirv.Source = !{!1}
!opencl.spir.version = !{!2}
!opencl.ocl.version = !{!3}
!opencl.used.extensions = !{!4}
!opencl.used.optional.core.features = !{!4}
!spirv.Generator = !{!5}

!0 = !{i32 2, i32 2}
!1 = !{i32 3, i32 100000}
!2 = !{i32 1, i32 2}
!3 = !{i32 1, i32 0}
!4 = !{}
!5 = !{i16 6, i16 14}
!6 = !{!"none", !"none"}
!7 = !{!"int*", !"int"}
!8 = !{!"", !""}
!9 = !{!4, !4}
