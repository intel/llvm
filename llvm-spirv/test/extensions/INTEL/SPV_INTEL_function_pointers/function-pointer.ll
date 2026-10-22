; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -spirv-text --spirv-ext=+SPV_INTEL_function_pointers -o %t.spt
; RUN: FileCheck < %t.spt %s --check-prefixes=CHECK-SPIRV,CHECK-SPIRV-TYPED
; RUN: llvm-spirv %t.bc --spirv-ext=+SPV_INTEL_function_pointers -o %t.spv
; RUN: llvm-spirv -r %t.spv -o %t.r.bc
; RUN: llvm-dis %t.r.bc -o %t.r.ll
; RUN: FileCheck < %t.r.ll %s --check-prefix=CHECK-LLVM
; RUN: llvm-spirv %t.bc -spirv-text --spirv-ext=+SPV_INTEL_function_pointers,+SPV_KHR_untyped_pointers -o %t.u.spt
; RUN: FileCheck < %t.u.spt %s --check-prefixes=CHECK-SPIRV,CHECK-SPIRV-UNTYPED
; RUN: llvm-spirv %t.bc --spirv-ext=+SPV_INTEL_function_pointers,+SPV_KHR_untyped_pointers -o %t.u.spv
; RUN: llvm-spirv -r %t.u.spv -o %t.ru.bc
; RUN: llvm-dis %t.ru.bc -o %t.ru.ll
; RUN: FileCheck < %t.ru.ll %s --check-prefix=CHECK-LLVM

; RUN: %if spirv-backend %{ llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_INTEL_function_pointers -filetype=obj %s -o %t.llc.spv %}
; RUN: %if spirv-backend %{ llvm-spirv -r %t.llc.spv -o %t.llc.rev.bc %}
; RUN: %if spirv-backend %{ llvm-dis %t.llc.rev.bc -o %t.llc.rev.ll %}
; RUN: %if spirv-backend %{ FileCheck %s --check-prefixes=CHECK-LLVM-LLC,CHECK-LLVM-LLC-TYPED < %t.llc.rev.ll %}

; RUN: %if spirv-backend %{ llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_INTEL_function_pointers,+SPV_KHR_untyped_pointers -filetype=obj %s -o %t.llc.u.spv %}
; RUN: %if spirv-backend %{ llvm-spirv -r %t.llc.u.spv -o %t.llc.u.rev.bc %}
; RUN: %if spirv-backend %{ llvm-dis %t.llc.u.rev.bc -o %t.llc.u.rev.ll %}
; RUN: %if spirv-backend %{ FileCheck %s --check-prefixes=CHECK-LLVM-LLC,CHECK-LLVM-LLC-UNTYPED < %t.llc.u.rev.ll %}
;
; Generated from:
; int foo(int arg) {
;   return arg + 10;
; }
;
; void __kernel test(__global int *data, int input) {
;   int (__constant *fp)(int) = &foo;
;
;   *data = fp(input);
; }
;
; CHECK-SPIRV-UNTYPED: Capability UntypedPointersKHR
; CHECK-SPIRV: Capability FunctionPointersINTEL
; CHECK-SPIRV: Extension "SPV_INTEL_function_pointers"
; CHECK-SPIRV-UNTYPED: Extension "SPV_KHR_untyped_pointers"
; CHECK-SPIRV: EntryPoint [[#]] [[#KERNEL_ID:]] "test"
; CHECK-SPIRV: TypeInt [[#TYPE_INT_ID:]]
; CHECK-SPIRV: TypeFunction [[#FOO_TYPE_ID:]] [[#TYPE_INT_ID]] [[#TYPE_INT_ID]]
; CHECK-SPIRV-TYPED: TypePointer [[#FOO_PTR_ID:]] {{[0-9]+}} [[#FOO_TYPE_ID]]
; CHECK-SPIRV-TYPED: TypePointer [[#FOO_PTR_ALLOCA_ID:]] 7 [[#FOO_PTR_ID]]
; CHECK-SPIRV-UNTYPED: TypeUntypedPointerKHR [[#FOO_PTR_ID:]] 7
; CHECK-SPIRV: ConstantFunctionPointerINTEL [[#FOO_PTR_ID]] [[#FOO_PTR:]] [[#FOO_ID:]]
;
; CHECK-SPIRV: Function {{[0-9]+}} [[#FOO_ID]] {{[0-9]+}} [[#FOO_TYPE_ID]]
; CHECK-SPIRV-UNTYPED: UntypedVariableKHR [[#FOO_PTR_ID]] [[#]] 7 [[#TYPE_INT_ID]]
; CHECK-SPIRV: Function {{[0-9]+}} [[#KERNEL_ID]]
; CHECK-SPIRV-TYPED: Variable [[#FOO_PTR_ALLOCA_ID]] [[#FOO_PTR_ALLOCA:]]
; CHECK-SPIRV-UNTYPED: UntypedVariableKHR [[#FOO_PTR_ID]] [[#FOO_PTR_ALLOCA:]] 7 [[#FOO_PTR_ID]]
; CHECK-SPIRV: Store [[#FOO_PTR_ALLOCA]] [[#FOO_PTR]]
; CHECK-SPIRV: Load [[#FOO_PTR_ID]] [[#LOADED_FOO_PTR:]] [[#FOO_PTR_ALLOCA]]
; CHECK-SPIRV-TYPED: FunctionPointerCallINTEL 2 {{[0-9]+}} [[#LOADED_FOO_PTR]]
; CHECK-SPIRV-UNTYPED: FunctionPointerCallINTEL [[#TYPE_INT_ID]] {{[0-9]+}} [[#LOADED_FOO_PTR]]
;
; CHECK-LLVM: define spir_kernel void @test
; CHECK-LLVM: %fp = alloca ptr
; CHECK-LLVM: store ptr @foo, ptr %fp
; CHECK-LLVM: %0 = load ptr, ptr %fp
; CHECK-LLVM: %call = call spir_func i32 %0(i32 %1)

; CHECK-LLVM-LLC: %fp = alloca ptr
; CHECK-LLVM-LLC-TYPED: %[[B1:.*]] = bitcast ptr %fp to ptr
; CHECK-LLVM-LLC-TYPED: %[[B2:.*]] = bitcast ptr %[[B1]] to ptr
; CHECK-LLVM-LLC-TYPED: store ptr @foo, ptr %[[B2]]
; CHECK-LLVM-LLC-TYPED: %[[B3:.*]] = bitcast ptr %fp to ptr
; CHECK-LLVM-LLC-TYPED: %[[LD:.*]] = load ptr, ptr %[[B3]]
; CHECK-LLVM-LLC-TYPED: call spir_func i32 %[[LD]](i32 %{{.*}})
; CHECK-LLVM-LLC-UNTYPED: %[[B1:.*]] = bitcast ptr %fp to ptr
; CHECK-LLVM-LLC-UNTYPED: store ptr @foo, ptr %[[B1]]
; CHECK-LLVM-LLC-UNTYPED: %[[LD:.*]] = load ptr, ptr %fp
; CHECK-LLVM-LLC-UNTYPED: call spir_func i32 %[[LD]](i32 %{{.*}})

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir64-unknown-unknown"

; Function Attrs: convergent noinline nounwind optnone
define spir_func i32 @foo(i32 %arg) #0 {
entry:
  %arg.addr = alloca i32, align 4
  store i32 %arg, ptr %arg.addr, align 4
  %0 = load i32, ptr %arg.addr, align 4
  %add = add nsw i32 %0, 10
  ret i32 %add
}

; Function Attrs: convergent noinline nounwind optnone
define spir_kernel void @test(ptr addrspace(1) %data, i32 %input) #1 !kernel_arg_addr_space !1 !kernel_arg_access_qual !5 !kernel_arg_type !6 !kernel_arg_base_type !6 !kernel_arg_type_qual !7 {
entry:
  %data.addr = alloca ptr addrspace(1), align 8
  %input.addr = alloca i32, align 4
  %fp = alloca ptr, align 8
  store ptr addrspace(1) %data, ptr %data.addr, align 8
  store i32 %input, ptr %input.addr, align 4
  store ptr @foo, ptr %fp, align 8
  %0 = load ptr, ptr %fp, align 8
  %1 = load i32, ptr %input.addr, align 4
  %call = call spir_func i32 %0(i32 %1) #2
  %2 = load ptr addrspace(1), ptr %data.addr, align 8
  store i32 %call, ptr addrspace(1) %2, align 4
  ret void
}

attributes #0 = { convergent noinline nounwind optnone "correctly-rounded-divide-sqrt-fp-math"="false" "denorms-are-zero"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { convergent noinline nounwind optnone "correctly-rounded-divide-sqrt-fp-math"="false" "denorms-are-zero"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "uniform-work-group-size"="true" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { convergent }

!llvm.module.flags = !{!0}
!opencl.ocl.version = !{!1}
!opencl.spir.version = !{!2}
!opencl.used.extensions = !{!3}
!opencl.used.optional.core.features = !{!3}
!opencl.compiler.options = !{!3}
!llvm.ident = !{!4}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 1, i32 0}
!2 = !{i32 1, i32 2}
!3 = !{}
!4 = !{!"clang version 7.0.0 "}
!5 = !{!"none", !"none"}
!6 = !{!"int*", !"int"}
!7 = !{!"", !""}

