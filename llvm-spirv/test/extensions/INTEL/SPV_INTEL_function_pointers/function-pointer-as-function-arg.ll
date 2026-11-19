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
; RUN: %if spirv-backend %{ FileCheck %s --check-prefixes=CHECK-LLVM-LLC < %t.llc.rev.ll %}

; TODO: reader currently crashes with output of llc with untyped pointers for this test
; RUN: %if spirv-backend %{ llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_INTEL_function_pointers,+SPV_KHR_untyped_pointers -filetype=obj %s -o %t.llc.u.spv %}
; RUNx: %if spirv-backend %{ llvm-spirv -r %t.llc.u.spv -o %t.llc.u.rev.bc %}
;
; Generated from:
; int helper(int (*f)(int), int arg) {
;   return f(arg);
; }
;
; int foo(int v) {
;   return v + 1;
; }
;
; int bar(int v) {
;   return v + 2;
; }
;
; __kernel void test(__global int *data, int control) {
;   int (*fp)(int) = 0;
;
;   if (get_global_id(0) % control == 0)
;     fp = &foo;
;   else
;     fp = &bar;
;
;   data[get_global_id(0)] = helper(fp, data[get_global_id(0)]);
; }
;
; CHECK-SPIRV-UNTYPED: Capability UntypedPointersKHR
; CHECK-SPIRV: Capability FunctionPointersINTEL
; CHECK-SPIRV: Extension "SPV_INTEL_function_pointers"
; CHECK-SPIRV-UNTYPED: Extension "SPV_KHR_untyped_pointers"
;
; CHECK-SPIRV: EntryPoint [[#]] [[#KERNEL_ID:]] "test"
; CHECK-SPIRV: TypeInt [[#TYPE_INT32_ID:]] 32
; CHECK-SPIRV: TypeFunction [[#FOO_TYPE_ID:]] [[#TYPE_INT32_ID]] [[#TYPE_INT32_ID]]
; CHECK-SPIRV-TYPED: TypePointer [[#FOO_PTR_TYPE_ID:]] {{[0-9]+}} [[#FOO_TYPE_ID]]
; CHECK-SPIRV-UNTYPED: TypeUntypedPointerKHR [[#FOO_PTR_TYPE_ID:]] 7
; CHECK-SPIRV: TypeFunction [[#HELPER_TYPE_ID:]] [[#TYPE_INT32_ID]] [[#FOO_PTR_TYPE_ID]] [[#TYPE_INT32_ID]]
; CHECK-SPIRV-TYPED: TypePointer [[#FOO_PTR_ALLOCA_TYPE_ID:]] {{[0-9]+}} [[#FOO_PTR_TYPE_ID]]
; CHECK-SPIRV-TYPED: TypePointer [[#TYPE_INT32_ALLOCA_ID:]] {{[0-9]+}} [[#TYPE_INT32_ID]]
; CHECK-SPIRV-TYPED: FunctionPointerINTEL [[#FOO_PTR_TYPE_ID]] [[#FOO_PTR_ID:]] [[#FOO_ID:]]
; CHECK-SPIRV-TYPED: FunctionPointerINTEL [[#FOO_PTR_TYPE_ID]] [[#BAR_PTR_ID:]] [[#BAR_ID:]]
; CHECK-SPIRV-UNTYPED: ConstantFunctionPointerINTEL [[#FOO_PTR_TYPE_ID]] [[#FOO_PTR_ID:]] [[#FOO_ID:]]
; CHECK-SPIRV-UNTYPED: ConstantFunctionPointerINTEL [[#FOO_PTR_TYPE_ID]] [[#BAR_PTR_ID:]] [[#BAR_ID:]]
;
; CHECK-SPIRV: Function {{[0-9]+}} [[#HELPER_ID:]] {{[0-9]+}} [[#HELPER_TYPE_ID]]
; CHECK-SPIRV: FunctionParameter [[#FOO_PTR_TYPE_ID]] [[#T_PTR_ARG_ID:]]
; CHECK-SPIRV-TYPED: FunctionParameter [[#TYPE_INT32_ID:]] [[#INT_ARG_ID:]]
; CHECK-SPIRV-TYPED: Variable [[#FOO_PTR_ALLOCA_TYPE_ID]] [[#T_PTR_ALLOCA_ID:]]
; CHECK-SPIRV-UNTYPED: UntypedVariableKHR [[#FOO_PTR_TYPE_ID]] [[#T_PTR_ALLOCA_ID:]] 7 [[#FOO_PTR_TYPE_ID]]
; CHECK-SPIRV-TYPED: Variable [[#TYPE_INT32_ALLOCA_ID]] [[#INT_ALLOCA_ID:]]
; CHECK-SPIRV: Store [[#T_PTR_ALLOCA_ID]] [[#T_PTR_ARG_ID]]
; CHECK-SPIRV-TYPED: Store [[#INT_ALLOCA_ID]] [[#INT_ARG_ID]]
; CHECK-SPIRV: Load [[#FOO_PTR_TYPE_ID]] [[#LOADED_T_PTR:]] [[#T_PTR_ALLOCA_ID]]
; CHECK-SPIRV-TYPED: Load [[#TYPE_INT32_ID]] [[#LOADED_INT:]] [[#INT_ALLOCA_ID]]
; CHECK-SPIRV-TYPED: FunctionPointerCallINTEL [[#TYPE_INT32_ID]] [[#RESULT:]] [[#LOADED_T_PTR]] [[#LOADED_INT]]
; CHECK-SPIRV-UNTYPED: FunctionPointerCallINTEL [[#TYPE_INT32_ID]] [[#]] [[#LOADED_T_PTR]]
; CHECK-SPIRV-TYPED: ReturnValue [[#RESULT]]
;
; CHECK-SPIRV: Function {{[0-9]+}} [[#FOO_ID]] {{[0-9]+}} [[#FOO_TYPE_ID]]
; CHECK-SPIRV: Function {{[0-9]+}} [[#BAR_ID]] {{[0-9]+}} [[#FOO_TYPE_ID]]
;
; CHECK-SPIRV: Function {{[0-9]+}} [[#KERNEL_ID]]
; CHECK-SPIRV-TYPED: Variable [[#FOO_PTR_ALLOCA_TYPE_ID]] [[#F_PTR_ALLOCA_ID:]]
; CHECK-SPIRV-UNTYPED: UntypedVariableKHR [[#FOO_PTR_TYPE_ID]] [[#F_PTR_ALLOCA_ID:]] 7 [[#FOO_PTR_TYPE_ID]]
; CHECK-SPIRV: Store [[#F_PTR_ALLOCA_ID]] [[#FOO_PTR_ID]]
; CHECK-SPIRV: Store [[#F_PTR_ALLOCA_ID]] [[#BAR_PTR_ID]]
; CHECK-SPIRV: Load [[#FOO_PTR_TYPE_ID]] [[#LOADED_F_PTR:]] [[#F_PTR_ALLOCA_ID]]
; CHECK-SPIRV: FunctionCall {{[0-9]+}} {{[0-9]+}} [[#HELPER_ID]] [[#LOADED_F_PTR]]
;
; CHECK-LLVM: define spir_func i32 @helper(ptr %[[F:.*]],
; CHECK-LLVM: %[[F_ADDR:.*]] = alloca ptr
; CHECK-LLVM: store ptr %[[F]], ptr %[[F_ADDR]]
; CHECK-LLVM: %[[F_LOADED:.*]] = load ptr, ptr %[[F_ADDR]]
; CHECK-LLVM: %[[CALL:.*]] = call spir_func i32 %[[F_LOADED]]
; CHECK-LLVM: ret i32 %[[CALL]]
;
; CHECK-LLVM: define spir_kernel void @test
; CHECK-LLVM: %{{.*}} = alloca ptr
; CHECK-LLVM: %[[FP:.*]] = alloca ptr
; CHECK-LLVM: store ptr @foo, ptr %[[FP]]
; CHECK-LLVM: store ptr @bar, ptr %[[FP]]
; CHECK-LLVM: %[[FP_LOADED:.*]] = load ptr, ptr %[[FP]]
; CHECK-LLVM: call spir_func i32 @helper(ptr %[[FP_LOADED]]

; CHECK-LLVM-LLC: %[[FLD:.*]] = load ptr, ptr %{{.*}}
; CHECK-LLVM-LLC: call spir_func i32 %[[FLD]](i32 %{{.*}})


target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir64-unknown-unknown"

; Function Attrs: convergent noinline nounwind optnone
define spir_func i32 @helper(ptr %f, i32 %arg) #0 {
entry:
  %f.addr = alloca ptr, align 8
  %arg.addr = alloca i32, align 4
  store ptr %f, ptr %f.addr, align 8
  store i32 %arg, ptr %arg.addr, align 4
  %0 = load ptr, ptr %f.addr, align 8
  %1 = load i32, ptr %arg.addr, align 4
  %call = call spir_func i32 %0(i32 %1) #3
  ret i32 %call
}

; Function Attrs: convergent noinline nounwind optnone
define spir_func i32 @foo(i32 %v) #0 {
entry:
  %v.addr = alloca i32, align 4
  store i32 %v, ptr %v.addr, align 4
  %0 = load i32, ptr %v.addr, align 4
  %add = add nsw i32 %0, 1
  ret i32 %add
}

; Function Attrs: convergent noinline nounwind optnone
define spir_func i32 @bar(i32 %v) #0 {
entry:
  %v.addr = alloca i32, align 4
  store i32 %v, ptr %v.addr, align 4
  %0 = load i32, ptr %v.addr, align 4
  %add = add nsw i32 %0, 2
  ret i32 %add
}

; Function Attrs: convergent noinline nounwind optnone
define spir_kernel void @test(ptr addrspace(1) %data, i32 %control) #1 !kernel_arg_addr_space !1 !kernel_arg_access_qual !4 !kernel_arg_type !5 !kernel_arg_base_type !5 !kernel_arg_type_qual !6 {
entry:
  %data.addr = alloca ptr addrspace(1), align 8
  %control.addr = alloca i32, align 4
  %fp = alloca ptr, align 8
  store ptr addrspace(1) %data, ptr %data.addr, align 8
  store i32 %control, ptr %control.addr, align 4
  %call = call spir_func i64 @_Z13get_global_idj(i32 0) #4
  %0 = load i32, ptr %control.addr, align 4
  %conv = sext i32 %0 to i64
  %rem = urem i64 %call, %conv
  %cmp = icmp eq i64 %rem, 0
  br i1 %cmp, label %if.then, label %if.else

if.then:                                          ; preds = %entry
  store ptr @foo, ptr %fp, align 8
  br label %if.end

if.else:                                          ; preds = %entry
  store ptr @bar, ptr %fp, align 8
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then
  %1 = load ptr, ptr %fp, align 8
  %2 = load ptr addrspace(1), ptr %data.addr, align 8
  %call2 = call spir_func i64 @_Z13get_global_idj(i32 0) #4
  %arrayidx = getelementptr inbounds i32, ptr addrspace(1) %2, i64 %call2
  %3 = load i32, ptr addrspace(1) %arrayidx, align 4
  %call3 = call spir_func i32 @helper(ptr %1, i32 %3) #3
  %4 = load ptr addrspace(1), ptr %data.addr, align 8
  %call4 = call spir_func i64 @_Z13get_global_idj(i32 0) #4
  %arrayidx5 = getelementptr inbounds i32, ptr addrspace(1) %4, i64 %call4
  store i32 %call3, ptr addrspace(1) %arrayidx5, align 4
  ret void
}

; Function Attrs: convergent nounwind readnone
declare spir_func i64 @_Z13get_global_idj(i32) #2

attributes #0 = { convergent noinline nounwind optnone "correctly-rounded-divide-sqrt-fp-math"="false" "denorms-are-zero"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { convergent noinline nounwind optnone "correctly-rounded-divide-sqrt-fp-math"="false" "denorms-are-zero"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "uniform-work-group-size"="true" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { convergent nounwind readnone "correctly-rounded-divide-sqrt-fp-math"="false" "denorms-are-zero"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { convergent }
attributes #4 = { convergent nounwind readnone }

!llvm.module.flags = !{!0}
!opencl.ocl.version = !{!1}
!opencl.spir.version = !{!2}
!llvm.ident = !{!3}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 1, i32 0}
!2 = !{i32 1, i32 2}
!3 = !{!"clang version 7.1.0 "}
!4 = !{!"none", !"none"}
!5 = !{!"int*", !"int"}
!6 = !{!"", !""}
