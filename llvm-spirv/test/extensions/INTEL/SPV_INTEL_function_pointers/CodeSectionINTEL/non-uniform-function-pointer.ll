; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -spirv-text --spirv-ext=+SPV_INTEL_function_pointers -o %t.spt
; RUN: FileCheck < %t.spt %s --check-prefixes=CHECK-SPIRV,CHECK-SPIRV-TYPED
; RUN: llvm-spirv %t.bc --spirv-ext=+SPV_INTEL_function_pointers -o %t.spv
; RUN: llvm-spirv -r -spirv-emit-function-ptr-addr-space %t.spv -o %t.r.bc
; RUN: llvm-dis %t.r.bc -o %t.r.ll
; RUN: FileCheck < %t.r.ll %s --check-prefix=CHECK-LLVM

; RUN: llvm-spirv %t.bc -spirv-text --spirv-ext=+SPV_INTEL_function_pointers,+SPV_KHR_untyped_pointers -o %t.u.spt
; RUN: FileCheck < %t.u.spt %s --check-prefixes=CHECK-SPIRV,CHECK-SPIRV-UNTYPED
; RUN: llvm-spirv %t.bc --spirv-ext=+SPV_INTEL_function_pointers,+SPV_KHR_untyped_pointers -o %t.u.spv
; RUN: llvm-spirv -r -spirv-emit-function-ptr-addr-space %t.u.spv -o %t.ru.bc
; RUN: llvm-dis %t.ru.bc -o %t.ru.ll
; RUN: FileCheck < %t.ru.ll %s --check-prefix=CHECK-LLVM-UNTYPED

; RUN: %if spirv-backend %{ llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_INTEL_function_pointers -filetype=obj %s -o %t.llc.spv %}
; RUN: %if spirv-backend %{ llvm-spirv -r -spirv-emit-function-ptr-addr-space %t.llc.spv -o %t.llc.rev.bc %}
; RUN: %if spirv-backend %{ llvm-dis %t.llc.rev.bc -o %t.llc.rev.ll %}
; RUN: %if spirv-backend %{ FileCheck %s --check-prefixes=CHECK-LLVM-LLC < %t.llc.rev.ll %}

; TODO: reader currently crashes with output of llc with untyped pointers for this test
; RUN: %if spirv-backend %{ llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_INTEL_function_pointers,+SPV_KHR_untyped_pointers -filetype=obj %s -o %t.llc.u.spv %}
; RUNx: %if spirv-backend %{ llvm-spirv -r -spirv-emit-function-ptr-addr-space %t.llc.u.spv -o %t.llc.u.rev.bc %}
;
; Generated from:
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
;   data[get_global_id(0)] = fp(data[get_global_id(0)]);
; }
;
; CHECK-SPIRV-UNTYPED: Capability UntypedPointersKHR
; CHECK-SPIRV: Capability FunctionPointersINTEL
; CHECK-SPIRV: Extension "SPV_INTEL_function_pointers"
; CHECK-SPIRV-UNTYPED: Extension "SPV_KHR_untyped_pointers"
;
; CHECK-SPIRV: EntryPoint [[#]] [[#KERNEL_ID:]] "test"
; CHECK-SPIRV-DAG: TypeInt [[#INT:]] 32
; CHECK-SPIRV-DAG: TypeFunction [[#FOO_TY:]] [[#INT]] [[#INT]]
; CHECK-SPIRV-TYPED-DAG: TypePointer [[#PTR:]] [[#]] [[#FOO_TY]]
; CHECK-SPIRV-TYPED-DAG: TypePointer [[#ALLOCA_TY:]] 7 [[#PTR]]
; CHECK-SPIRV-UNTYPED-DAG: TypeUntypedPointerKHR [[#PTR:]] [[#]]
; CHECK-SPIRV-TYPED-DAG: ConstantFunctionPointerINTEL [[#PTR]] [[#FOO_PTR:]] [[#FOO:]]
; CHECK-SPIRV-TYPED-DAG: ConstantFunctionPointerINTEL [[#PTR]] [[#BAR_PTR:]] [[#BAR:]]
; CHECK-SPIRV-UNTYPED-DAG: ConstantFunctionPointerINTEL [[#PTR]] [[#FOO_PTR:]] [[#FOO:]]
; CHECK-SPIRV-UNTYPED-DAG: ConstantFunctionPointerINTEL [[#PTR]] [[#BAR_PTR:]] [[#BAR:]]
;
; CHECK-SPIRV-TYPED: Function [[#]] [[#FOO]] [[#]] [[#FOO_TY]]
; CHECK-SPIRV-TYPED: Function [[#]] [[#BAR]] [[#]] [[#FOO_TY]]
; CHECK-SPIRV-UNTYPED: Function [[#]] [[#FOO]] [[#]] [[#FOO_TY]]
; CHECK-SPIRV-UNTYPED: Function [[#]] [[#BAR]] [[#]] [[#FOO_TY]]
;
; CHECK-SPIRV: Function [[#]] [[#KERNEL_ID]]
; CHECK-SPIRV-TYPED: Variable [[#ALLOCA_TY]] [[#ALLOCA:]]
; CHECK-SPIRV-UNTYPED: UntypedVariableKHR [[#PTR]] [[#ALLOCA:]] [[#]] [[#PTR]]
; CHECK-SPIRV: Store [[#ALLOCA]] [[#FOO_PTR]]
; CHECK-SPIRV: Store [[#ALLOCA]] [[#BAR_PTR]]
; CHECK-SPIRV: Load [[#PTR]] [[#LOADED:]] [[#ALLOCA]]
; CHECK-SPIRV: FunctionPointerCallINTEL [[#]] [[#]] [[#LOADED]]
;
; CHECK-LLVM: define spir_kernel void @test
; CHECK-LLVM: %fp = alloca ptr addrspace(9)
; CHECK-LLVM: store ptr addrspace(9) @foo, ptr %fp
; CHECK-LLVM: store ptr addrspace(9) @bar, ptr %fp
; CHECK-LLVM: %[[FP:.*]] = load ptr addrspace(9), ptr %fp
; CHECK-LLVM: call spir_func addrspace(9) i32 %[[FP]](i32 %{{.*}})

; CHECK-LLVM-UNTYPED: define spir_kernel void @test
; CHECK-LLVM-UNTYPED: %fp = alloca ptr
; CHECK-LLVM-UNTYPED: store ptr addrspacecast (ptr addrspace(9) @foo to ptr), ptr %fp
; CHECK-LLVM-UNTYPED: store ptr addrspacecast (ptr addrspace(9) @bar to ptr), ptr %fp
; CHECK-LLVM-UNTYPED: %[[FP:.*]] = load ptr, ptr %fp
; CHECK-LLVM-UNTYPED: call spir_func i32 %[[FP]](i32 %{{.*}})

; CHECK-LLVM-LLC: %fp = alloca ptr
; CHECK-LLVM-LLC: %[[LD:.*]] = load ptr addrspace(9), ptr %{{.*}}
; CHECK-LLVM-LLC: call spir_func addrspace(9) i32 %[[LD]](i32 %{{.*}})


target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir64-unknown-unknown"

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
  %call = call spir_func i64 @_Z13get_global_idj(i32 0) #3
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
  %call2 = call spir_func i64 @_Z13get_global_idj(i32 0) #3
  %arrayidx = getelementptr inbounds i32, ptr addrspace(1) %2, i64 %call2
  %3 = load i32, ptr addrspace(1) %arrayidx, align 4
  %call3 = call spir_func i32 %1(i32 %3) #4
  %4 = load ptr addrspace(1), ptr %data.addr, align 8
  %call4 = call spir_func i64 @_Z13get_global_idj(i32 0) #3
  %arrayidx5 = getelementptr inbounds i32, ptr addrspace(1) %4, i64 %call4
  store i32 %call3, ptr addrspace(1) %arrayidx5, align 4
  ret void
}

; Function Attrs: convergent nounwind readnone
declare spir_func i64 @_Z13get_global_idj(i32) #2

attributes #0 = { convergent noinline nounwind optnone "correctly-rounded-divide-sqrt-fp-math"="false" "denorms-are-zero"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { convergent noinline nounwind optnone "correctly-rounded-divide-sqrt-fp-math"="false" "denorms-are-zero"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "uniform-work-group-size"="true" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { convergent nounwind readnone "correctly-rounded-divide-sqrt-fp-math"="false" "denorms-are-zero"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { convergent nounwind readnone }
attributes #4 = { convergent }

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
