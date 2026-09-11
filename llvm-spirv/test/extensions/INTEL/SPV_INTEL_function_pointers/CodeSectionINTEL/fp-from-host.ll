; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -spirv-text --spirv-ext=+SPV_INTEL_function_pointers -o %t.spt
; RUN: FileCheck < %t.spt %s --check-prefixes=CHECK-SPIRV,CHECK-SPIRV-TYPED
; RUN: llvm-spirv %t.bc --spirv-ext=+SPV_INTEL_function_pointers -o %t.spv
; RUN: llvm-spirv -r %t.spv -spirv-emit-function-ptr-addr-space -o %t.r.bc
; RUN: llvm-dis %t.r.bc -o %t.r.ll
; RUN: FileCheck < %t.r.ll %s --check-prefix=CHECK-LLVM

; RUN: llvm-spirv %t.bc -spirv-text --spirv-ext=+SPV_INTEL_function_pointers,+SPV_KHR_untyped_pointers -o %t.u.spt
; RUN: FileCheck < %t.u.spt %s --check-prefixes=CHECK-SPIRV,CHECK-SPIRV-UNTYPED
; RUN: llvm-spirv %t.bc --spirv-ext=+SPV_INTEL_function_pointers,+SPV_KHR_untyped_pointers -o %t.u.spv
; RUN: llvm-spirv -r %t.u.spv -spirv-emit-function-ptr-addr-space -o %t.ru.bc
; RUN: llvm-dis %t.ru.bc -o %t.ru.ll
; RUN: FileCheck < %t.ru.ll %s --check-prefix=CHECK-LLVM-UNTYPED

; RUN: %if spirv-backend %{ llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_INTEL_function_pointers -filetype=obj %s -o %t.llc.spv %}
; RUN: %if spirv-backend %{ llvm-spirv -r -spirv-emit-function-ptr-addr-space %t.llc.spv -o %t.llc.rev.bc %}
; RUN: %if spirv-backend %{ llvm-dis %t.llc.rev.bc -o %t.llc.rev.ll %}
; RUN: %if spirv-backend %{ FileCheck %s --check-prefixes=CHECK-LLVM-LLC < %t.llc.rev.ll %}

; RUN: %if spirv-backend %{ llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_INTEL_function_pointers,+SPV_KHR_untyped_pointers -filetype=obj %s -o %t.llc.u.spv %}
; RUN: %if spirv-backend %{ llvm-spirv -r -spirv-emit-function-ptr-addr-space %t.llc.u.spv -o %t.llc.u.rev.bc %}
; RUN: %if spirv-backend %{ llvm-dis %t.llc.u.rev.bc -o %t.llc.u.rev.ll %}
; RUN: %if spirv-backend %{ FileCheck %s --check-prefixes=CHECK-LLVM-LLC < %t.llc.u.rev.ll %}
;
; Generated from:
; typedef int (*fp_t)(int);
;
; __kernel void test(__global int *fp, __global int *data) {
;
;   data[0] = ((fp_t)(*fp))(data[1]);
; }
;
; CHECK-SPIRV-UNTYPED: Capability UntypedPointersKHR
; CHECK-SPIRV: Capability FunctionPointersINTEL
; CHECK-SPIRV: Extension "SPV_INTEL_function_pointers"
; CHECK-SPIRV-UNTYPED: Extension "SPV_KHR_untyped_pointers"
;
; CHECK-SPIRV: EntryPoint [[#]] [[#KERNEL_ID:]] "test"
; CHECK-SPIRV: TypeInt [[#INT32:]] 32
; CHECK-SPIRV-TYPED: TypePointer [[#INT_PTR:]] 5 [[#INT32]]
; CHECK-SPIRV-UNTYPED: TypeUntypedPointerKHR [[#INT_PTR:]] [[#]]
; CHECK-SPIRV: TypeFunction [[#FOO_TY:]] [[#INT32]] [[#INT32]]
; CHECK-SPIRV-TYPED: TypePointer [[#FOO_TY_PTR:]] [[#]] [[#FOO_TY]]
; CHECK-SPIRV-UNTYPED: TypeUntypedPointerKHR [[#FOO_TY_PTR:]] [[#]]
;
; CHECK-SPIRV: Function [[#]] [[#KERNEL_ID]]
; CHECK-SPIRV: FunctionParameter [[#INT_PTR]] [[#FP:]]
; CHECK-SPIRV: Load [[#INT32]] [[#FUNC_ADDR:]] [[#FP]]
; CHECK-SPIRV: ConvertUToPtr [[#FOO_TY_PTR]] [[#FOO_PTR:]] [[#FUNC_ADDR]]
; CHECK-SPIRV: FunctionPointerCallINTEL [[#INT32]] [[#]] [[#FOO_PTR]]
;
; CHECK-LLVM: define spir_kernel void @test(ptr addrspace(1)
; CHECK-LLVM: %[[FP:.*]] = inttoptr i32 %{{.*}} to ptr addrspace(9)
; CHECK-LLVM: %call = call spir_func addrspace(9) i32 %[[FP]](i32 %{{.*}})
; CHECK-LLVM-UNTYPED: define spir_kernel void @test(ptr addrspace(1)
; CHECK-LLVM-UNTYPED: %[[FP:.*]] = inttoptr i32 %{{.*}} to ptr
; CHECK-LLVM-UNTYPED: %call = call spir_func i32 %[[FP]](i32 %{{.*}})

; CHECK-LLVM-LLC: %[[FP:.*]] = inttoptr i32 %{{.*}} to ptr addrspace(9)
; CHECK-LLVM-LLC: call spir_func addrspace(9) i32 %[[FP]](i32 %{{.*}})

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir64-unknown-unknown"

; Function Attrs: convergent nounwind
define spir_kernel void @test(ptr addrspace(1) %fp, ptr addrspace(1) %data) #0 !kernel_arg_addr_space !4 !kernel_arg_access_qual !5 !kernel_arg_type !6 !kernel_arg_base_type !6 !kernel_arg_type_qual !7 {
entry:
  %arrayidx = getelementptr inbounds i32, ptr addrspace(1) %data, i64 1
  %0 = load i32, ptr addrspace(1) %arrayidx, align 4, !tbaa !8
  %1 = load i32, ptr addrspace(1) %fp, align 4, !tbaa !8
  %2 = inttoptr i32 %1 to ptr
  %call = call spir_func i32 %2(i32 %0) #1
  store i32 %call, ptr addrspace(1) %data, align 4, !tbaa !8
  ret void
}

attributes #0 = { convergent nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "denorms-are-zero"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "uniform-work-group-size"="true" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { convergent }

!llvm.module.flags = !{!0}
!opencl.ocl.version = !{!1}
!opencl.spir.version = !{!2}
!llvm.ident = !{!3}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 1, i32 0}
!2 = !{i32 1, i32 2}
!3 = !{!"clang version 7.1.0 "}
!4 = !{i32 1, i32 1}
!5 = !{!"none", !"none"}
!6 = !{!"int*", !"int*"}
!7 = !{!"", !""}
!8 = !{!9, !9, i64 0}
!9 = !{!"int", !10, i64 0}
!10 = !{!"omnipotent char", !11, i64 0}
!11 = !{!"Simple C/C++ TBAA"}
