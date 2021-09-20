; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -spirv-text --spirv-ext=+SPV_INTEL_function_pointers -o %t.spt
; RUN: FileCheck < %t.spt %s --check-prefix=CHECK-SPIRV
; RUN: llvm-spirv %t.bc --spirv-ext=+SPV_INTEL_function_pointers -o %t.spv
; RUN: llvm-spirv -r %t.spv -o %t.r.bc
; RUN: llvm-dis %t.r.bc -o %t.r.ll
; RUN: FileCheck < %t.r.ll %s --check-prefix=CHECK-LLVM
;
; Generated from:
; typedef int (*fp_t)(int);
;
; __kernel void test(__global int *fp, __global int *data) {
;
;   data[0] = ((fp_t)(*fp))(data[1]);
; }
;
; CHECK-SPIRV: Capability FunctionPointersINTEL
; CHECK-SPIRV: Extension "SPV_INTEL_function_pointers"
;
; CHECK-SPIRV: Name [[KERNEL_ID:[0-9]+]] "test"
; CHECK-SPIRV: TypeInt [[INT32_TYPE_ID:[0-9]+]] 32
; CHECK-SPIRV: TypePointer [[INT_PTR:[0-9]+]] 5 [[INT32_TYPE_ID]]
; CHECK-SPIRV: TypeFunction [[FOO_TYPE_ID:[0-9]+]] [[INT32_TYPE_ID]] [[INT32_TYPE_ID]]
; CHECK-SPIRV: TypePointer [[FOO_TYPE_PTR_ID:[0-9]+]] {{[0-9]+}} [[FOO_TYPE_ID]]
;
; CHECK-SPIRV: Function {{[0-9]+}} [[KERNEL_ID]]
; CHECK-SPIRV: FunctionParameter [[INT_PTR]] [[FP:[0-9]+]]
; CHECK-SPIRV: Load [[INT32_TYPE_ID]] [[FUNC_ADDR:[0-9]+]] [[FP]]
; CHECK-SPIRV: ConvertUToPtr [[FOO_TYPE_PTR_ID]] [[FOO_PTR:[0-9]+]] [[FUNC_ADDR]]
; CHECK-SPIRV: FunctionPointerCallINTEL [[INT32_TYPE_ID]] {{[0-9]+}} [[FOO_PTR]]
;
; CHECK-LLVM: define spir_kernel void @test(i32 addrspace(1)*
; CHECK-LLVM: %{{.*}} = call spir_func i32 %{{.*}}(i32

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir64-unknown-unknown"

; Function Attrs: convergent nounwind
define spir_kernel void @test(i32 addrspace(1)* %fp, i32 addrspace(1)* %data) #0 !kernel_arg_addr_space !4 !kernel_arg_access_qual !5 !kernel_arg_type !6 !kernel_arg_base_type !6 !kernel_arg_type_qual !7 {
entry:
  %arrayidx = getelementptr inbounds i32, i32 addrspace(1)* %data, i64 1
  %0 = load i32, i32 addrspace(1)* %arrayidx, align 4, !tbaa !8
  %1 = load i32, i32 addrspace(1)* %fp, align 4, !tbaa !8
  %2 = inttoptr i32 %1 to i32 (i32)*
  %call = call spir_func i32 %2(i32 %0) #1
  %arrayidx1 = getelementptr inbounds i32, i32 addrspace(1)* %data, i64 0
  store i32 %call, i32 addrspace(1)* %arrayidx1, align 4, !tbaa !8
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
