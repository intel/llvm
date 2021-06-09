; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -spirv-ext=+SPV_INTEL_function_pointers -o %t.spv
; RUN: llvm-spirv -r %t.spv -o %t.r.bc
; RUN: llvm-dis %t.r.bc -o %t.r.ll
; RUN: FileCheck < %t.r.ll %s --check-prefix=CHECK-LLVM

; ModuleID = 't.bc'
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024"
target triple = "spir64-unknown-unknown"

%structtype.3 = type { [1 x i8 addrspace(4)*] }

; CHECK-LLVM: @A = addrspace(1) constant %structtype.3 { [1 x i8 addrspace(4)*] [i8 addrspace(4)* addrspacecast (i8* bitcast (void ()* @foo to i8*) to i8 addrspace(4)*)] }, align 8

@A = linkonce_odr addrspace(1) constant %structtype.3 { [1 x i8 addrspace(4)*] [i8 addrspace(4)* addrspacecast (i8* bitcast (void ()* @foo to i8*) to i8 addrspace(4)*)] }, align 8

; Function Attrs: nounwind
define linkonce_odr spir_func void @foo() #0 {
entry:
; CHECK-LLVM: %0 = getelementptr inbounds %structtype.3, %structtype.3 addrspace(1)* @A, i64 0, i32 0, i64 2
  %0 = getelementptr inbounds %structtype.3, %structtype.3 addrspace(1)* @A, i64 0, i32 0, i64 2
  ret void
}

attributes #0 = { nounwind }

!spirv.MemoryModel = !{!0}
!spirv.Source = !{!1}
!opencl.spir.version = !{!2}
!opencl.ocl.version = !{!2}
!opencl.used.extensions = !{!3}
!opencl.used.optional.core.features = !{!4}
!spirv.Generator = !{!5}

!0 = !{i32 2, i32 2}
!1 = !{i32 4, i32 200000}
!2 = !{i32 2, i32 0}
!3 = !{!"cl_khr_int64_extended_atomics", !"cl_khr_subgroups"}
!4 = !{!"cl_doubles"}
!5 = !{i16 6, i16 14}
