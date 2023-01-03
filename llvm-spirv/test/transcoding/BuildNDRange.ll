; RUN: llvm-as %s -o %t.bc
; Added -opaque-pointers.
; FIXME: Align with the community code when project is ready to enable opaque
; pointers by default
; RUN: llvm-spirv -opaque-pointers %t.bc -spirv-text -o %t.spv.txt
; RUN: FileCheck < %t.spv.txt %s --check-prefix=CHECK-SPIRV
; Added -opaque-pointers.
; FIXME: Align with the community code when project is ready to enable opaque
; pointers by default
; RUN: llvm-spirv -opaque-pointers %t.bc -o %t.spv
; RUN: spirv-val %t.spv
; Added -opaque-pointers.
; FIXME: Align with the community code when project is ready to enable opaque
; pointers by default
; RUN: llvm-spirv -opaque-pointers -r -emit-opaque-pointers %t.spv -o %t.rev.bc
; Added -opaque-pointers.
; FIXME: Align with the community code when project is ready to enable opaque
; pointers by default
; RUN: llvm-dis -opaque-pointers %t.rev.bc
; RUN: FileCheck < %t.rev.ll %s --check-prefix=CHECK-LLVM
; Added -opaque-pointers.
; FIXME: Align with the community code when project is ready to enable opaque
; pointers by default
; RUN: llvm-spirv -opaque-pointers -r -emit-opaque-pointers %t.spv --spirv-target-env=SPV-IR -o %t.rev.bc
; Added -opaque-pointers.
; FIXME: Align with the community code when project is ready to enable opaque
; pointers by default
; RUN: llvm-dis -opaque-pointers %t.rev.bc
; RUN: FileCheck < %t.rev.ll %s --check-prefix=CHECK-LLVM-SPV

; Added -opaque-pointers.
; FIXME: Align with the community code when project is ready to enable opaque
; pointers by default
; RUN: llvm-spirv -opaque-pointers %t.rev.bc -spirv-text -o %t.spt
; RUN: FileCheck < %t.spt %s --check-prefix=CHECK-SPIRV

; CHECK-SPIRV-DAG: BuildNDRange {{[0-9]+}} {{[0-9]+}} [[GWS:[0-9]+]] [[LWS:[0-9]+]] [[GWO:[0-9]+]]
; CHECK-SPIRV-DAG: Constant {{[0-9]+}} [[GWS]] 123
; CHECK-SPIRV-DAG: Constant {{[0-9]+}} [[LWS]] 456
; CHECK-SPIRV-DAG: Constant {{[0-9]+}} [[GWO]] 0

; CHECK-LLVM: call spir_func void @_Z10ndrange_1Djjj(ptr sret(%struct.ndrange_t) %ndrange, i32 0, i32 123, i32 456)
; CHECK-LLVM-SPV: call spir_func void @_Z23__spirv_BuildNDRange_1Diii(ptr sret(%struct.ndrange_t) %ndrange, i32 123, i32 456, i32 0)

; ModuleID = 'BuildNDRange.bc'
target datalayout = "e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir"

%struct.ndrange_t = type { i32, [3 x i32], [3 x i32], [3 x i32] }

; Function Attrs: nounwind
define spir_kernel void @test() #0 !kernel_arg_addr_space !0 !kernel_arg_access_qual !0 !kernel_arg_type !0 !kernel_arg_base_type !0 !kernel_arg_type_qual !0 {
  %ndrange = alloca %struct.ndrange_t, align 4
  call spir_func void @_Z10ndrange_1Djj(%struct.ndrange_t* sret(%struct.ndrange_t) %ndrange, i32 123, i32 456)
  ret void
}

declare spir_func void @_Z10ndrange_1Djj(%struct.ndrange_t* sret(%struct.ndrange_t), i32, i32) #1

attributes #0 = { nounwind "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-realign-stack" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-realign-stack" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }

!opencl.enable.FP_CONTRACT = !{}
!opencl.spir.version = !{!1}
!opencl.ocl.version = !{!2}
!opencl.used.extensions = !{!0}
!opencl.used.optional.core.features = !{!0}
!opencl.compiler.options = !{!0}

!0 = !{}
!1 = !{i32 1, i32 2}
!2 = !{i32 2, i32 0}
