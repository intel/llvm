; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -spirv-text -o %t.txt
; RUN: FileCheck < %t.txt %s --check-prefix=CHECK-SPIRV
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: spirv-val %t.spv
; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis < %t.rev.bc | FileCheck %s --check-prefix=CHECK-LLVM

; CHECK-LLVM: call spir_func i32 @_Z14work_group_alli(
; CHECK-LLVM: call spir_func i32 @_Z14work_group_anyi(

; CHECK-SPIRV: 2 Capability Groups
; CHECK-SPIRV: 2 TypeBool [[BoolTypeID:[0-9]+]]
; CHECK-SPIRV: 3 ConstantTrue [[BoolTypeID]] [[ConstID:[0-9]+]]
; CHECK-SPIRV: 5 GroupAll [[BoolTypeID]] {{[0-9]+}} {{[0-9]+}} [[ConstID]]
; CHECK-SPIRV: 5 GroupAny [[BoolTypeID]] {{[0-9]+}} {{[0-9]+}} [[ConstID]]

target datalayout = "e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir-unknown-unknown"

; Function Attrs: nounwind
define spir_kernel void @test(ptr addrspace(1) captures(none) readnone %i) #0 !kernel_arg_addr_space !1 !kernel_arg_access_qual !2 !kernel_arg_type !3 !kernel_arg_base_type !4 !kernel_arg_type_qual !5 {
entry:
  %call = tail call spir_func i32 @_Z14work_group_alli(i32 5) #2
  %call1 = tail call spir_func i32 @_Z14work_group_anyi(i32 5) #2
  ret void
}

declare spir_func i32 @_Z14work_group_alli(i32) #1

declare spir_func i32 @_Z14work_group_anyi(i32) #1

attributes #0 = { nounwind "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-realign-stack" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-realign-stack" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { nounwind }

!opencl.enable.FP_CONTRACT = !{}
!opencl.spir.version = !{!6}
!opencl.ocl.version = !{!7}
!opencl.used.extensions = !{!8}
!opencl.used.optional.core.features = !{!8}
!opencl.compiler.options = !{!8}

!1 = !{i32 1}
!2 = !{!"none"}
!3 = !{!"int*"}
!4 = !{!"int*"}
!5 = !{!""}
!6 = !{i32 1, i32 2}
!7 = !{i32 2, i32 0}
!8 = !{}
