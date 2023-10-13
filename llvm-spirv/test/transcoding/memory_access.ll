; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -spirv-text -o %t.spt
; RUN: FileCheck < %t.spt %s --check-prefix=CHECK-SPIRV
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: spirv-val %t.spv
; RUN: llvm-spirv -r --spirv-target-env=CL2.0 %t.spv -o %t.bc
; RUN: llvm-dis < %t.bc | FileCheck %s --check-prefix=CHECK-LLVM

; CHECK-SPIRV-NOT: 6 Store {{[0-9]+}} {{[0-9]+}} 1 2 8
; CHECK-SPIRV: 5 Store {{[0-9]+}} {{[0-9]+}} 3 8
; CHECK-SPIRV-NOT: 7 Load {{[0-9]+}} {{[0-9]+}} {{[0-9]+}} 1 2 8
; CHECK-SPIRV: 6 Load {{[0-9]+}} {{[0-9]+}} {{[0-9]+}} 3 8
; CHECK-SPIRV: 6 Load {{[0-9]+}} {{[0-9]+}} {{[0-9]+}} 2 4
; CHECK-SPIRV-NOT: 7 Load {{[0-9]+}} {{[0-9]+}} {{[0-9]+}} 1 2 8
; CHECK-SPIRV: 6 Load {{[0-9]+}} {{[0-9]+}} {{[0-9]+}} 3 8
; CHECK-SPIRV-NOT: 7 Load {{[0-9]+}} {{[0-9]+}} {{[0-9]+}} 1 2 0
; CHECK-SPIRV: 6 Load {{[0-9]+}} {{[0-9]+}} {{[0-9]+}} 3 8
; CHECK-SPIRV-NOT: 7 Load {{[0-9]+}} {{[0-9]+}} {{[0-9]+}} 1 2 8
; CHECK-SPIRV: 6 Load {{[0-9]+}} {{[0-9]+}} {{[0-9]+}} 7 8
; CHECK-SPIRV-NOT: 5 Store {{[0-9]+}} {{[0-9]+}} 2 4
; CHECK-SPIRV: 5 Store {{[0-9]+}} {{[0-9]+}} 6 4
; CHECK-SPIRV-NOT: 5 Store {{[0-9]+}} {{[0-9]+}} 2 0
; CHECK-SPIRV: 5 Store {{[0-9]+}} {{[0-9]+}}

; CHECK-LLVM: store volatile ptr addrspace(4) %0, ptr %ptr, align 8
; CHECK-LLVM: load volatile ptr addrspace(4), ptr %ptr, align 8
; CHECK-LLVM: load i32, ptr addrspace(4) %1, align 4
; CHECK-LLVM: load volatile ptr addrspace(4), ptr %ptr, align 8
; CHECK-LLVM: load volatile ptr addrspace(4), ptr %ptr
; CHECK-LLVM: %[[VOLATILELOAD:[0-9]+]] = load volatile ptr addrspace(4), ptr %ptr, align 8, !nontemporal ![[NTMetadata:[0-9]+]]
; CHECK-LLVM: store i32 %call, ptr addrspace(4) %arrayidx, align 4, !nontemporal ![[NTMetadata:[0-9]+]]
; CHECK-LLVM: store ptr addrspace(4) %[[VOLATILELOAD]], ptr %ptr
; CHECK-LLVM: ![[NTMetadata:[0-9]+]] = !{i32 1}

; ModuleID = 'test.bc'
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

; Function Attrs: nounwind
define spir_kernel void @test_load_store(ptr addrspace(1) %destMemory, ptr addrspace(1) %oldValues, i32 %newValue) #0 !kernel_arg_addr_space !1 !kernel_arg_access_qual !2 !kernel_arg_type !3 !kernel_arg_base_type !4 !kernel_arg_type_qual !5 {
entry:
  %ptr = alloca ptr addrspace(4), align 8
  %0 = addrspacecast ptr addrspace(1) %oldValues to ptr addrspace(4)
  store volatile ptr addrspace(4) %0, ptr %ptr, align 8
  %1 = load volatile ptr addrspace(4), ptr %ptr, align 8
  %2 = load i32, ptr addrspace(4) %1, align 4
  %call = call spir_func i32 @_Z14atomic_cmpxchgPVU3AS1iii(ptr addrspace(1) %destMemory, i32 %2, i32 %newValue)
  %3 = load volatile ptr addrspace(4), ptr %ptr, align 8
  %4 = load volatile ptr addrspace(4), ptr %ptr
  %5 = load volatile ptr addrspace(4), ptr %ptr, align 8, !nontemporal !9
  %arrayidx = getelementptr inbounds i32, ptr addrspace(4) %3, i64 0
  store i32 %call, ptr addrspace(4) %arrayidx, align 4, !nontemporal !9
  store ptr addrspace(4) %5, ptr %ptr
  ret void
}

declare spir_func i32 @_Z14atomic_cmpxchgPVU3AS1iii(ptr addrspace(1), i32, i32) #1

attributes #0 = { nounwind "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-realign-stack" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-realign-stack" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }

!opencl.enable.FP_CONTRACT = !{}
!opencl.spir.version = !{!6}
!opencl.ocl.version = !{!7}
!opencl.used.extensions = !{!8}
!opencl.used.optional.core.features = !{!8}
!opencl.compiler.options = !{!8}
!llvm.ident = !{!10}

!1 = !{i32 1, i32 1, i32 0}
!2 = !{!"none", !"none", !"none"}
!3 = !{!"int*", !"int*", !"int"}
!4 = !{!"int*", !"int*", !"int"}
!5 = !{!"volatile", !"", !""}
!6 = !{i32 1, i32 2}
!7 = !{i32 2, i32 0}
!8 = !{}
!9 = !{i32 1}
!10 = !{!"clang version 3.6.1"}
