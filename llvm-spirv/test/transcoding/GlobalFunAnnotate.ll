;RUN: llvm-as %s -o %t.bc
;RUN: llvm-spirv %t.bc -o %t.spv
;RUN: llvm-spirv %t.spv -to-text -o - | FileCheck %s --check-prefix=CHECK-SPIRV
;RUN: llvm-spirv -r %t.spv -o %t.rev.bc
;RUN: llvm-dis %t.rev.bc -o - | FileCheck %s --check-prefix=CHECK-LLVM

;CHECK-SPIRV: Decorate {{[0-9]+}} UserSemantic "annotation_on_function"

;CHECK-LLVM: @0 = private unnamed_addr constant [23 x i8] c"annotation_on_function\00", section "llvm.metadata"
;CHECK-LLVM: @llvm.global.annotations = appending global [1 x { ptr, ptr, ptr, i32, ptr }] [{ ptr, ptr, ptr, i32, ptr } { ptr @foo, ptr @0, ptr undef, i32 undef, ptr undef }], section "llvm.metadata"

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-linux"

@.str = private unnamed_addr addrspace(1) constant [23 x i8] c"annotation_on_function\00", section "llvm.metadata"
@.str.1 = private unnamed_addr addrspace(1) constant [6 x i8] c"an.cl\00", section "llvm.metadata"
@llvm.global.annotations = appending global [1 x { ptr, ptr addrspace(1), ptr addrspace(1), i32, ptr }] [{ ptr, ptr addrspace(1), ptr addrspace(1), i32, ptr } { ptr @foo, ptr addrspace(1) @.str, ptr addrspace(1) @.str.1, i32 2, ptr null }], section "llvm.metadata"

; Function Attrs: convergent norecurse nounwind
define dso_local spir_func void @foo() #0 {
entry:
  ret void
}

attributes #0 = { convergent norecurse nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.module.flags = !{!0}
!opencl.ocl.version = !{!1}
!opencl.spir.version = !{!2, !2}
!spirv.Source = !{!3}
!llvm.ident = !{!4}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 1, i32 0}
!2 = !{i32 1, i32 2}
!3 = !{i32 4, i32 100000}
!4 = !{!"clang version 12.0.0 (https://github.com/c199914007/llvm.git 074e97d48896b959dfc832e2f1dd806796cde390)"}
