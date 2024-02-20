; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc --spirv-ext=+SPV_INTEL_function_pointers -spirv-text -o - | FileCheck %s
; RUN: llvm-spirv %t.bc --spirv-ext=+SPV_INTEL_function_pointers -o %t.spv
; RUN: llvm-spirv -r %t.spv


target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

@llvm.global_ctors = appending addrspace(1) global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 65535, ptr @_GLOBAL__sub_I_test.cpp.ctor, ptr null }]
@llvm.global_dtors = appending addrspace(1) global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 65535, ptr @_GLOBAL__sub_I_test.cpp.dtor, ptr null }]
@_ZL15DIVERGENCE = internal addrspace(1) global i32 0, align 4

; CHECK: Name [[NameCtor:[0-9]+]] "_GLOBAL__sub_I_test.cpp.ctor"
; CHECK: Name [[Name1:[0-9]+]] "llvm.global_ctors"
; CHECK: Name [[NameDtor:[0-9]+]] "_GLOBAL__sub_I_test.cpp.dtor"
; CHECK: Name [[Name2:[0-9]+]] "llvm.global_dtors"
; CHECK: Name [[NameInit:[0-9]+]] "__cxx_global_var_init"

; CHECK-DAG: Decorate {{[0-9]+}} LinkageAttributes "llvm.global_ctors" Export
; CHECK-DAG: Decorate {{[0-9]+}} LinkageAttributes "llvm.global_dtors" Export

; CHECK: TypeFunction {{[0-9]+}} [[TF:[0-9]+]]

; CHECK: TypePointer [[TP:[0-9]+]]
; CHECK: ConstantFunctionPointerINTEL [[TP]] [[FPCtor:[0-9]+]] [[NameCtor]]
; CHECK: SpecConstantOp {{[0-9]+}} [[FPCtorI8:[0-9]+]] 124 [[FPCtor]]
; CHECK: ConstantComposite {{[0-9]+}} {{[0-9]+}} {{[0-9]+}} [[FPCtorI8]]
; CHECK: ConstantFunctionPointerINTEL [[TP]] [[FPDtor:[0-9]+]] [[NameDtor]]
; CHECK: SpecConstantOp {{[0-9]+}} [[FPDtorI8:[0-9]+]] 124 [[FPDtor]]
; CHECK: ConstantComposite {{[0-9]+}} {{[0-9]+}} {{[0-9]+}} [[FPDtorI8]]

; CHECK: 5 Function [[TF]] [[NameCtor]] 0
; CHECK-EMPTY:
; CHECK-NEXT: Label {{[0-9]+}}
; CHECK-NEXT: FunctionCall {{[0-9]+}} {{[0-9]+}} [[NameInit]]
; CHECK-NEXT: Return
; CHECK-EMPTY:
; CHECK-NEXT: FunctionEnd

; Function Attrs: nounwind sspstrong
define internal void @_GLOBAL__sub_I_test.cpp.ctor() #0 {
  call void @__cxx_global_var_init()
  ret void
}

; Function Attrs: nounwind sspstrong
define internal void @__cxx_global_var_init() #0 {
  store i32 0, ptr addrspace(4) addrspacecast (ptr addrspace(1) @_ZL15DIVERGENCE to ptr addrspace(4)), align 4
  ret void
}

; CHECK: 5 Function [[TF]] [[NameDtor]] 0
; CHECK-EMPTY:
; CHECK-NEXT: Label {{[0-9]+}}
; CHECK-NEXT: Return
; CHECK-EMPTY:
; CHECK-NEXT: FunctionEnd

; Function Attrs: nounwind sspstrong
define internal void @_GLOBAL__sub_I_test.cpp.dtor() #0 {
  ret void
}

; Ctor/dtor functions should not be serialized twice.
; CHECK-NOT: 5 Function [[TF]] [[NameCtor]] 0
; CHECK-NOT: 5 Function [[TF]] [[NameDtor]] 0

attributes #0 = { nounwind sspstrong "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }

!opencl.spir.version = !{!0}
!opencl.used.extensions = !{!1}
!opencl.used.optional.core.features = !{!1}
!opencl.compiler.options = !{!1}

!0 = !{i32 1, i32 2}
!1 = !{}