; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc --spirv-ext=+all -spirv-text -o %t
; RUN: FileCheck < %t %s

; CHECK: Name [[#NAME:]] "struct._ZTS6Object.Object"
; CHECK-COUNT-1: TypeStruct [[#NAME]]
; CHECK: TypePointer [[#PTRTY:]] {{.*}} [[#NAME]]
; CHECK: FunctionParameter [[#PTRTY]]

; ModuleID = 'sycl_test.bc'
source_filename = "sycl_test.cpp"
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

%struct._ZTS4Args.Args = type { %struct._ZTS6Object.Object addrspace(4)* }
%struct._ZTS6Object.Object = type { i32 (%struct._ZTS6Object.Object addrspace(4)*, i32)* }

; Function Attrs: convergent norecurse nounwind mustprogress
define dso_local spir_func i32 @_Z9somefunc0P4Args(%struct._ZTS4Args.Args addrspace(4)* %a, %struct._ZTS6Object.Object addrspace(4)* %b) {
entry:
  ret i32 0
}

!opencl.spir.version = !{!0}
!spirv.Source = !{!1}
!0 = !{i32 1, i32 2}
!1 = !{i32 4, i32 100000}
