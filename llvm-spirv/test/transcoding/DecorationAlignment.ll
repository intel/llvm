; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: llvm-spirv %t.spv -to-text -o - | FileCheck %s --check-prefix=CHECK-SPIRV
; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis < %t.rev.bc | FileCheck %s --check-prefix=CHECK-LLVM

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

; CHECK-SPIRV: Decorate [[ALIGNMENT:[0-9]+]] Alignment 16
; CHECK-SPIRV: FunctionParameter {{[0-9]+}} [[ALIGNMENT]]

%struct._ZTS6Struct.Struct = type { %struct._ZTS11floatStruct.floatStruct, %struct._ZTS11floatStruct.floatStruct }
%struct._ZTS11floatStruct.floatStruct = type { float, float, float, float }

; CHECK-LLVM: define spir_func void @_ZN3FooC2Ev(%struct._ZTS6Struct.Struct addrspace(4)* align 16 %0)
; Function Attrs: noinline nounwind
define spir_func void @_ZN3FooC2Ev(%struct._ZTS6Struct.Struct addrspace(4)* align 16 %0) #0 {
  ret void
}

attributes #0 = { noinline nounwind }

!llvm.module.flags = !{!0, !1}
!opencl.spir.version = !{!2}
!spirv.Source = !{!3}
!opencl.used.extensions = !{!4}
!opencl.used.optional.core.features = !{!4}
!opencl.compiler.options = !{!4}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 7, !"frame-pointer", i32 2}
!2 = !{i32 1, i32 2}
!3 = !{i32 4, i32 100000}
!4 = !{}
