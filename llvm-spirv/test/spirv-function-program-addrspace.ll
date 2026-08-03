; Test that --spirv-function-program-addrspace controls the address space of
; function definitions when translating from SPIR-V to LLVM IR.
; Demonstrates interaction with --spirv-addrspace-map,
; --spirv-emit-function-ptr-addr-space (CodeSectionINTEL), and the
; addrspacecast required when the function AS differs from the pointer type AS.

; RUN: llvm-as < %s -o %t.bc
; RUN: llvm-spirv %t.bc --spirv-ext=+SPV_INTEL_function_pointers -o %t.spv
; RUN: llvm-spirv -r %t.spv \
; RUN:   -o - | llvm-dis | FileCheck %s --check-prefix=CHECK-DEFAULT
; RUN: llvm-spirv -r %t.spv --spirv-function-program-addrspace=4 \
; RUN:   -o - | llvm-dis | FileCheck %s --check-prefix=CHECK-AS4
; RUN: llvm-spirv -r %t.spv --spirv-emit-function-ptr-addr-space \
; RUN:   --spirv-function-program-addrspace=4 \
; RUN:   -o - | llvm-dis | FileCheck %s --check-prefix=CHECK-CODESECTION-AS4
; RUN: llvm-spirv -r %t.spv --spirv-addrspace-map=0:4,1:1,2:2,3:3,4:0 \
; RUN:   -o - | llvm-dis | FileCheck %s --check-prefix=CHECK-MAPPED
; RUN: llvm-spirv -r %t.spv --spirv-addrspace-map=0:4,1:1,2:2,3:3,4:0 \
; RUN:   --spirv-function-program-addrspace=3 \
; RUN:   -o - | llvm-dis | FileCheck %s --check-prefix=CHECK-MAPPED-EXPLICIT
; RUN: llvm-spirv -r %t.spv --spirv-addrspace-map=0:4,1:1,2:2,3:3,4:0 \
; RUN:   --spirv-emit-function-ptr-addr-space \
; RUN:   -o - | llvm-dis | FileCheck %s --check-prefix=CHECK-MAPPED-CODESECTION
; RUN: llvm-spirv -r %t.spv --spirv-addrspace-map=0:4,1:1,2:2,3:3,4:0,9:5 \
; RUN:   --spirv-emit-function-ptr-addr-space \
; RUN:   -o - | llvm-dis | FileCheck %s --check-prefix=CHECK-MAPPED-CODESECTION-REMAPPED

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-G1"
target triple = "spir64-unknown-unknown"

; Without any options all functions land in the default AS (Private/AS0).
; CHECK-DEFAULT: define spir_func i32 @id_indirect({{.*}}) #
; CHECK-DEFAULT: define spir_func i32 @id_direct({{.*}}) #
; CHECK-DEFAULT: store ptr @id_indirect, ptr

; --spirv-function-program-addrspace=4 places all functions in AS4. @id_indirect's
; address must be cast from AS4 to AS0 to match the alloca pointer type.
; CHECK-AS4: define spir_func i32 @id_indirect({{.*}}) addrspace(4)
; CHECK-AS4: define spir_func i32 @id_direct({{.*}}) addrspace(4)
; CHECK-AS4: store ptr addrspacecast (ptr addrspace(4) @id_indirect to ptr), ptr

; --spirv-emit-function-ptr-addr-space overrides --spirv-function-program-addrspace
; for functions referenced via OpConstantFunctionPointerINTEL: @id_indirect lands
; in CodeSectionINTEL (AS9), while @id_direct (address not taken) keeps AS4.
; No addrspacecast is needed since the alloca matches @id_indirect's AS,
; intended CodeSectionINTEL behavior.
; CHECK-CODESECTION-AS4: define spir_func i32 @id_indirect({{.*}}) addrspace(9)
; CHECK-CODESECTION-AS4: define spir_func i32 @id_direct({{.*}}) addrspace(4)
; CHECK-CODESECTION-AS4: store ptr addrspace(9) @id_indirect, ptr

; With --spirv-addrspace-map alone, functions land in the AS mapped from Private
; (0->4), so no addrspacecast is needed as all pointers use the same mapping.
; CHECK-MAPPED: define spir_func i32 @id_indirect({{.*}}) addrspace(4)
; CHECK-MAPPED: define spir_func i32 @id_direct({{.*}}) addrspace(4)
; CHECK-MAPPED: store ptr addrspace(4) @id_indirect, ptr addrspace(4)

; --spirv-function-program-addrspace overrides the map: functions land in AS3
; while the alloca pointer type still follows the map (Private->AS4), requiring
; a cast.
; CHECK-MAPPED-EXPLICIT: define spir_func i32 @id_indirect({{.*}}) addrspace(3)
; CHECK-MAPPED-EXPLICIT: define spir_func i32 @id_direct({{.*}}) addrspace(3)
; CHECK-MAPPED-EXPLICIT: store ptr addrspace(4) addrspacecast (ptr addrspace(3) @id_indirect to ptr addrspace(4)), ptr addrspace(4)

; --spirv-emit-function-ptr-addr-space overrides the map for @id_indirect
; (CodeSection/AS9) while @id_direct still follows the map (Private->AS4).
; CHECK-MAPPED-CODESECTION: define spir_func i32 @id_indirect({{.*}}) addrspace(9)
; CHECK-MAPPED-CODESECTION: define spir_func i32 @id_direct({{.*}}) addrspace(4)
; CHECK-MAPPED-CODESECTION: store ptr addrspace(9) @id_indirect, ptr addrspace(4)

; When the map also remaps CodeSectionINTEL (9->5), @id_indirect lands in AS5.
; CHECK-MAPPED-CODESECTION-REMAPPED: define spir_func i32 @id_indirect({{.*}}) addrspace(5)
; CHECK-MAPPED-CODESECTION-REMAPPED: define spir_func i32 @id_direct({{.*}}) addrspace(4)
; CHECK-MAPPED-CODESECTION-REMAPPED: store ptr addrspace(5) @id_indirect, ptr addrspace(4)

define spir_func i32 @id_indirect(i32 %x) #0 {
  ret i32 %x
}

define spir_func i32 @id_direct(i32 %x) #0 {
  ret i32 %x
}

define spir_kernel void @test(ptr addrspace(1) %data) {
  %fp = alloca ptr
  store ptr @id_indirect, ptr %fp
  %fn = load ptr, ptr %fp
  %v = call spir_func i32 %fn(i32 42)
  store i32 %v, ptr addrspace(1) %data
  ret void
}

attributes #0 = { nounwind }
