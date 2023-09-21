; RUN: sed -e s/.T1://g %s | opt -mtriple=amdgcn-amd-amdhsa -passes=amdgpu-add-global-for-atomic-xor -S | FileCheck %s --check-prefix=CHECK1
; RUN: sed -e s/.T2://g %s | opt -mtriple=amdgcn-amd-amdhsa -passes=amdgpu-add-global-for-atomic-xor -S | FileCheck %s --check-prefix=CHECK2

define i32 @test(ptr %p) {
; CHECK1:      @HipAtomicXorModuleNeedsPrefetch
; CHECK2-NOT:  @HipAtomicXorModuleNeedsPrefetch
;T1:   %1 = atomicrmw volatile xor ptr %p, i32 1 syncscope("agent-one-as") monotonic, align 4
;T2:   %1 = atomicrmw volatile add ptr %p, i32 1 syncscope("agent-one-as") monotonic, align 4
  ret i32 %1
}

