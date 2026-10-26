; --spirv-target-triple derives the reverse address-space map.
;
; Reverse translation renumbers each address space from SPIR to the target's.
; Under amdgcn the derived map applies; default/SPIR keeps SPIR numbers. AMDGPU
; flat is AS 0, which LLVM prints as a bare `ptr` (addrspace(0) elided).
;
;   class          SPIR   AMDGPU
;   private          0       5
;   global           1       1
;   constant         2       4
;   local            3       3
;   generic          4       0   (bare `ptr`)
;   GlobalDevice     5       1
;   GlobalHost       6       1

; SPV_INTEL_usm_storage_classes: keeps GlobalDevice/GlobalHost forward, so the
; reverse map is exercised for them.
; RUN: llvm-spirv %s --spirv-ext=+SPV_INTEL_usm_storage_classes -o %t.spv

; RUN: llvm-spirv -r %t.spv --spirv-target-triple=amdgcn-amd-amdhsa \
; RUN:   -o - | llvm-dis | FileCheck %s --check-prefix=CHECK-AMDGCN

; RUN: llvm-spirv -r %t.spv \
; RUN:   -o - | llvm-dis | FileCheck %s --check-prefix=CHECK-DEFAULT

; Explicit SPIR triple: default address space map, no error.
; RUN: llvm-spirv -r %t.spv --spirv-target-triple=spir64-unknown-unknown \
; RUN:   -o - | llvm-dis | FileCheck %s --check-prefix=CHECK-DEFAULT

; Untabled non-SPIR triple: no map, errors.
; RUN: not llvm-spirv -r %t.spv --spirv-target-triple=nvptx64-nvidia-cuda \
; RUN:   -o - 2>&1 | FileCheck %s --check-prefix=CHECK-ERR
; CHECK-ERR: No address space map for target triple 'nvptx64-nvidia-cuda'

; Explicit --spirv-addrspace-map beats the triple-derived map: 0:5 remaps only
; Private (0) -> 5.
; RUN: llvm-spirv -r %t.spv --spirv-target-triple=amdgcn-amd-amdhsa \
; RUN:   --spirv-addrspace-map=0:5 \
; RUN:   -o - | llvm-dis | FileCheck %s --check-prefix=CHECK-OVERRIDE

; Explicit --spirv-function-program-addrspace wins over the triple-pinned program
; AS; the derived map still applies.
; RUN: llvm-spirv -r %t.spv --spirv-target-triple=amdgcn-amd-amdhsa \
; RUN:   --spirv-function-program-addrspace=3 \
; RUN:   -o - | llvm-dis | FileCheck %s --check-prefix=CHECK-FPAS

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-G1"
target triple = "spir64-unknown-unknown"

; Datalayout follows the triple: -A is the alloca AS, -P the program
; (function) AS.

; -A5: derived map put Private -> 5. No -P: the triple pins the program AS to
; flat (0), the default, so it is elided.
; CHECK-AMDGCN: target datalayout = "e-m:e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-{{.*}}-n32:64-S32-A5-G1-ni:7:8:9"
; CHECK-AMDGCN: target triple = "amdgcn-amd-amdhsa"

; SPIR map: neither -A nor -P.
; CHECK-DEFAULT: target datalayout = "e-p:64:64:64{{.*}}-G1"
; CHECK-DEFAULT: target triple = "spir64-unknown-unknown"

; -P5: an explicit --spirv-addrspace-map skips the triple's program-AS pin, so
; it falls back to the mapped private AS.
; CHECK-OVERRIDE: target datalayout = "e-m:e-p:64:64{{.*}}-A5-G1-ni:7:8:9-P5"
; CHECK-OVERRIDE: target triple = "amdgcn-amd-amdhsa"

; -P3: explicit --spirv-function-program-addrspace beat the triple pin.
; CHECK-FPAS: target datalayout = "e-m:e-p:64:64{{.*}}-A5-G1-ni:7:8:9-P3"
; CHECK-FPAS: target triple = "amdgcn-amd-amdhsa"

; global: SPIR 1 -> AMDGPU 1 (unchanged), so addrspace(1) every case.
; CHECK-AMDGCN: @gv = {{.*}}addrspace(1){{.*}}global i32
; CHECK-DEFAULT: @gv = {{.*}}addrspace(1){{.*}}global i32
; CHECK-OVERRIDE: @gv = {{.*}}addrspace(1){{.*}}global i32
@gv = addrspace(1) global i32 0, align 4

; global (1), local (3) unchanged. generic: SPIR 4 -> AMDGPU 0 (flat) under
; amdgcn, i.e. bare `ptr`; stays addrspace(4) under default and 0:5.
; CHECK-AMDGCN: define{{.*}} @test_stable_and_generic({{.*}}ptr addrspace(1){{.*}}ptr addrspace(3){{.*}}ptr{{( addrspace\(0\))?}}
; CHECK-DEFAULT: define{{.*}} @test_stable_and_generic({{.*}}ptr addrspace(1){{.*}}ptr addrspace(3){{.*}}ptr addrspace(4)
; CHECK-OVERRIDE: define{{.*}} @test_stable_and_generic({{.*}}ptr addrspace(1){{.*}}ptr addrspace(3){{.*}}ptr addrspace(4)
define spir_kernel void @test_stable_and_generic(ptr addrspace(1) %global_p,
                                                 ptr addrspace(3) %local_p,
                                                 ptr addrspace(4) %generic_p) {
  ret void
}

; constant: SPIR 2 -> AMDGPU 4 under amdgcn; stays addrspace(2) under default and 0:5.
; CHECK-AMDGCN: define{{.*}} @test_constant({{.*}}ptr addrspace(4)
; CHECK-DEFAULT: define{{.*}} @test_constant({{.*}}ptr addrspace(2)
; CHECK-OVERRIDE: define{{.*}} @test_constant({{.*}}ptr addrspace(2)
define spir_kernel void @test_constant(ptr addrspace(2) %const_p) {
  ret void
}

; private: alloca SPIR 0 -> AMDGPU 5 under amdgcn; AS 0 under default;
; addrspace(5) under 0:5.
; CHECK-AMDGCN: define{{.*}} @test_private(
; CHECK-AMDGCN: alloca i32,{{.*}} addrspace(5)
; CHECK-DEFAULT: define{{.*}} @test_private(
; CHECK-DEFAULT: alloca i32, align
; CHECK-OVERRIDE: define{{.*}} @test_private(
; CHECK-OVERRIDE: alloca i32,{{.*}} addrspace(5)
define spir_func i32 @test_private() {
  %x = alloca i32
  %v = load i32, ptr %x
  ret i32 %v
}

; GlobalDevice (5), GlobalHost (6) -> AMDGPU 1 under amdgcn; stay addrspace(5)/(6)
; under default and 0:5 (overrides index 0 only).
; CHECK-AMDGCN: define{{.*}} @test_usm({{.*}}ptr addrspace(1){{.*}}ptr addrspace(1)
; CHECK-DEFAULT: define{{.*}} @test_usm({{.*}}ptr addrspace(5){{.*}}ptr addrspace(6)
; CHECK-OVERRIDE: define{{.*}} @test_usm({{.*}}ptr addrspace(5){{.*}}ptr addrspace(6)
define spir_kernel void @test_usm(ptr addrspace(5) %device_p,
                                  ptr addrspace(6) %host_p) {
  ret void
}
