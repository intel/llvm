; REQUIRES: pass-plugin
; UNSUPPORTED: target={{.*windows.*}}

; RUN: opt %load_spirv_lib -passes=spirv-to-ocl20 %s -S -o - | FileCheck %s

target triple = "spir64-unknown-unknown"

declare spir_func i32 @_Z18__spirv_ocl_printfPU3AS2Kcz(ptr addrspace(2), ...)

define spir_func void @__asan_set_shadow_dynamic_local() {
; CHECK: call spir_func i32 (ptr addrspace(2), ...) @printf(ptr addrspace(2) null, i32 0, i32 0)
  %call = call spir_func i32 (ptr addrspace(2), ...) @_Z18__spirv_ocl_printfPU3AS2Kcz(ptr addrspace(2) null, i32 0, i32 0)
  ret void
}
