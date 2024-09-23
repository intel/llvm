; REQUIRES: pass-plugin
; UNSUPPORTED: target={{.*windows.*}}

; RUN: opt %load_spirv_lib -passes=spirv-to-ocl20 %s -S -o - | FileCheck %s

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

define spir_func ptr @_Z41__SYCL_GenericCastToPtrExplicit_ToPrivateIN4sycl3_V16marrayIdLm17EEEEPU3AS0T_Pv(ptr addrspace(4) %ptr) {
entry:
; CHECK: tail call spir_func noundef ptr @__to_private(ptr addrspace(4) noundef %ptr)

  %call = tail call spir_func noundef ptr @_Z42__spirv_GenericCastToPtrExplicit_ToPrivatePvi(ptr addrspace(4) noundef %ptr, i32 noundef 0)
  ret ptr null
}

declare spir_func ptr @_Z42__spirv_GenericCastToPtrExplicit_ToPrivatePvi(ptr addrspace(4), i32)
