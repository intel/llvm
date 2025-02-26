; RUN: opt < %s -passes='function(tsan),module(tsan-module)' -tsan-instrument-func-entry-exit=0 -tsan-instrument-memintrinsics=0 -S | FileCheck %s
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64-G1"
target triple = "spir64-unknown-unknown"

; Function Attrs: sanitize_thread
define linkonce_odr dso_local spir_func void @_Z3fooPc(ptr addrspace(4) %array) #0 {
; CHECK-LABEL: void @_Z3fooPc
entry:
  %array.addr = alloca ptr addrspace(4), align 8
  %array.addr.ascast = addrspacecast ptr %array.addr to ptr addrspace(4)
  store ptr addrspace(4) %array, ptr addrspace(4) %array.addr.ascast, align 8
  %0 = load ptr addrspace(4), ptr addrspace(4) %array.addr.ascast, align 8
  %arrayidx = getelementptr inbounds i8, ptr addrspace(4) %0, i64 0
  %1 = load i8, ptr addrspace(4) %arrayidx, align 1
  %inc = add i8 %1, 1
  ; CHECK: ptrtoint ptr addrspace(4) %arrayidx to i64
  ; CHECK: call void @__tsan_write1
  store i8 %inc, ptr addrspace(4) %arrayidx, align 1
  ret void
}

attributes #0 = { sanitize_thread }
