; RUN: opt < %s -passes='function(tsan),module(tsan-module)' -tsan-instrument-func-entry-exit=0 -tsan-instrument-memintrinsics=0 -S | FileCheck %s
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64-G1"
target triple = "spir64-unknown-unknown"

; Function Attrs: sanitize_thread
define linkonce_odr dso_local spir_func void @write_1_byte(ptr addrspace(4) %a) #0 {
; CHECK-LABEL: void @write_1_byte
entry:
  %tmp1 = load i8, ptr addrspace(4) %a, align 1
  %inc = add i8 %tmp1, 1
  ; CHECK: ptrtoint ptr addrspace(4) %a to i64
  ; CHECK: call void @__tsan_write1
  store i8 %inc, ptr addrspace(4) %a, align 1
  ret void
}

; Function Attrs: sanitize_thread
define linkonce_odr dso_local spir_func void @write_2_bytes(ptr addrspace(4) %a) #0 {
; CHECK-LABEL: void @write_2_bytes
entry:
  %tmp1 = load i16, ptr addrspace(4) %a, align 2
  %inc = add i16 %tmp1, 1
  ; CHECK: ptrtoint ptr addrspace(4) %a to i64
  ; CHECK: call void @__tsan_write2
  store i16 %inc, ptr addrspace(4) %a, align 2
  ret void
}

; Function Attrs: sanitize_thread
define linkonce_odr dso_local spir_func void @write_4_bytes(ptr addrspace(4) %a) #0 {
; CHECK-LABEL: void @write_4_bytes
entry:
  %tmp1 = load i32, ptr addrspace(4) %a, align 4
  %inc = add i32 %tmp1, 1
  ; CHECK: ptrtoint ptr addrspace(4) %a to i64
  ; CHECK: call void @__tsan_write4
  store i32 %inc, ptr addrspace(4) %a, align 4
  ret void
}

; Function Attrs: sanitize_thread
define linkonce_odr dso_local spir_func void @write_8_bytes(ptr addrspace(4) %a) #0 {
; CHECK-LABEL: void @write_8_bytes
entry:
  %tmp1 = load i64, ptr addrspace(4) %a, align 8
  %inc = add i64 %tmp1, 1
  ; CHECK: ptrtoint ptr addrspace(4) %a to i64
  ; CHECK: call void @__tsan_write8
  store i64 %inc, ptr addrspace(4) %a, align 8
  ret void
}

define linkonce_odr dso_local spir_func i8 @read_1_byte(ptr addrspace(4) %a) #0 {
; CHECK-LABEL: i8 @read_1_byte
entry:
  %tmp1 = load i8, ptr addrspace(4) %a, align 1
  ; CHECK: ptrtoint ptr addrspace(4) %a to i64
  ; CHECK: call void @__tsan_read1
  ret i8 %tmp1
}

define linkonce_odr dso_local spir_func i16 @read_2_bytes(ptr addrspace(4) %a) #0 {
; CHECK-LABEL: i16 @read_2_bytes
entry:
  %tmp1 = load i16, ptr addrspace(4) %a, align 2
  ; CHECK: ptrtoint ptr addrspace(4) %a to i64
  ; CHECK: call void @__tsan_read2
  ret i16 %tmp1
}

define linkonce_odr dso_local spir_func i32 @read_4_bytes(ptr addrspace(4) %a) #0 {
; CHECK-LABEL: i32 @read_4_bytes
entry:
  %tmp1 = load i32, ptr addrspace(4) %a, align 4
  ; CHECK: ptrtoint ptr addrspace(4) %a to i64
  ; CHECK: call void @__tsan_read4
  ret i32 %tmp1
}

define linkonce_odr dso_local spir_func i64 @read_8_bytes(ptr addrspace(4) %a) #0 {
; CHECK-LABEL: i64 @read_8_bytes
entry:
  %tmp1 = load i64, ptr addrspace(4) %a, align 8
  ; CHECK: ptrtoint ptr addrspace(4) %a to i64
  ; CHECK: call void @__tsan_read8
  ret i64 %tmp1
}

attributes #0 = { sanitize_thread }
