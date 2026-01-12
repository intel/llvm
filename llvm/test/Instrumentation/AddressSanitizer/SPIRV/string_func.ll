; RUN: opt < %s -passes=asan -asan-instrumentation-with-call-threshold=0 -asan-stack=0 -asan-globals=0 -asan-constructor-kind=none -asan-use-after-return=never -S | FileCheck %s

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64-G1"
target triple = "spir64-unknown-unknown"

; Function Attrs: sanitize_address
define weak_odr dso_local spir_kernel void @test_memset_as0() #0 {
; CHECK-LABEL: define weak_odr dso_local spir_kernel void @test_memset_as0
entry:
  %p.i = alloca [4 x i8], align 4
  call void @llvm.memset.p0.i64(ptr %p.i, i8 1, i64 5, i1 false)
  ; CHECK: [[MEMSET_PTR:%[0-9]+]] = getelementptr i8, ptr %MyAlloca
  ; CHECK: call ptr @__asan_memset_p0(ptr [[MEMSET_PTR]], i32 1, i64 5
  ret void
}

; Function Attrs: sanitize_address
define weak_odr dso_local spir_kernel void @test_memset_as1(ptr addrspace(1) %_arg_ptr) #0 {
; CHECK-LABEL: define weak_odr dso_local spir_kernel void @test_memset_as1
entry:
  call void @llvm.memset.p1.i64(ptr addrspace(1) %_arg_ptr, i8 1, i64 13, i1 false)
  ; CHECK: call ptr addrspace(1) @__asan_memset_p1(ptr addrspace(1) %_arg_ptr, i32 1, i64 13
  ret void
}

; Function Attrs: sanitize_address
define weak_odr dso_local spir_kernel void @test_memset_as3(ptr addrspace(3) %_arg_ptr) #0 {
; CHECK-LABEL: define weak_odr dso_local spir_kernel void @test_memset_as3
entry:
  call void @llvm.memset.p3.i64(ptr addrspace(3) %_arg_ptr, i8 1, i64 13, i1 false)
  ; CHECK: call ptr addrspace(3) @__asan_memset_p3(ptr addrspace(3) %_arg_ptr, i32 1, i64 13
  ret void
}

; Function Attrs: sanitize_address
define weak_odr dso_local spir_kernel void @test_memset_as4(ptr addrspace(4) %_arg_ptr) #0 {
; CHECK-LABEL: define weak_odr dso_local spir_kernel void @test_memset_as4
entry:
  call void @llvm.memset.p4.i64(ptr addrspace(4) %_arg_ptr, i8 1, i64 13, i1 false)
  ; CHECK: call ptr addrspace(4) @__asan_memset_p4(ptr addrspace(4) %_arg_ptr, i32 1, i64 13
  ret void
}

; Function Attrs: sanitize_address
define weak_odr dso_local spir_kernel void @test_memcpy_as0() #0 {
; CHECK-LABEL: define weak_odr dso_local spir_kernel void @test_memcpy_as0
entry:
  %dst = alloca [4 x i8], align 4
  %src = alloca [4 x i8], align 4
  call void @llvm.memcpy.p0.p0.i64(ptr %dst, ptr %src, i64 5, i1 false)
  ; CHECK: [[MEMCPY_DST:%[0-9]+]] = getelementptr i8, ptr %MyAlloca
  ; CHECK-NEXT: [[MEMCPY_SRC:%[0-9]+]] = getelementptr i8, ptr %MyAlloca
  ; CHECK: call ptr @__asan_memcpy_p0_p0(ptr [[MEMCPY_DST]], ptr [[MEMCPY_SRC]], i64 5
  ret void
}

; Function Attrs: sanitize_address
define weak_odr dso_local spir_kernel void @test_memcpy_as1(ptr addrspace(1) %_arg_dst, ptr addrspace(1) %_arg_src) #0 {
; CHECK-LABEL: define weak_odr dso_local spir_kernel void @test_memcpy_as1
entry:
  call void @llvm.memcpy.p1.p1.i64(ptr addrspace(1) %_arg_dst, ptr addrspace(1) %_arg_src, i64 12, i1 false)
  ; CHECK: call ptr addrspace(1) @__asan_memcpy_p1_p1(ptr addrspace(1) %_arg_dst, ptr addrspace(1) %_arg_src, i64 12
  ret void
}

; Function Attrs: sanitize_address
define weak_odr dso_local spir_kernel void @test_memcpy_as3(ptr addrspace(3) %_arg_dst, ptr addrspace(3) %_arg_src) #0 {
; CHECK-LABEL: define weak_odr dso_local spir_kernel void @test_memcpy_as3
entry:
  call void @llvm.memcpy.p3.p3.i64(ptr addrspace(3) %_arg_dst, ptr addrspace(3) %_arg_src, i64 12, i1 false)
  ; CHECK: call ptr addrspace(3) @__asan_memcpy_p3_p3(ptr addrspace(3) %_arg_dst, ptr addrspace(3) %_arg_src, i64 12
  ret void
}

; Function Attrs: sanitize_address
define weak_odr dso_local spir_kernel void @test_memcpy_as4(ptr addrspace(4) %_arg_dst, ptr addrspace(4) %_arg_src) #0 {
; CHECK-LABEL: define weak_odr dso_local spir_kernel void @test_memcpy_as4
entry:
  call void @llvm.memcpy.p4.p4.i64(ptr addrspace(4) %_arg_dst, ptr addrspace(4) %_arg_src, i64 12, i1 false)
  ; CHECK: call ptr addrspace(4) @__asan_memcpy_p4_p4(ptr addrspace(4) %_arg_dst, ptr addrspace(4) %_arg_src, i64 12
  ret void
}

; Function Attrs: sanitize_address
define weak_odr dso_local spir_kernel void @test_memmove_as0() #0 {
; CHECK-LABEL: define weak_odr dso_local spir_kernel void @test_memmove_as0
entry:
  %dst = alloca [4 x i8], align 4
  %src = alloca [4 x i8], align 4
  call void @llvm.memmove.p0.p0.i64(ptr %dst, ptr %src, i64 5, i1 false)
  ; CHECK: [[MEMMOVE_DST:%[0-9]+]] = getelementptr i8, ptr %MyAlloca
  ; CHECK-NEXT: [[MEMMOVE_SRC:%[0-9]+]] = getelementptr i8, ptr %MyAlloca
  ; CHECK: call ptr @__asan_memmove_p0_p0(ptr [[MEMMOVE_DST]], ptr [[MEMMOVE_SRC]], i64 5
  ret void
}

; Function Attrs: sanitize_address
define weak_odr dso_local spir_kernel void @test_memmove_as1(ptr addrspace(1) %_arg_dst, ptr addrspace(1) %_arg_src) #0 {
; CHECK-LABEL: define weak_odr dso_local spir_kernel void @test_memmove_as1
entry:
  call void @llvm.memmove.p1.p1.i64(ptr addrspace(1) %_arg_dst, ptr addrspace(1) %_arg_src, i64 12, i1 false)
  ; CHECK: call ptr addrspace(1) @__asan_memmove_p1_p1(ptr addrspace(1) %_arg_dst, ptr addrspace(1) %_arg_src, i64 12
  ret void
}

; Function Attrs: sanitize_address
define weak_odr dso_local spir_kernel void @test_memmove_as3(ptr addrspace(3) %_arg_dst, ptr addrspace(3) %_arg_src) #0 {
; CHECK-LABEL: define weak_odr dso_local spir_kernel void @test_memmove_as3
entry:
  call void @llvm.memmove.p3.p3.i64(ptr addrspace(3) %_arg_dst, ptr addrspace(3) %_arg_src, i64 12, i1 false)
  ; CHECK: call ptr addrspace(3) @__asan_memmove_p3_p3(ptr addrspace(3) %_arg_dst, ptr addrspace(3) %_arg_src, i64 12
  ret void
}

; Function Attrs: sanitize_address
define weak_odr dso_local spir_kernel void @test_memmove_as4(ptr addrspace(4) %_arg_dst, ptr addrspace(4) %_arg_src) #0 {
; CHECK-LABEL: define weak_odr dso_local spir_kernel void @test_memmove_as4
entry:
  call void @llvm.memmove.p4.p4.i64(ptr addrspace(4) %_arg_dst, ptr addrspace(4) %_arg_src, i64 12, i1 false)
  ; CHECK: call ptr addrspace(4) @__asan_memmove_p4_p4(ptr addrspace(4) %_arg_dst, ptr addrspace(4) %_arg_src, i64 12
  ret void
}

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: write)
declare void @llvm.memset.p0.i64(ptr writeonly captures(none), i8, i64, i1 immarg) #1

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: write)
declare void @llvm.memset.p1.i64(ptr addrspace(1) writeonly captures(none), i8, i64, i1 immarg) #1

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: write)
declare void @llvm.memset.p3.i64(ptr addrspace(3) writeonly captures(none), i8, i64, i1 immarg) #1

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: write)
declare void @llvm.memset.p4.i64(ptr addrspace(4) writeonly captures(none), i8, i64, i1 immarg) #1

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #2

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p1.p1.i64(ptr addrspace(1) noalias writeonly captures(none), ptr addrspace(1) noalias readonly captures(none), i64, i1 immarg) #2

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p3.p3.i64(ptr addrspace(3) noalias writeonly captures(none), ptr addrspace(3) noalias readonly captures(none), i64, i1 immarg) #2

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p4.p4.i64(ptr addrspace(4) noalias writeonly captures(none), ptr addrspace(4) noalias readonly captures(none), i64, i1 immarg) #2

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memmove.p0.p0.i64(ptr writeonly captures(none), ptr readonly captures(none), i64, i1 immarg) #2

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memmove.p1.p1.i64(ptr addrspace(1) writeonly captures(none), ptr addrspace(1) readonly captures(none), i64, i1 immarg) #2

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memmove.p3.p3.i64(ptr addrspace(3) writeonly captures(none), ptr addrspace(3) readonly captures(none), i64, i1 immarg) #2

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memmove.p4.p4.i64(ptr addrspace(4) writeonly captures(none), ptr addrspace(4) readonly captures(none), i64, i1 immarg) #2

attributes #0 = { sanitize_address }
attributes #1 = { nocallback nofree nounwind willreturn memory(argmem: write) }
attributes #2 = { nocallback nofree nounwind willreturn memory(argmem: readwrite) }
