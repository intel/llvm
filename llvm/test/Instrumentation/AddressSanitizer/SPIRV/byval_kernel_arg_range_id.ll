; RUN: opt < %s -passes=asan -asan-instrumentation-with-call-threshold=0 -asan-stack=0 -asan-globals=0 -asan-constructor-kind=none -asan-mapping-scale=4 -asan-spir-privates=1 -asan-use-after-return=never -S | FileCheck %s

; If every use of a byval kernel argument is a read at a constant offset that
; stays inside the argument, the argument can't be accessed out of bounds.
; The sycl::range and sycl::id members of accessors are used this way. ASan
; doesn't instrument those reads and doesn't copy the argument into its
; redzoned private frame. Any other use keeps the copy, so out-of-bounds
; accesses through the argument are still reported.

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

%"class.sycl::_V1::range" = type { %"class.sycl::_V1::detail::array" }
%"class.sycl::_V1::detail::array" = type { [1 x i64] }
%"class.sycl::_V1::id" = type { %"class.sycl::_V1::detail::array" }
%"class.sycl::_V1::range.1" = type { %"class.sycl::_V1::detail::array.1" }
%"class.sycl::_V1::detail::array.1" = type { [2 x i64] }
%struct.Input = type { [4 x i32] }
%struct.S = type { i64, [2 x i64] }

declare spir_func void @use(ptr)
declare void @llvm.memcpy.p4.p4.i64(ptr addrspace(4), ptr addrspace(4), i64, i1)
declare void @llvm.memcpy.p1.p0.i64(ptr addrspace(1), ptr, i64, i1)

; ASan re-creates kernels with an extra launch-info argument at the end of the
; module, so the non-kernel function comes first.

; Byval arguments of non-kernel functions are always copied.
define spir_func void @func_range(ptr byval(%"class.sycl::_V1::range") align 8 %range) #0 {
; CHECK-LABEL: define spir_func void @func_range
; CHECK: [[COPY:%[0-9]+]] = getelementptr i8, ptr %MyAlloca
; CHECK: call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[COPY]], ptr align 8 %range, i64 8, i1 false)
entry:
  %0 = load i64, ptr %range, align 8
  ret void
}

; Direct and constant-offset in-bounds reads: no copy, no checks.
define spir_kernel void @kernel_inbounds_reads(ptr addrspace(1) %acc, ptr byval(%"class.sycl::_V1::range.1") align 8 %range2d, ptr byval(%"class.sycl::_V1::id") align 8 %offset) #0 {
; CHECK-LABEL: define spir_kernel void @kernel_inbounds_reads
; CHECK-NOT: call void @llvm.memcpy
; CHECK-NOT: call void @__asan_set_shadow_private
; CHECK-NOT: call void @__asan_load
; CHECK: %0 = load i64, ptr %offset
; CHECK: %elem1 = getelementptr inbounds i8, ptr %range2d, i64 8
; CHECK-NEXT: %1 = load i64, ptr %elem1
; CHECK-NOT: call void @__asan_load
; CHECK: call void @__asan_store4_as1
; CHECK: ret void
entry:
  %0 = load i64, ptr %offset, align 8
  %elem1 = getelementptr inbounds i8, ptr %range2d, i64 8
  %1 = load i64, ptr %elem1, align 8
  %2 = add i64 %0, %1
  %arrayidx = getelementptr inbounds i32, ptr addrspace(1) %acc, i64 %2
  store i32 0, ptr addrspace(1) %arrayidx, align 4
  ret void
}

; -O0 copies an accessor's range through a generic pointer. The memcpy reads
; the whole argument, so the argument isn't copied, and only the memcpy's
; destination is checked.
define spir_kernel void @kernel_o0_generic_memcpy(ptr byval(%"class.sycl::_V1::range") align 8 %_arg_AccessRange) #0 {
; CHECK-LABEL: define spir_kernel void @kernel_o0_generic_memcpy
; CHECK-NOT: __asan_memcpy
; CHECK-NOT: ptr align 8 %_arg_AccessRange,
; CHECK: %_arg_AccessRange.ascast = addrspacecast ptr %_arg_AccessRange to ptr addrspace(4)
; CHECK-NEXT: [[DST:%[0-9]+]] = ptrtoint ptr addrspace(4) %tmp.ascast to i64
; CHECK-NEXT: call void @__asan_storeN_as4(i64 [[DST]], i64 8,
; CHECK-NEXT: call void @llvm.memcpy.p4.p4.i64(ptr addrspace(4) align 8 %tmp.ascast, ptr addrspace(4) align 8 %_arg_AccessRange.ascast, i64 8, i1 false)
; CHECK-NOT: __asan_memcpy
; CHECK: ret void
entry:
  %tmp = alloca %"class.sycl::_V1::range", align 8
  %tmp.ascast = addrspacecast ptr %tmp to ptr addrspace(4)
  %_arg_AccessRange.ascast = addrspacecast ptr %_arg_AccessRange to ptr addrspace(4)
  call void @llvm.memcpy.p4.p4.i64(ptr addrspace(4) align 8 %tmp.ascast, ptr addrspace(4) align 8 %_arg_AccessRange.ascast, i64 8, i1 false)
  ret void
}

; -O0 Out[I] = R: the argument isn't copied, but the out-of-bounds-prone
; generic destination is checked.
define spir_kernel void @kernel_o0_copy_out(ptr addrspace(4) %out, ptr byval(%"class.sycl::_V1::range.1") align 8 %R, i64 %i) #0 {
; CHECK-LABEL: define spir_kernel void @kernel_o0_copy_out
; CHECK-NOT: ptr align 8 %R,
; CHECK: [[DST:%[0-9]+]] = ptrtoint ptr addrspace(4) %dst to i64
; CHECK-NEXT: call void @__asan_storeN_as4(i64 [[DST]], i64 16,
; CHECK-NEXT: call void @llvm.memcpy.p4.p4.i64(ptr addrspace(4) align 8 %dst, ptr addrspace(4) align 8 %R.ascast, i64 16, i1 false)
entry:
  %R.ascast = addrspacecast ptr %R to ptr addrspace(4)
  %dst = getelementptr inbounds %"class.sycl::_V1::range.1", ptr addrspace(4) %out, i64 %i
  call void @llvm.memcpy.p4.p4.i64(ptr addrspace(4) align 8 %dst, ptr addrspace(4) align 8 %R.ascast, i64 16, i1 false)
  ret void
}

; Copying a member of the argument to global memory: the argument isn't
; copied, but the global destination is checked.
define spir_kernel void @kernel_member_to_global(ptr addrspace(1) %out, ptr byval(%struct.S) align 8 %s, i64 %i) #0 {
; CHECK-LABEL: define spir_kernel void @kernel_member_to_global
; CHECK-NOT: ptr align 8 %s,
; CHECK: [[DST:%[0-9]+]] = ptrtoint ptr addrspace(1) %dst to i64
; CHECK-NEXT: call void @__asan_storeN_as1(i64 [[DST]], i64 16,
; CHECK-NEXT: call void @llvm.memcpy.p1.p0.i64(ptr addrspace(1) align 8 %dst, ptr align 8 %src, i64 16, i1 false)
entry:
  %dst = getelementptr inbounds i8, ptr addrspace(1) %out, i64 %i
  %src = getelementptr inbounds i8, ptr %s, i64 8
  call void @llvm.memcpy.p1.p0.i64(ptr addrspace(1) align 8 %dst, ptr align 8 %src, i64 16, i1 false)
  ret void
}

; A load larger than the argument keeps the copy, and the load from the copy is
; checked.
define spir_kernel void @kernel_oversized_load(ptr addrspace(1) %out, ptr byval(%"class.sycl::_V1::range") align 8 %range) #0 {
; CHECK-LABEL: define spir_kernel void @kernel_oversized_load
; CHECK: [[COPY:%[0-9]+]] = getelementptr i8, ptr %MyAlloca
; CHECK: call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[COPY]], ptr align 8 %range, i64 8, i1 false)
; CHECK: [[ADDR:%[0-9]+]] = ptrtoint ptr [[COPY]] to i64
; CHECK-NEXT: call void @__asan_loadN_as0(i64 [[ADDR]], i64 16,
; CHECK-NEXT: %v = load i128, ptr [[COPY]]
entry:
  %v = load i128, ptr %range, align 8
  store i128 %v, ptr addrspace(1) %out, align 8
  ret void
}

; A runtime index (e.g. a user-captured range indexed with R[d]) keeps the
; redzoned copy, and the access is checked against the copy.
define spir_kernel void @kernel_runtime_index(ptr addrspace(1) %out, ptr byval(%"class.sycl::_V1::range.1") align 8 %R, i64 %d) #0 {
; CHECK-LABEL: define spir_kernel void @kernel_runtime_index
; CHECK: [[COPY:%[0-9]+]] = getelementptr i8, ptr %MyAlloca
; CHECK: call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[COPY]], ptr align 8 %R, i64 16, i1 false)
; CHECK: %elem = getelementptr inbounds [2 x i64], ptr [[COPY]], i64 0, i64 %d
; CHECK-NEXT: [[ADDR:%[0-9]+]] = ptrtoint ptr %elem to i64
; CHECK-NEXT: call void @__asan_load8_as0(i64 [[ADDR]]
entry:
  %elem = getelementptr inbounds [2 x i64], ptr %R, i64 0, i64 %d
  %v = load i64, ptr %elem, align 8
  store i64 %v, ptr addrspace(1) %out, align 8
  ret void
}

; A constant offset past the end of the argument keeps the copy, so the
; out-of-bounds read hits a redzone.
define spir_kernel void @kernel_constant_oob(ptr addrspace(1) %out, ptr byval(%"class.sycl::_V1::range") align 8 %range) #0 {
; CHECK-LABEL: define spir_kernel void @kernel_constant_oob
; CHECK: [[COPY:%[0-9]+]] = getelementptr i8, ptr %MyAlloca
; CHECK: call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[COPY]], ptr align 8 %range, i64 8, i1 false)
; CHECK: %past_end = getelementptr inbounds i8, ptr [[COPY]], i64 8
; CHECK-NEXT: [[ADDR:%[0-9]+]] = ptrtoint ptr %past_end to i64
; CHECK-NEXT: call void @__asan_load8_as0(i64 [[ADDR]]
entry:
  %past_end = getelementptr inbounds i8, ptr %range, i64 8
  %v = load i64, ptr %past_end, align 8
  store i64 %v, ptr addrspace(1) %out, align 8
  ret void
}

; Passing the argument to a callee keeps the copy, and the callee gets the
; copy.
define spir_kernel void @kernel_escape_to_callee(ptr byval(%"class.sycl::_V1::range") align 8 %range) #0 {
; CHECK-LABEL: define spir_kernel void @kernel_escape_to_callee
; CHECK: [[COPY:%[0-9]+]] = getelementptr i8, ptr %MyAlloca
; CHECK: call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[COPY]], ptr align 8 %range, i64 8, i1 false)
; CHECK: call spir_func void @use(ptr [[COPY]])
entry:
  call spir_func void @use(ptr %range)
  ret void
}

; Writing to the argument keeps the copy, and the store to the copy is checked.
define spir_kernel void @kernel_store_to_arg(ptr byval(%"class.sycl::_V1::range") align 8 %range) #0 {
; CHECK-LABEL: define spir_kernel void @kernel_store_to_arg
; CHECK: [[COPY:%[0-9]+]] = getelementptr i8, ptr %MyAlloca
; CHECK: call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[COPY]], ptr align 8 %range, i64 8, i1 false)
; CHECK: [[ADDR:%[0-9]+]] = ptrtoint ptr [[COPY]] to i64
; CHECK-NEXT: call void @__asan_store8_as0(i64 [[ADDR]]
; CHECK-NEXT: store i64 0, ptr [[COPY]]
entry:
  store i64 0, ptr %range, align 8
  ret void
}

; A kernel that mixes an in-bounds-read range argument with a dynamically
; indexed user struct and a local array. Only the struct is copied; the struct
; and the local array still get redzones and checks.
define spir_kernel void @kernel_mixed(ptr addrspace(1) %out, ptr byval(%"class.sycl::_V1::range") align 8 %range, ptr byval(%struct.Input) align 4 %input, i64 %i) #0 {
; CHECK-LABEL: define spir_kernel void @kernel_mixed
; CHECK-NOT: ptr align 8 %range,
; CHECK: call void @llvm.memcpy.p0.p0.i64(ptr align 4 [[COPY:%[0-9]+]], ptr align 4 %input, i64 16, i1 false)
; CHECK-NOT: ptr align 8 %range,
; CHECK: %r = load i64, ptr %range
; CHECK: %local.elem = getelementptr inbounds [4 x i32], ptr [[LOCAL:%[0-9]+]], i64 0, i64 %i
; CHECK: call void @__asan_store4_as0
; CHECK: %input.elem = getelementptr inbounds [4 x i32], ptr [[COPY]], i64 0, i64 %i
; CHECK: call void @__asan_load4_as0
; CHECK: ret void
entry:
  %local = alloca [4 x i32], align 4
  %r = load i64, ptr %range, align 8
  %local.elem = getelementptr inbounds [4 x i32], ptr %local, i64 0, i64 %i
  store i32 1, ptr %local.elem, align 4
  %input.elem = getelementptr inbounds [4 x i32], ptr %input, i64 0, i64 %i
  %v = load i32, ptr %input.elem, align 4
  %v64 = zext i32 %v to i64
  %sum = add i64 %r, %v64
  store i64 %sum, ptr addrspace(1) %out, align 8
  ret void
}

attributes #0 = { sanitize_address }
