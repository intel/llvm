;;===----------------------------------------------------------------------===//
;;
;; Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
;; See https://llvm.org/LICENSE.txt for license information.
;; SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
;;
;;===----------------------------------------------------------------------===//

#if __clang_major__ >= 7
target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7"
#else
target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7"
#endif

@.assert_fmt = private unnamed_addr constant [79 x i8] c"%s:%u: %s: global id: [%u,%u,%u], local id: [%u,%u,%u] Assertion `%s` failed.\0A\00", align 1

declare void @llvm.trap() cold noreturn nounwind

declare i64 @__ockl_fprintf_stderr_begin() local_unnamed_addr
declare i64 @__ockl_fprintf_append_string_n(i64, i8* readonly, i64, i32) local_unnamed_addr
declare i64 @__ockl_fprintf_append_args(i64, i32, i64, i64, i64, i64, i64, i64, i64, i32) local_unnamed_addr

declare dso_local i64 @_Z28__spirv_GlobalInvocationId_xv() local_unnamed_addr
declare dso_local i64 @_Z28__spirv_GlobalInvocationId_yv() local_unnamed_addr
declare dso_local i64 @_Z28__spirv_GlobalInvocationId_zv() local_unnamed_addr

declare dso_local i64 @_Z27__spirv_LocalInvocationId_xv() local_unnamed_addr
declare dso_local i64 @_Z27__spirv_LocalInvocationId_yv() local_unnamed_addr
declare dso_local i64 @_Z27__spirv_LocalInvocationId_zv() local_unnamed_addr

define dso_local hidden noundef i64 @__strlen_assert(i8* noundef %str) local_unnamed_addr {
entry:
  br label %while.cond

while.cond:
  %tmp.0 = phi i8* [ %str, %entry ], [ %incdec.ptr, %while.cond ]
  %incdec.ptr = getelementptr inbounds i8, i8* %tmp.0, i64 1
  %0 = load i8, i8* %tmp.0, align 1
  %tobool.not = icmp eq i8 %0, 0
  br i1 %tobool.not, label %while.end, label %while.cond

while.end:
  %sub.ptr.lhs.cast = ptrtoint i8* %incdec.ptr to i64
  %sub.ptr.rhs.cast = ptrtoint i8* %str to i64
  %sub.ptr.sub = sub i64 %sub.ptr.lhs.cast, %sub.ptr.rhs.cast
  ret i64 %sub.ptr.sub
}

define hidden void @__assert_fail(i8* %assertion, i8* %file, i32 %line, i8* %function) nounwind alwaysinline {
entry:
  %msg = call i64 @__ockl_fprintf_stderr_begin()
  %msg.1 = call i64 @__ockl_fprintf_append_string_n(i64 %msg, i8* readonly getelementptr inbounds ([79 x i8], [79 x i8]* @.assert_fmt, i64 0, i64 0), i64 79, i32 0)
  %len.file = call i64 @__strlen_assert(i8* %file)
  %msg.2 = call i64 @__ockl_fprintf_append_string_n(i64 %msg.1, i8* readonly %file, i64 %len.file, i32 0)
  %line.i64 = sext i32 %line to i64
  %msg.3 = call i64 @__ockl_fprintf_append_args(i64 %msg.2, i32 1, i64 %line.i64, i64 0, i64 0, i64 0, i64 0, i64 0, i64 0, i32 0)
  %len.func = call i64 @__strlen_assert(i8* %function)
  %msg.4 = call i64 @__ockl_fprintf_append_string_n(i64 %msg.3, i8* readonly %function, i64 %len.func, i32 0)
  %gidx = tail call i64 @_Z28__spirv_GlobalInvocationId_xv()
  %gidy = tail call i64 @_Z28__spirv_GlobalInvocationId_yv()
  %gidz = tail call i64 @_Z28__spirv_GlobalInvocationId_zv()
  %lidx = tail call i64 @_Z27__spirv_LocalInvocationId_xv()
  %lidy = tail call i64 @_Z27__spirv_LocalInvocationId_yv()
  %lidz = tail call i64 @_Z27__spirv_LocalInvocationId_zv()
  %msg.5 = call i64 @__ockl_fprintf_append_args(i64 %msg.4, i32 6, i64 %gidx, i64 %gidy, i64 %gidz, i64 %lidx, i64 %lidy, i64 %lidz, i64 0, i32 0)
  %len.assertion = call i64 @__strlen_assert(i8* %assertion)
  %msg.6 = call i64 @__ockl_fprintf_append_string_n(i64 %msg.4, i8* readonly %assertion, i64 %len.assertion, i32 1)
  tail call void @llvm.trap()
  unreachable
}
