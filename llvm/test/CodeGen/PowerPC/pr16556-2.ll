; RUN: llc -verify-machineinstrs < %s

; This test formerly failed because of wrong custom lowering for
; fptosi of ppc_fp128.

target datalayout = "E-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f16:16:16-f32:32:32-f64:64:64-f128:64:128-v64:64:64-v128:128:128-a0:0:64-n32"
target triple = "powerpc-unknown-linux-gnu"

%core.time.TickDuration = type { i64 }

@_D4core4time12TickDuration11ticksPerSecyl = global i64 0
@.str5 = internal unnamed_addr constant [40 x i8] c"..\5Cldc\5Cruntime\5Cdruntime\5Csrc\5Ccore\5Ctime.d\00"
@.str83 = internal constant [10 x i8] c"null this\00"
@.modulefilename = internal constant { i32, ptr } { i32 39, ptr @.str5 }

declare ptr @_d_assert_msg({ i32, ptr }, { i32, ptr }, i32)


define weak_odr fastcc i64 @_D4core4time12TickDuration30__T2toVAyaa7_7365636f6e6473TlZ2toMxFNaNbNfZl(ptr %.this_arg) {
entry:
  %unitsPerSec = alloca i64, align 8
  %tmp = icmp ne ptr %.this_arg, null
  br i1 %tmp, label %noassert, label %assert

assert:                                           ; preds = %entry
  %tmp1 = load { i32, ptr }, ptr @.modulefilename
  %0 = call ptr @_d_assert_msg({ i32, ptr } { i32 9, ptr @.str83 }, { i32, ptr } %tmp1, i32 1586)
  unreachable

noassert:                                         ; preds = %entry
  %tmp3 = load i64, ptr %.this_arg
  %tmp4 = sitofp i64 %tmp3 to ppc_fp128
  %tmp5 = load i64, ptr @_D4core4time12TickDuration11ticksPerSecyl
  %tmp6 = sitofp i64 %tmp5 to ppc_fp128
  %tmp7 = fdiv ppc_fp128 %tmp6, 0xM80000000000000000000000000000000
  %tmp8 = fdiv ppc_fp128 %tmp4, %tmp7
  %tmp9 = fptosi ppc_fp128 %tmp8 to i64
  ret i64 %tmp9
}

