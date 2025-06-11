; RUN: opt < %s -passes='function(tsan),module(tsan-module)' -tsan-instrument-func-entry-exit=0 -tsan-instrument-memintrinsics=0 -S | FileCheck %s
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64-G1"
target triple = "spir64-unknown-unknown"

%"class.sycl::_V1::range" = type { %"class.sycl::_V1::detail::array" }
%"class.sycl::_V1::detail::array" = type { [1 x i64] }

%struct.Dimensions = type { i32, i32, i32, i32, i32, i32 }

define spir_kernel void @test(i32 %val) {
entry:
  %agg.tmp = alloca %"class.sycl::_V1::range", align 8
  %cmp = icmp eq i32 %val, 1
  br i1 %cmp, label %for.body.preheader, label %exit

for.body.preheader:                     ; preds = %entry
  br label %for.body

for.body:                               ; preds = %for.body.preheader
  %device-byval-temp.ascast234298 = alloca %struct.Dimensions, i32 0, align 8, addrspace(4)
  br label %exit

exit:
; CHECK: [[REG1:%[0-9]+]] = ptrtoint ptr %agg.tmp to i64
; CHECK-NEXT: call void @__tsan_cleanup_private(i64 [[REG1]], i64 8)
; CHECK-NOT: ptrtoint ptr %device-byval-temp.ascast234298 to i64
  ret void
}
