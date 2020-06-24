#if __clang_major__ >= 7
target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5"
#else
target datalayout = "e-p:32:32-p1:64:64-p2:64:64-p3:32:32-p4:64:64-p5:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64"
#endif

define i64 @__clc__sampled_image_unpack_image(i64 %img, i32 %sampl) nounwind alwaysinline {
entry:
  ret i64 %img
}

define i32 @__clc__sampled_image_unpack_sampler(i64 %img, i32 %sampl) nounwind alwaysinline {
entry:
  ret i32 %sampl
}

define {i64, i32} @__clc__sampled_image_pack(i64 %img, i32 %sampl) nounwind alwaysinline {
entry:
  %0 = insertvalue {i64, i32} undef, i64 %img, 0
  %1 = insertvalue {i64, i32} %0, i32 %sampl, 1
  ret {i64, i32} %1
}