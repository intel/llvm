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

define i32 @__clc__sampler_extract_normalized_coords_prop(i32 %sampl) nounwind alwaysinline {
entry:
  %0 = and i32 %sampl, 1
  ret i32 %0
}

define i32 @__clc__sampler_extract_filter_mode_prop(i32 %sampl) nounwind alwaysinline {
entry:
  %0 = lshr i32 %sampl, 1
  %1 = and i32 %0, 1
  ret i32 %1
}

define i32 @__clc__sampler_extract_addressing_mode_prop(i32 %sampl) nounwind alwaysinline {
entry:
  %0 = lshr i32 %sampl, 2
  ret i32 %0
}