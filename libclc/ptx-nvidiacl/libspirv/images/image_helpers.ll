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

// We need wrappers around intrinsics as intrinsics are not allowed to return named structs
declare {i16,i16,i16,i16} @llvm.nvvm.suld.1d.v4i16.trap(i64, i32)
define {i16,i16,i16,i16} @__clc_llvm_nvvm_suld_1d_v4i16_trap_s(i64 %img, i32 %x) nounwind alwaysinline {
entry:
  %0 = tail call {i16,i16,i16,i16} @llvm.nvvm.suld.1d.v4i16.trap(i64 %img, i32 %x);
  ret {i16,i16,i16,i16} %0
}

declare {i16,i16,i16,i16} @llvm.nvvm.suld.2d.v4i16.trap(i64, i32, i32)
define {i16,i16,i16,i16} @__clc_llvm_nvvm_suld_2d_v4i16_trap(i64 %img, i32 %x, i32 %y) nounwind alwaysinline {
entry:
  %0 = tail call {i16,i16,i16,i16} @llvm.nvvm.suld.2d.v4i16.trap(i64 %img, i32 %x, i32 %y);
  ret {i16,i16,i16,i16} %0
}

declare {i16,i16,i16,i16} @llvm.nvvm.suld.3d.v4i16.trap(i64, i32, i32, i32)
define {i16,i16,i16,i16} @__clc_llvm_nvvm_suld_3d_v4i16_trap(i64 %img, i32 %x, i32 %y, i32 %z) nounwind alwaysinline {
entry:
  %0 = tail call {i16,i16,i16,i16} @llvm.nvvm.suld.3d.v4i16.trap(i64 %img, i32 %x, i32 %y, i32 %z);
  ret {i16,i16,i16,i16} %0
}

declare {i16,i16,i16,i16} @llvm.nvvm.suld.1d.v4i16.clamp(i64, i32)
define {i16,i16,i16,i16} @__clc_llvm_nvvm_suld_1d_v4i16_clamp(i64 %img, i32 %x) nounwind alwaysinline {
entry:
  %0 = tail call {i16,i16,i16,i16} @llvm.nvvm.suld.1d.v4i16.clamp(i64 %img, i32 %x);
  ret {i16,i16,i16,i16} %0
}

declare {i16,i16,i16,i16} @llvm.nvvm.suld.2d.v4i16.clamp(i64, i32, i32)
define {i16,i16,i16,i16} @__clc_llvm_nvvm_suld_2d_v4i16_clamp(i64 %img, i32 %x, i32 %y) nounwind alwaysinline {
entry:
  %0 = tail call {i16,i16,i16,i16} @llvm.nvvm.suld.2d.v4i16.clamp(i64 %img, i32 %x, i32 %y);
  ret {i16,i16,i16,i16} %0
}

declare {i16,i16,i16,i16} @llvm.nvvm.suld.3d.v4i16.clamp(i64, i32, i32, i32)
define {i16,i16,i16,i16} @__clc_llvm_nvvm_suld_3d_v4i16_clamp(i64 %img, i32 %x, i32 %y, i32 %z) nounwind alwaysinline {
entry:
  %0 = tail call {i16,i16,i16,i16} @llvm.nvvm.suld.3d.v4i16.clamp(i64 %img, i32 %x, i32 %y, i32 %z);
  ret {i16,i16,i16,i16} %0
}

declare {i16,i16,i16,i16} @llvm.nvvm.suld.1d.v4i16.zero(i64, i32)
define {i16,i16,i16,i16} @__clc_llvm_nvvm_suld_1d_v4i16_zero(i64 %img, i32 %x) nounwind alwaysinline {
entry:
  %0 = tail call {i16,i16,i16,i16} @llvm.nvvm.suld.1d.v4i16.zero(i64 %img, i32 %x);
  ret {i16,i16,i16,i16} %0
}

declare {i16,i16,i16,i16} @llvm.nvvm.suld.2d.v4i16.zero(i64, i32, i32)
define {i16,i16,i16,i16} @__clc_llvm_nvvm_suld_2d_v4i16_zero(i64 %img, i32 %x, i32 %y) nounwind alwaysinline {
entry:
  %0 = tail call {i16,i16,i16,i16} @llvm.nvvm.suld.2d.v4i16.zero(i64 %img, i32 %x, i32 %y);
  ret {i16,i16,i16,i16} %0
}

declare {i16,i16,i16,i16} @llvm.nvvm.suld.3d.v4i16.zero(i64, i32, i32, i32)
define {i16,i16,i16,i16} @__clc_llvm_nvvm_suld_3d_v4i16_zero(i64 %img, i32 %x, i32 %y, i32 %z) nounwind alwaysinline {
entry:
  %0 = tail call {i16,i16,i16,i16} @llvm.nvvm.suld.3d.v4i16.zero(i64 %img, i32 %x, i32 %y, i32 %z);
  ret {i16,i16,i16,i16} %0
}

declare {i32,i32,i32,i32} @llvm.nvvm.suld.1d.v4i32.trap(i64, i32)
define {i32,i32,i32,i32} @__clc_llvm_nvvm_suld_1d_v4i32_trap(i64 %img, i32 %x) nounwind alwaysinline {
entry:
  %0 = tail call {i32,i32,i32,i32} @llvm.nvvm.suld.1d.v4i32.trap(i64 %img, i32 %x);
  ret {i32,i32,i32,i32} %0
}

declare {i32,i32,i32,i32} @llvm.nvvm.suld.2d.v4i32.trap(i64, i32, i32)
define {i32,i32,i32,i32} @__clc_llvm_nvvm_suld_2d_v4i32_trap(i64 %img, i32 %x, i32 %y) nounwind alwaysinline {
entry:
  %0 = tail call {i32,i32,i32,i32} @llvm.nvvm.suld.2d.v4i32.trap(i64 %img, i32 %x, i32 %y);
  ret {i32,i32,i32,i32} %0
}

declare {i32,i32,i32,i32} @llvm.nvvm.suld.3d.v4i32.trap(i64, i32, i32, i32)
define {i32,i32,i32,i32} @__clc_llvm_nvvm_suld_3d_v4i32_trap(i64 %img, i32 %x, i32 %y, i32 %z) nounwind alwaysinline {
entry:
  %0 = tail call {i32,i32,i32,i32} @llvm.nvvm.suld.3d.v4i32.trap(i64 %img, i32 %x, i32 %y, i32 %z);
  ret {i32,i32,i32,i32} %0
}

declare {i32,i32,i32,i32} @llvm.nvvm.suld.1d.v4i32.clamp(i64, i32)
define {i32,i32,i32,i32} @__clc_llvm_nvvm_suld_1d_v4i32_clamp(i64 %img, i32 %x) nounwind alwaysinline {
entry:
  %0 = tail call {i32,i32,i32,i32} @llvm.nvvm.suld.1d.v4i32.clamp(i64 %img, i32 %x);
  ret {i32,i32,i32,i32} %0
}

declare {i32,i32,i32,i32} @llvm.nvvm.suld.2d.v4i32.clamp(i64, i32, i32)
define {i32,i32,i32,i32} @__clc_llvm_nvvm_suld_2d_v4i32_clamp(i64 %img, i32 %x, i32 %y) nounwind alwaysinline {
entry:
  %0 = tail call {i32,i32,i32,i32} @llvm.nvvm.suld.2d.v4i32.clamp(i64 %img, i32 %x, i32 %y);
  ret {i32,i32,i32,i32} %0
}

declare {i32,i32,i32,i32} @llvm.nvvm.suld.3d.v4i32.clamp(i64, i32, i32, i32)
define {i32,i32,i32,i32} @__clc_llvm_nvvm_suld_3d_v4i32_clamp(i64 %img, i32 %x, i32 %y, i32 %z) nounwind alwaysinline {
entry:
  %0 = tail call {i32,i32,i32,i32} @llvm.nvvm.suld.3d.v4i32.clamp(i64 %img, i32 %x, i32 %y, i32 %z);
  ret {i32,i32,i32,i32} %0
}

declare {i32,i32,i32,i32} @llvm.nvvm.suld.1d.v4i32.zero(i64, i32)
define {i32,i32,i32,i32} @__clc_llvm_nvvm_suld_1d_v4i32_zero(i64 %img, i32 %x) nounwind alwaysinline {
entry:
  %0 = tail call {i32,i32,i32,i32} @llvm.nvvm.suld.1d.v4i32.zero(i64 %img, i32 %x);
  ret {i32,i32,i32,i32} %0
}

declare {i32,i32,i32,i32} @llvm.nvvm.suld.2d.v4i32.zero(i64, i32, i32)
define {i32,i32,i32,i32} @__clc_llvm_nvvm_suld_2d_v4i32_zero(i64 %img, i32 %x, i32 %y) nounwind alwaysinline {
entry:
  %0 = tail call {i32,i32,i32,i32} @llvm.nvvm.suld.2d.v4i32.zero(i64 %img, i32 %x, i32 %y);
  ret {i32,i32,i32,i32} %0
}

declare {i32,i32,i32,i32} @llvm.nvvm.suld.3d.v4i32.zero(i64, i32, i32, i32)
define {i32,i32,i32,i32} @__clc_llvm_nvvm_suld_3d_v4i32_zero(i64 %img, i32 %x, i32 %y, i32 %z) nounwind alwaysinline {
entry:
  %0 = tail call {i32,i32,i32,i32} @llvm.nvvm.suld.3d.v4i32.zero(i64 %img, i32 %x, i32 %y, i32 %z);
  ret {i32,i32,i32,i32} %0
}
