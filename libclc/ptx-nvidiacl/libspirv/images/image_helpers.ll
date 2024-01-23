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

define <4 x i32> @__clc_struct32_to_vector({i32,i32,i32,i32} %s) nounwind alwaysinline {
  %a = alloca {i32,i32,i32,i32}
  store {i32,i32,i32,i32} %s, {i32,i32,i32,i32}* %a
  %bc = bitcast {i32,i32,i32,i32} * %a to <4 x i32> *
  %v = load <4 x i32>, <4 x i32> * %bc, align 128
  ret <4 x i32> %v
}

define <2 x i32> @__clc_struct32_to_vector2({i32,i32} %s) nounwind alwaysinline {
  %a = alloca {i32,i32}
  store {i32,i32} %s, {i32,i32}* %a
  %bc = bitcast {i32,i32} * %a to <2 x i32> *
  %v = load <2 x i32>, <2 x i32> * %bc, align 128
  ret <2 x i32> %v
}

define <4 x float> @__clc_structf32_to_vector({float,float,float,float} %s) nounwind alwaysinline {
  %a = alloca {float,float,float,float}
  store {float,float,float,float} %s, {float,float,float,float}* %a
  %bc = bitcast {float,float,float,float} * %a to <4 x float> *
  %v = load <4 x float>, <4 x float> * %bc, align 128
  ret <4 x float> %v
}

define <4 x i16> @__clc_struct16_to_vector({i16,i16,i16,i16} %s) nounwind alwaysinline {
  %a = alloca {i16,i16,i16,i16}
  store {i16,i16,i16,i16} %s, {i16,i16,i16,i16}* %a
  %bc = bitcast {i16,i16,i16,i16} * %a to <4 x i16> *
  %v = load <4 x i16>, <4 x i16> * %bc, align 128
  ret <4 x i16> %v
}

define <2 x i16> @__clc_struct16_to_vector2({i16,i16} %s) nounwind alwaysinline {
  %a = alloca {i16,i16}
  store {i16,i16} %s, {i16,i16}* %a
  %bc = bitcast {i16,i16} * %a to <2 x i16> *
  %v = load <2 x i16>, <2 x i16> * %bc, align 128
  ret <2 x i16> %v
}

// We need wrappers to convert intrisic return structures to vectors
declare {i16,i16,i16,i16} @llvm.nvvm.suld.1d.v4i16.trap(i64, i32)
define <4 x i16> @__clc_llvm_nvvm_suld_1d_v4i16_trap(i64 %img, i32 %x) nounwind alwaysinline {
entry:
  %0 = tail call {i16,i16,i16,i16} @llvm.nvvm.suld.1d.v4i16.trap(i64 %img, i32 %x);
  %1 = tail call <4 x i16> @__clc_struct16_to_vector({i16,i16,i16,i16} %0)
  ret <4 x i16> %1
}

declare {i16,i16,i16,i16} @llvm.nvvm.suld.2d.v4i16.trap(i64, i32, i32)
define <4 x i16> @__clc_llvm_nvvm_suld_2d_v4i16_trap(i64 %img, i32 %x, i32 %y) nounwind alwaysinline {
entry:
  %0 = tail call {i16,i16,i16,i16} @llvm.nvvm.suld.2d.v4i16.trap(i64 %img, i32 %x, i32 %y);
  %1 = tail call <4 x i16> @__clc_struct16_to_vector({i16,i16,i16,i16} %0)
  ret <4 x i16> %1
}

declare {i16,i16,i16,i16} @llvm.nvvm.suld.3d.v4i16.trap(i64, i32, i32, i32)
define <4 x i16> @__clc_llvm_nvvm_suld_3d_v4i16_trap(i64 %img, i32 %x, i32 %y, i32 %z) nounwind alwaysinline {
entry:
  %0 = tail call {i16,i16,i16,i16} @llvm.nvvm.suld.3d.v4i16.trap(i64 %img, i32 %x, i32 %y, i32 %z);
  %1 = tail call <4 x i16> @__clc_struct16_to_vector({i16,i16,i16,i16} %0)
  ret <4 x i16> %1
}

declare {i16,i16,i16,i16} @llvm.nvvm.suld.1d.v4i16.clamp(i64, i32)
define <4 x i16> @__clc_llvm_nvvm_suld_1d_v4i16_clamp(i64 %img, i32 %x) nounwind alwaysinline {
entry:
  %0 = tail call {i16,i16,i16,i16} @llvm.nvvm.suld.1d.v4i16.clamp(i64 %img, i32 %x);
  %1 = tail call <4 x i16> @__clc_struct16_to_vector({i16,i16,i16,i16} %0)
  ret <4 x i16> %1
}

declare {i16,i16,i16,i16} @llvm.nvvm.suld.2d.v4i16.clamp(i64, i32, i32)
define <4 x i16> @__clc_llvm_nvvm_suld_2d_v4i16_clamp(i64 %img, i32 %x, i32 %y) nounwind alwaysinline {
entry:
  %0 = tail call {i16,i16,i16,i16} @llvm.nvvm.suld.2d.v4i16.clamp(i64 %img, i32 %x, i32 %y);
  %1 = tail call <4 x i16> @__clc_struct16_to_vector({i16,i16,i16,i16} %0)
  ret <4 x i16> %1
}

declare {i16,i16,i16,i16} @llvm.nvvm.suld.3d.v4i16.clamp(i64, i32, i32, i32)
define <4 x i16> @__clc_llvm_nvvm_suld_3d_v4i16_clamp(i64 %img, i32 %x, i32 %y, i32 %z) nounwind alwaysinline {
entry:
  %0 = tail call {i16,i16,i16,i16} @llvm.nvvm.suld.3d.v4i16.clamp(i64 %img, i32 %x, i32 %y, i32 %z);
  %1 = tail call <4 x i16> @__clc_struct16_to_vector({i16,i16,i16,i16} %0)
  ret <4 x i16> %1
}

declare {i16,i16,i16,i16} @llvm.nvvm.suld.1d.v4i16.zero(i64, i32)
define <4 x i16> @__clc_llvm_nvvm_suld_1d_v4i16_zero(i64 %img, i32 %x) nounwind alwaysinline {
entry:
  %0 = tail call {i16,i16,i16,i16} @llvm.nvvm.suld.1d.v4i16.zero(i64 %img, i32 %x);
  %1 = tail call <4 x i16> @__clc_struct16_to_vector({i16,i16,i16,i16} %0)
  ret <4 x i16> %1
}

declare {i16,i16,i16,i16} @llvm.nvvm.suld.2d.v4i16.zero(i64, i32, i32)
define <4 x i16> @__clc_llvm_nvvm_suld_2d_v4i16_zero(i64 %img, i32 %x, i32 %y) nounwind alwaysinline {
entry:
  %0 = tail call {i16,i16,i16,i16} @llvm.nvvm.suld.2d.v4i16.zero(i64 %img, i32 %x, i32 %y);
  %1 = tail call <4 x i16> @__clc_struct16_to_vector({i16,i16,i16,i16} %0)
  ret <4 x i16> %1
}

declare {i16,i16,i16,i16} @llvm.nvvm.suld.3d.v4i16.zero(i64, i32, i32, i32)
define <4 x i16> @__clc_llvm_nvvm_suld_3d_v4i16_zero(i64 %img, i32 %x, i32 %y, i32 %z) nounwind alwaysinline {
entry:
  %0 = tail call {i16,i16,i16,i16} @llvm.nvvm.suld.3d.v4i16.zero(i64 %img, i32 %x, i32 %y, i32 %z);
  %1 = tail call <4 x i16> @__clc_struct16_to_vector({i16,i16,i16,i16} %0)
  ret <4 x i16> %1
}

declare {i32,i32,i32,i32} @llvm.nvvm.suld.1d.v4i32.trap(i64, i32)
define <4 x i32> @__clc_llvm_nvvm_suld_1d_v4i32_trap(i64 %img, i32 %x) nounwind alwaysinline {
entry:
  %0 = tail call {i32,i32,i32,i32} @llvm.nvvm.suld.1d.v4i32.trap(i64 %img, i32 %x);
  %1 = tail call <4 x i32> @__clc_struct32_to_vector({i32,i32,i32,i32} %0)
  ret <4 x i32> %1
}

declare {i32,i32,i32,i32} @llvm.nvvm.suld.2d.v4i32.trap(i64, i32, i32)
define <4 x i32> @__clc_llvm_nvvm_suld_2d_v4i32_trap(i64 %img, i32 %x, i32 %y) nounwind alwaysinline {
entry:
  %0 = tail call {i32,i32,i32,i32} @llvm.nvvm.suld.2d.v4i32.trap(i64 %img, i32 %x, i32 %y);
  %1 = tail call <4 x i32> @__clc_struct32_to_vector({i32,i32,i32,i32} %0)
  ret <4 x i32> %1
}

declare {i32,i32,i32,i32} @llvm.nvvm.suld.3d.v4i32.trap(i64, i32, i32, i32)
define <4 x i32> @__clc_llvm_nvvm_suld_3d_v4i32_trap(i64 %img, i32 %x, i32 %y, i32 %z) nounwind alwaysinline {
entry:
  %0 = tail call {i32,i32,i32,i32} @llvm.nvvm.suld.3d.v4i32.trap(i64 %img, i32 %x, i32 %y, i32 %z);
  %1 = tail call <4 x i32> @__clc_struct32_to_vector({i32,i32,i32,i32} %0)
  ret <4 x i32> %1
}

declare {i32,i32,i32,i32} @llvm.nvvm.suld.1d.v4i32.clamp(i64, i32)
define <4 x i32> @__clc_llvm_nvvm_suld_1d_v4i32_clamp(i64 %img, i32 %x) nounwind alwaysinline {
entry:
  %0 = tail call {i32,i32,i32,i32} @llvm.nvvm.suld.1d.v4i32.clamp(i64 %img, i32 %x);
  %1 = tail call <4 x i32> @__clc_struct32_to_vector({i32,i32,i32,i32} %0)
  ret <4 x i32> %1
}

declare {i32,i32,i32,i32} @llvm.nvvm.suld.2d.v4i32.clamp(i64, i32, i32)
define <4 x i32> @__clc_llvm_nvvm_suld_2d_v4i32_clamp(i64 %img, i32 %x, i32 %y) nounwind alwaysinline {
entry:
  %0 = tail call {i32,i32,i32,i32} @llvm.nvvm.suld.2d.v4i32.clamp(i64 %img, i32 %x, i32 %y);
  %1 = tail call <4 x i32> @__clc_struct32_to_vector({i32,i32,i32,i32} %0)
  ret <4 x i32> %1
}

declare {i32,i32,i32,i32} @llvm.nvvm.suld.3d.v4i32.clamp(i64, i32, i32, i32)
define <4 x i32> @__clc_llvm_nvvm_suld_3d_v4i32_clamp(i64 %img, i32 %x, i32 %y, i32 %z) nounwind alwaysinline {
entry:
  %0 = tail call {i32,i32,i32,i32} @llvm.nvvm.suld.3d.v4i32.clamp(i64 %img, i32 %x, i32 %y, i32 %z);
  %1 = tail call <4 x i32> @__clc_struct32_to_vector({i32,i32,i32,i32} %0)
  ret <4 x i32> %1
}

declare {i32,i32,i32,i32} @llvm.nvvm.suld.1d.v4i32.zero(i64, i32)
define <4 x i32> @__clc_llvm_nvvm_suld_1d_v4i32_zero(i64 %img, i32 %x) nounwind alwaysinline {
entry:
  %0 = tail call {i32,i32,i32,i32} @llvm.nvvm.suld.1d.v4i32.zero(i64 %img, i32 %x);
  %1 = tail call <4 x i32> @__clc_struct32_to_vector({i32,i32,i32,i32} %0)
  ret <4 x i32> %1
}

declare {i32,i32,i32,i32} @llvm.nvvm.suld.2d.v4i32.zero(i64, i32, i32)
define <4 x i32> @__clc_llvm_nvvm_suld_2d_v4i32_zero(i64 %img, i32 %x, i32 %y) nounwind alwaysinline {
entry:
  %0 = tail call {i32,i32,i32,i32} @llvm.nvvm.suld.2d.v4i32.zero(i64 %img, i32 %x, i32 %y);
  %1 = tail call <4 x i32> @__clc_struct32_to_vector({i32,i32,i32,i32} %0)
  ret <4 x i32> %1
}

declare {i32,i32,i32,i32} @llvm.nvvm.suld.3d.v4i32.zero(i64, i32, i32, i32)
define <4 x i32> @__clc_llvm_nvvm_suld_3d_v4i32_zero(i64 %img, i32 %x, i32 %y, i32 %z) nounwind alwaysinline {
entry:
  %0 = tail call {i32,i32,i32,i32} @llvm.nvvm.suld.3d.v4i32.zero(i64 %img, i32 %x, i32 %y, i32 %z);
  %1 = tail call <4 x i32> @__clc_struct32_to_vector({i32,i32,i32,i32} %0)
  ret <4 x i32> %1
}


; <--- BINDLESS IMAGES PROTOTYPE --->

; <--- SURFACES --->

declare {i16,i16} @llvm.nvvm.suld.1d.v2i8.clamp(i64, i32)
define <2 x i16> @__clc_llvm_nvvm_suld_1d_v2i8_clamp(i64 %img, i32 %x) nounwind alwaysinline {
entry:
  %0 = tail call {i16,i16} @llvm.nvvm.suld.1d.v2i8.clamp(i64 %img, i32 %x);
  %1 = tail call <2 x i16> @__clc_struct16_to_vector2({i16,i16} %0)
  ret <2 x i16> %1
}

declare {i16,i16} @llvm.nvvm.suld.2d.v2i8.clamp(i64, i32, i32)
define <2 x i16> @__clc_llvm_nvvm_suld_2d_v2i8_clamp(i64 %img, i32 %x, i32 %y) nounwind alwaysinline {
entry:
  %0 = tail call {i16,i16} @llvm.nvvm.suld.2d.v2i8.clamp(i64 %img, i32 %x, i32 %y);
  %1 = tail call <2 x i16> @__clc_struct16_to_vector2({i16,i16} %0)
  ret <2 x i16> %1
}

declare {i16,i16} @llvm.nvvm.suld.3d.v2i8.clamp(i64, i32, i32, i32)
define <2 x i16> @__clc_llvm_nvvm_suld_3d_v2i8_clamp(i64 %img, i32 %x, i32 %y, i32 %z) nounwind alwaysinline {
entry:
  %0 = tail call {i16,i16} @llvm.nvvm.suld.3d.v2i8.clamp(i64 %img, i32 %x, i32 %y, i32 %z);
  %1 = tail call <2 x i16> @__clc_struct16_to_vector2({i16,i16} %0)
  ret <2 x i16> %1
}

declare {i16,i16,i16,i16} @llvm.nvvm.suld.1d.v4i8.clamp(i64, i32)
define <4 x i16> @__clc_llvm_nvvm_suld_1d_v4i8_clamp(i64 %img, i32 %x) nounwind alwaysinline {
entry:
  %0 = tail call {i16,i16,i16,i16} @llvm.nvvm.suld.1d.v4i8.clamp(i64 %img, i32 %x);
  %1 = tail call <4 x i16> @__clc_struct16_to_vector({i16,i16,i16,i16} %0)
  ret <4 x i16> %1
}

declare {i16,i16,i16,i16} @llvm.nvvm.suld.2d.v4i8.clamp(i64, i32, i32)
define <4 x i16> @__clc_llvm_nvvm_suld_2d_v4i8_clamp(i64 %img, i32 %x, i32 %y) nounwind alwaysinline {
entry:
  %0 = tail call {i16,i16,i16,i16} @llvm.nvvm.suld.2d.v4i8.clamp(i64 %img, i32 %x, i32 %y);
  %1 = tail call <4 x i16> @__clc_struct16_to_vector({i16,i16,i16,i16} %0)
  ret <4 x i16> %1
}

declare {i16,i16,i16,i16} @llvm.nvvm.suld.3d.v4i8.clamp(i64, i32, i32, i32)
define <4 x i16> @__clc_llvm_nvvm_suld_3d_v4i8_clamp(i64 %img, i32 %x, i32 %y, i32 %z) nounwind alwaysinline {
entry:
  %0 = tail call {i16,i16,i16,i16} @llvm.nvvm.suld.3d.v4i8.clamp(i64 %img, i32 %x, i32 %y, i32 %z);
  %1 = tail call <4 x i16> @__clc_struct16_to_vector({i16,i16,i16,i16} %0)
  ret <4 x i16> %1
}

; <--- TEXTURES --->
declare {i32,i32,i32,i32} @llvm.nvvm.tex.unified.1d.v4s32.f32(i64, float)
define <4 x i32> @__clc_llvm_nvvm_tex_1d_v4i32_f32(i64 %img, float %x) nounwind alwaysinline {
entry:
  %0 = tail call {i32,i32,i32,i32} @llvm.nvvm.tex.unified.1d.v4s32.f32(i64 %img, float %x);
  %1 = tail call <4 x i32>@__clc_struct32_to_vector({i32,i32,i32,i32} %0)
  ret <4 x i32> %1
}

declare {i32,i32,i32,i32} @llvm.nvvm.tex.unified.2d.v4s32.f32(i64, float, float)
define <4 x i32> @__clc_llvm_nvvm_tex_2d_v4i32_f32(i64 %img, float %x, float %y) nounwind alwaysinline {
entry:
  %0 = tail call {i32,i32,i32,i32} @llvm.nvvm.tex.unified.2d.v4s32.f32(i64 %img, float %x, float %y);
  %1 = tail call <4 x i32>@__clc_struct32_to_vector({i32,i32,i32,i32} %0)
  ret <4 x i32> %1
}

declare {i32,i32,i32,i32} @llvm.nvvm.tex.unified.3d.v4s32.f32(i64, float, float, float)
define <4 x i32> @__clc_llvm_nvvm_tex_3d_v4i32_f32(i64 %img, float %x, float %y, float %z) nounwind alwaysinline {
entry:
  %0 = tail call {i32,i32,i32,i32} @llvm.nvvm.tex.unified.3d.v4s32.f32(i64 %img, float %x, float %y, float %z);
  %1 = tail call <4 x i32>@__clc_struct32_to_vector({i32,i32,i32,i32} %0)
  ret <4 x i32> %1
}

declare {i32,i32,i32,i32} @llvm.nvvm.tex.unified.1d.v4u32.f32(i64, float)
define <4 x i32> @__clc_llvm_nvvm_tex_1d_v4j32_f32(i64 %img, float %x) nounwind alwaysinline {
entry:
  %0 = tail call {i32,i32,i32,i32} @llvm.nvvm.tex.unified.1d.v4u32.f32(i64 %img, float %x);
  %1 = tail call <4 x i32>@__clc_struct32_to_vector({i32,i32,i32,i32} %0)
  ret <4 x i32> %1
}

declare {i32,i32,i32,i32} @llvm.nvvm.tex.unified.2d.v4u32.f32(i64, float, float)
define <4 x i32> @__clc_llvm_nvvm_tex_2d_v4j32_f32(i64 %img, float %x, float %y) nounwind alwaysinline {
entry:
  %0 = tail call {i32,i32,i32,i32} @llvm.nvvm.tex.unified.2d.v4u32.f32(i64 %img, float %x, float %y);
  %1 = tail call <4 x i32>@__clc_struct32_to_vector({i32,i32,i32,i32} %0)
  ret <4 x i32> %1
}

declare {i32,i32,i32,i32} @llvm.nvvm.tex.unified.3d.v4u32.f32(i64, float, float, float)
define <4 x i32> @__clc_llvm_nvvm_tex_3d_v4j32_f32(i64 %img, float %x, float %y, float %z) nounwind alwaysinline {
entry:
  %0 = tail call {i32,i32,i32,i32} @llvm.nvvm.tex.unified.3d.v4u32.f32(i64 %img, float %x, float %y, float %z);
  %1 = tail call <4 x i32>@__clc_struct32_to_vector({i32,i32,i32,i32} %0)
  ret <4 x i32> %1
}

declare {float,float,float,float} @llvm.nvvm.tex.unified.1d.v4f32.f32(i64, float)
define <4 x float> @__clc_llvm_nvvm_tex_1d_v4f32_f32(i64 %img, float %x) nounwind alwaysinline {
entry:
  %0 = tail call {float,float,float,float} @llvm.nvvm.tex.unified.1d.v4f32.f32(i64 %img, float %x);
  %1 = tail call <4 x float>@__clc_structf32_to_vector({float,float,float,float} %0)
  ret <4 x float> %1
}

declare {float,float,float,float} @llvm.nvvm.tex.unified.2d.v4f32.f32(i64, float, float)
define <4 x float> @__clc_llvm_nvvm_tex_2d_v4f32_f32(i64 %img, float %x, float %y) nounwind alwaysinline {
entry:
  %0 = tail call {float,float,float,float} @llvm.nvvm.tex.unified.2d.v4f32.f32(i64 %img, float %x, float %y);
  %1 = tail call <4 x float>@__clc_structf32_to_vector({float,float,float,float} %0)
  ret <4 x float> %1
}

declare {float,float,float,float} @llvm.nvvm.tex.unified.3d.v4f32.f32(i64, float, float, float)
define <4 x float> @__clc_llvm_nvvm_tex_3d_v4f32_f32(i64 %img, float %x, float %y, float %z) nounwind alwaysinline {
entry:
  %0 = tail call {float,float,float,float} @llvm.nvvm.tex.unified.3d.v4f32.f32(i64 %img, float %x, float %y, float %z);
  %1 = tail call <4 x float>@__clc_structf32_to_vector({float,float,float,float} %0)
  ret <4 x float> %1
}



; <--- MIPMAP --->
; Level
declare {float,float,float,float} @llvm.nvvm.tex.unified.1d.level.v4f32.f32(i64, float, float)
define <4 x float> @__clc_llvm_nvvm_tex_1d_level_v4f32_f32(i64 %img, float %x, float %lvl) nounwind alwaysinline {
entry:
  %0 = tail call {float,float,float,float} @llvm.nvvm.tex.unified.1d.level.v4f32.f32(i64 %img, float %x, float %lvl);
  %1 = tail call <4 x float>@__clc_structf32_to_vector({float,float,float,float} %0)
  ret <4 x float> %1
}

declare {float,float,float,float} @llvm.nvvm.tex.unified.2d.level.v4f32.f32(i64, float, float, float)
define <4 x float> @__clc_llvm_nvvm_tex_2d_level_v4f32_f32(i64 %img, float %x, float %y, float %lvl) nounwind alwaysinline {
entry:
  %0 = tail call {float,float,float,float} @llvm.nvvm.tex.unified.2d.level.v4f32.f32(i64 %img, float %x, float %y, float %lvl);
  %1 = tail call <4 x float>@__clc_structf32_to_vector({float,float,float,float} %0)
  ret <4 x float> %1
}

declare {float,float,float,float} @llvm.nvvm.tex.unified.3d.level.v4f32.f32(i64, float, float, float, float)
define <4 x float> @__clc_llvm_nvvm_tex_3d_level_v4f32_f32(i64 %img, float %x, float %y, float %z, float %lvl) nounwind alwaysinline {
entry:
  %0 = tail call {float,float,float,float} @llvm.nvvm.tex.unified.3d.level.v4f32.f32(i64 %img, float %x, float %y, float %z, float %lvl);
  %1 = tail call <4 x float>@__clc_structf32_to_vector({float,float,float,float} %0)
  ret <4 x float> %1
}

declare {i32,i32,i32,i32} @llvm.nvvm.tex.unified.1d.level.v4s32.f32(i64, float, float)
define <4 x i32> @__clc_llvm_nvvm_tex_1d_level_v4i32_f32(i64 %img, float %x, float %lvl) nounwind alwaysinline {
entry:
  %0 = tail call {i32,i32,i32,i32} @llvm.nvvm.tex.unified.1d.level.v4s32.f32(i64 %img, float %x, float %lvl);
  %1 = tail call <4 x i32>@__clc_struct32_to_vector({i32,i32,i32,i32} %0)
  ret <4 x i32> %1
}

declare {i32,i32,i32,i32} @llvm.nvvm.tex.unified.2d.level.v4s32.f32(i64, float, float, float)
define <4 x i32> @__clc_llvm_nvvm_tex_2d_level_v4i32_f32(i64 %img, float %x, float %y, float %lvl) nounwind alwaysinline {
entry:
  %0 = tail call {i32,i32,i32,i32} @llvm.nvvm.tex.unified.2d.level.v4s32.f32(i64 %img, float %x, float %y, float %lvl);
  %1 = tail call <4 x i32>@__clc_struct32_to_vector({i32,i32,i32,i32} %0)
  ret <4 x i32> %1
}

declare {i32,i32,i32,i32} @llvm.nvvm.tex.unified.3d.level.v4s32.f32(i64, float, float, float, float)
define <4 x i32> @__clc_llvm_nvvm_tex_3d_level_v4i32_f32(i64 %img, float %x, float %y, float %z, float %lvl) nounwind alwaysinline {
entry:
  %0 = tail call {i32,i32,i32,i32} @llvm.nvvm.tex.unified.3d.level.v4s32.f32(i64 %img, float %x, float %y, float %z, float %lvl);
  %1 = tail call <4 x i32>@__clc_struct32_to_vector({i32,i32,i32,i32} %0)
  ret <4 x i32> %1
}

declare {i32,i32,i32,i32} @llvm.nvvm.tex.unified.1d.level.v4u32.f32(i64, float, float)
define <4 x i32> @__clc_llvm_nvvm_tex_1d_level_v4j32_f32(i64 %img, float %x, float %lvl) nounwind alwaysinline {
entry:
  %0 = tail call {i32,i32,i32,i32} @llvm.nvvm.tex.unified.1d.level.v4u32.f32(i64 %img, float %x, float %lvl);
  %1 = tail call <4 x i32>@__clc_struct32_to_vector({i32,i32,i32,i32} %0)
  ret <4 x i32> %1
}

declare {i32,i32,i32,i32} @llvm.nvvm.tex.unified.2d.level.v4u32.f32(i64, float, float, float)
define <4 x i32> @__clc_llvm_nvvm_tex_2d_level_v4j32_f32(i64 %img, float %x, float %y, float %lvl) nounwind alwaysinline {
entry:
  %0 = tail call {i32,i32,i32,i32} @llvm.nvvm.tex.unified.2d.level.v4u32.f32(i64 %img, float %x, float %y, float %lvl);
  %1 = tail call <4 x i32>@__clc_struct32_to_vector({i32,i32,i32,i32} %0)
  ret <4 x i32> %1
}

declare {i32,i32,i32,i32} @llvm.nvvm.tex.unified.3d.level.v4u32.f32(i64, float, float, float, float)
define <4 x i32> @__clc_llvm_nvvm_tex_3d_level_v4j32_f32(i64 %img, float %x, float %y, float %z, float %lvl) nounwind alwaysinline {
entry:
  %0 = tail call {i32,i32,i32,i32} @llvm.nvvm.tex.unified.3d.level.v4u32.f32(i64 %img, float %x, float %y, float %z, float %lvl);
  %1 = tail call <4 x i32>@__clc_struct32_to_vector({i32,i32,i32,i32} %0)
  ret <4 x i32> %1
}

; Grad
declare {float,float,float,float} @llvm.nvvm.tex.unified.1d.grad.v4f32.f32(i64, float, float, float)
define <4 x float> @__clc_llvm_nvvm_tex_1d_grad_v4f32_f32(i64 %img, float %x, float %dX, float %dY) nounwind alwaysinline {
entry:
  %0 = tail call {float,float,float,float} @llvm.nvvm.tex.unified.1d.grad.v4f32.f32(i64 %img, float %x, float %dX, float %dY);
  %1 = tail call <4 x float>@__clc_structf32_to_vector({float,float,float,float} %0)
  ret <4 x float> %1
}

declare {float,float,float,float} @llvm.nvvm.tex.unified.2d.grad.v4f32.f32(i64, float, float, float, float, float, float)
define <4 x float> @__clc_llvm_nvvm_tex_2d_grad_v4f32_f32(i64 %img, float %x, float %y, float %dXx, float %dXy, float %dYx, float %dYy) nounwind alwaysinline {
entry:
  %0 = tail call {float,float,float,float} @llvm.nvvm.tex.unified.2d.grad.v4f32.f32(i64 %img, float %x, float %y, float %dXx, float %dXy, float %dYx, float %dYy);
  %1 = tail call <4 x float>@__clc_structf32_to_vector({float,float,float,float} %0)
  ret <4 x float> %1
}

declare {float,float,float,float} @llvm.nvvm.tex.unified.3d.grad.v4f32.f32(i64, float, float, float, float, float, float, float, float, float)
define <4 x float> @__clc_llvm_nvvm_tex_3d_grad_v4f32_f32(i64 %img, float %x, float %y, float %z, float %dXx, float %dXy, float %dXz, float %dYx, float %dYy, float %dYz) nounwind alwaysinline {
entry:
  %0 = tail call {float,float,float,float} @llvm.nvvm.tex.unified.3d.grad.v4f32.f32(i64 %img, float %x, float %y, float %z, float %dXx, float %dXy, float %dXz, float %dYx, float %dYy, float %dYz);
  %1 = tail call <4 x float>@__clc_structf32_to_vector({float,float,float,float} %0)
  ret <4 x float> %1
}

declare {i32,i32,i32,i32} @llvm.nvvm.tex.unified.1d.grad.v4s32.f32(i64, float, float, float)
define <4 x i32> @__clc_llvm_nvvm_tex_1d_grad_v4i32_f32(i64 %img, float %x, float %dX, float %dY) nounwind alwaysinline {
entry:
  %0 = tail call {i32,i32,i32,i32} @llvm.nvvm.tex.unified.1d.grad.v4s32.f32(i64 %img, float %x, float %dX, float %dY);
  %1 = tail call <4 x i32>@__clc_struct32_to_vector({i32,i32,i32,i32} %0)
  ret <4 x i32> %1
}

declare {i32,i32,i32,i32} @llvm.nvvm.tex.unified.2d.grad.v4s32.f32(i64, float, float, float, float, float, float)
define <4 x i32> @__clc_llvm_nvvm_tex_2d_grad_v4i32_f32(i64 %img, float %x, float %y, float %dXx, float %dXy, float %dYx, float %dYy) nounwind alwaysinline {
entry:
  %0 = tail call {i32,i32,i32,i32} @llvm.nvvm.tex.unified.2d.grad.v4s32.f32(i64 %img, float %x, float %y, float %dXx, float %dXy, float %dYx, float %dYy);
  %1 = tail call <4 x i32>@__clc_struct32_to_vector({i32,i32,i32,i32} %0)
  ret <4 x i32> %1
}

declare {i32,i32,i32,i32} @llvm.nvvm.tex.unified.3d.grad.v4s32.f32(i64, float, float, float, float, float, float, float, float, float)
define <4 x i32> @__clc_llvm_nvvm_tex_3d_grad_v4i32_f32(i64 %img, float %x, float %y, float %z, float %dXx, float %dXy, float %dXz, float %dYx, float %dYy, float %dYz) nounwind alwaysinline {
entry:
  %0 = tail call {i32,i32,i32,i32} @llvm.nvvm.tex.unified.3d.grad.v4s32.f32(i64 %img, float %x, float %y, float %z, float %dXx, float %dXy, float %dXz, float %dYx, float %dYy, float %dYz);
  %1 = tail call <4 x i32>@__clc_struct32_to_vector({i32,i32,i32,i32} %0)
  ret <4 x i32> %1
}

declare {i32,i32,i32,i32} @llvm.nvvm.tex.unified.1d.grad.v4u32.f32(i64, float, float, float)
define <4 x i32> @__clc_llvm_nvvm_tex_1d_grad_v4j32_f32(i64 %img, float %x, float %dX, float %dY) nounwind alwaysinline {
entry:
  %0 = tail call {i32,i32,i32,i32} @llvm.nvvm.tex.unified.1d.grad.v4u32.f32(i64 %img, float %x, float %dX, float %dY);
  %1 = tail call <4 x i32>@__clc_struct32_to_vector({i32,i32,i32,i32} %0)
  ret <4 x i32> %1
}

declare {i32,i32,i32,i32} @llvm.nvvm.tex.unified.2d.grad.v4u32.f32(i64, float, float, float, float, float, float)
define <4 x i32> @__clc_llvm_nvvm_tex_2d_grad_v4j32_f32(i64 %img, float %x, float %y, float %dXx, float %dXy, float %dYx, float %dYy) nounwind alwaysinline {
entry:
  %0 = tail call {i32,i32,i32,i32} @llvm.nvvm.tex.unified.2d.grad.v4u32.f32(i64 %img, float %x, float %y, float %dXx, float %dXy, float %dYx, float %dYy);
  %1 = tail call <4 x i32>@__clc_struct32_to_vector({i32,i32,i32,i32} %0)
  ret <4 x i32> %1
}

declare {i32,i32,i32,i32} @llvm.nvvm.tex.unified.3d.grad.v4u32.f32(i64, float, float, float, float, float, float, float, float, float)
define <4 x i32> @__clc_llvm_nvvm_tex_3d_grad_v4j32_f32(i64 %img, float %x, float %y, float %z, float %dXx, float %dXy, float %dXz, float %dYx, float %dYy, float %dYz) nounwind alwaysinline {
entry:
  %0 = tail call {i32,i32,i32,i32} @llvm.nvvm.tex.unified.3d.grad.v4u32.f32(i64 %img, float %x, float %y, float %z, float %dXx, float %dXy, float %dXz, float %dYx, float %dYy, float %dYz);
  %1 = tail call <4 x i32>@__clc_struct32_to_vector({i32,i32,i32,i32} %0)
  ret <4 x i32> %1
}

; <--- IMAGE ARRAYS --->

; Surface Reads
;
; @llvm.nvvm.suld.<NDims>.array.v<NChannels><DType>.clamp
;
; <NDims> = { 1d, 2d, 3d }
; <NChannels> = { 2, 4 } 
; <Dtype> = { i8, i16, i32 }
;
; Note: The case of NChannels=1 doesn't need to be handled here as it can be
; called directly.


; @llvm.nvvm.suld.<NDims>.array.v<NChannels>{i8, i16, i32}.clamp

; - @llvm.nvvm.suld.<NDims>.array.v{2, 4}i8.clamp

; - - @llvm.nvvm.suld.{1d, 2d, 3d}.array.v2i8.clamp

declare {i16,i16} @llvm.nvvm.suld.1d.array.v2i8.clamp(i64, i32, i32)
define <2 x i16> @__clc_llvm_nvvm_suld_1d_array_v2i8_clamp(i64 %img, i32 %idx, i32 %x) nounwind alwaysinline {
entry:
  %0 = tail call {i16,i16} @llvm.nvvm.suld.1d.array.v2i8.clamp(i64 %img, i32 %idx, i32 %x);
  %1 = tail call <2 x i16> @__clc_struct16_to_vector2({i16,i16} %0)
  ret <2 x i16> %1
}

declare {i16,i16} @llvm.nvvm.suld.2d.array.v2i8.clamp(i64, i32, i32, i32)
define <2 x i16> @__clc_llvm_nvvm_suld_2d_array_v2i8_clamp(i64 %img, i32 %idx, i32 %x, i32 %y) nounwind alwaysinline {
entry:
  %0 = tail call {i16,i16} @llvm.nvvm.suld.2d.array.v2i8.clamp(i64 %img, i32 %idx, i32 %x, i32 %y);
  %1 = tail call <2 x i16> @__clc_struct16_to_vector2({i16,i16} %0)
  ret <2 x i16> %1
}

declare {i16,i16} @llvm.nvvm.suld.3d.array.v2i8.clamp(i64, i32, i32, i32, i32)
define <2 x i16> @__clc_llvm_nvvm_suld_3d_array_v2i8_clamp(i64 %img, i32 %idx, i32 %x, i32 %y, i32 %z) nounwind alwaysinline {
entry:
  %0 = tail call {i16,i16} @llvm.nvvm.suld.3d.array.v2i8.clamp(i64 %img, i32 %idx, i32 %x, i32 %y, i32 %z);
  %1 = tail call <2 x i16> @__clc_struct16_to_vector2({i16,i16} %0)
  ret <2 x i16> %1
}

; - - @llvm.nvvm.suld.{1d, 2d, 3d}.array.v4i8.clamp

declare {i16,i16,i16,i16} @llvm.nvvm.suld.1d.array.v4i8.clamp(i64, i32, i32)
define <4 x i16> @__clc_llvm_nvvm_suld_1d_array_v4i8_clamp(i64 %img, i32 %idx, i32 %x) nounwind alwaysinline {
entry:
  %0 = tail call {i16,i16,i16,i16} @llvm.nvvm.suld.1d.array.v4i8.clamp(i64 %img, i32 %idx, i32 %x);
  %1 = tail call <4 x i16> @__clc_struct16_to_vector({i16,i16,i16,i16} %0)
  ret <4 x i16> %1
}

declare {i16,i16,i16,i16} @llvm.nvvm.suld.2d.array.v4i8.clamp(i64, i32, i32, i32)
define <4 x i16> @__clc_llvm_nvvm_suld_2d_array_v4i8_clamp(i64 %img, i32 %idx, i32 %x, i32 %y) nounwind alwaysinline {
entry:
  %0 = tail call {i16,i16,i16,i16} @llvm.nvvm.suld.2d.array.v4i8.clamp(i64 %img, i32 %idx, i32 %x, i32 %y);
  %1 = tail call <4 x i16> @__clc_struct16_to_vector({i16,i16,i16,i16} %0)
  ret <4 x i16> %1
}

declare {i16,i16,i16,i16} @llvm.nvvm.suld.3d.array.v4i8.clamp(i64, i32, i32, i32, i32)
define <4 x i16> @__clc_llvm_nvvm_suld_3d_array_v4i8_clamp(i64 %img, i32 %idx, i32 %x, i32 %y, i32 %z) nounwind alwaysinline {
entry:
  %0 = tail call {i16,i16,i16,i16} @llvm.nvvm.suld.3d.array.v4i8.clamp(i64 %img, i32 %idx, i32 %x, i32 %y, i32 %z);
  %1 = tail call <4 x i16> @__clc_struct16_to_vector({i16,i16,i16,i16} %0)
  ret <4 x i16> %1
}

; - @llvm.nvvm.suld.<NDims>.array.v{2, 4}i16.clamp

; - - @llvm.nvvm.suld.{1d, 2d, 3d}.array.v2i16.clamp

declare {i16,i16} @llvm.nvvm.suld.1d.array.v2i16.clamp(i64, i32, i32)
define <2 x i16> @__clc_llvm_nvvm_suld_1d_array_v2i16_clamp(i64 %img, i32 %idx, i32 %x) nounwind alwaysinline {
entry:
  %0 = tail call {i16,i16} @llvm.nvvm.suld.1d.array.v2i16.clamp(i64 %img, i32 %idx, i32 %x);
  %1 = tail call <2 x i16> @__clc_struct16_to_vector2({i16,i16} %0)
  ret <2 x i16> %1
}

declare {i16,i16} @llvm.nvvm.suld.2d.array.v2i16.clamp(i64, i32, i32, i32)
define <2 x i16> @__clc_llvm_nvvm_suld_2d_array_v2i16_clamp(i64 %img, i32 %idx, i32 %x, i32 %y) nounwind alwaysinline {
entry:
  %0 = tail call {i16,i16} @llvm.nvvm.suld.2d.array.v2i16.clamp(i64 %img, i32 %idx, i32 %x, i32 %y);
  %1 = tail call <2 x i16> @__clc_struct16_to_vector2({i16,i16} %0)
  ret <2 x i16> %1
}

declare {i16,i16} @llvm.nvvm.suld.3d.array.v2i16.clamp(i64, i32, i32, i32, i32)
define <2 x i16> @__clc_llvm_nvvm_suld_3d_array_v2i16_clamp(i64 %img, i32 %idx, i32 %x, i32 %y, i32 %z) nounwind alwaysinline {
entry:
  %0 = tail call {i16,i16} @llvm.nvvm.suld.3d.array.v2i16.clamp(i64 %img, i32 %idx, i32 %x, i32 %y, i32 %z);
  %1 = tail call <2 x i16> @__clc_struct16_to_vector2({i16,i16} %0)
  ret <2 x i16> %1
}

; - - @llvm.nvvm.suld.{1d, 2d, 3d}.array.v4i16.clamp

declare {i16,i16,i16,i16} @llvm.nvvm.suld.1d.array.v4i16.clamp(i64, i32, i32)
define <4 x i16> @__clc_llvm_nvvm_suld_1d_array_v4i16_clamp(i64 %img, i32 %idx, i32 %x) nounwind alwaysinline {
entry:
  %0 = tail call {i16,i16,i16,i16} @llvm.nvvm.suld.1d.array.v4i16.clamp(i64 %img, i32 %idx, i32 %x);
  %1 = tail call <4 x i16> @__clc_struct16_to_vector({i16,i16,i16,i16} %0)
  ret <4 x i16> %1
}

declare {i16,i16,i16,i16} @llvm.nvvm.suld.2d.array.v4i16.clamp(i64, i32, i32, i32)
define <4 x i16> @__clc_llvm_nvvm_suld_2d_array_v4i16_clamp(i64 %img, i32 %idx, i32 %x, i32 %y) nounwind alwaysinline {
entry:
  %0 = tail call {i16,i16,i16,i16} @llvm.nvvm.suld.2d.array.v4i16.clamp(i64 %img, i32 %idx, i32 %x, i32 %y);
  %1 = tail call <4 x i16> @__clc_struct16_to_vector({i16,i16,i16,i16} %0)
  ret <4 x i16> %1
}

declare {i16,i16,i16,i16} @llvm.nvvm.suld.3d.array.v4i16.clamp(i64, i32, i32, i32, i32)
define <4 x i16> @__clc_llvm_nvvm_suld_3d_array_v4i16_clamp(i64 %img, i32 %idx, i32 %x, i32 %y, i32 %z) nounwind alwaysinline {
entry:
  %0 = tail call {i16,i16,i16,i16} @llvm.nvvm.suld.3d.array.v4i16.clamp(i64 %img, i32 %idx, i32 %x, i32 %y, i32 %z);
  %1 = tail call <4 x i16> @__clc_struct16_to_vector({i16,i16,i16,i16} %0)
  ret <4 x i16> %1
}

; - @llvm.nvvm.suld.<NDims>.array.v{2, 4}i32.clamp

; - - @llvm.nvvm.suld.{1d, 2d, 3d}.array.v2i32.clamp

declare {i32,i32} @llvm.nvvm.suld.1d.array.v2i32.clamp(i64, i32, i32)
define <2 x i32> @__clc_llvm_nvvm_suld_1d_array_v2i32_clamp(i64 %img, i32 %idx, i32 %x) nounwind alwaysinline {
entry:
  %0 = tail call {i32,i32} @llvm.nvvm.suld.1d.array.v2i32.clamp(i64 %img, i32 %idx, i32 %x);
  %1 = tail call <2 x i32> @__clc_struct32_to_vector2({i32,i32} %0)
  ret <2 x i32> %1
}

declare {i32,i32} @llvm.nvvm.suld.2d.array.v2i32.clamp(i64, i32, i32, i32)
define <2 x i32> @__clc_llvm_nvvm_suld_2d_array_v2i32_clamp(i64 %img, i32 %idx, i32 %x, i32 %y) nounwind alwaysinline {
entry:
  %0 = tail call {i32,i32} @llvm.nvvm.suld.2d.array.v2i32.clamp(i64 %img, i32 %idx, i32 %x, i32 %y);
  %1 = tail call <2 x i32> @__clc_struct32_to_vector2({i32,i32} %0)
  ret <2 x i32> %1
}

declare {i32,i32} @llvm.nvvm.suld.3d.array.v2i32.clamp(i64, i32, i32, i32, i32)
define <2 x i32> @__clc_llvm_nvvm_suld_3d_array_v2i32_clamp(i64 %img, i32 %idx, i32 %x, i32 %y, i32 %z) nounwind alwaysinline {
entry:
  %0 = tail call {i32,i32} @llvm.nvvm.suld.3d.array.v2i32.clamp(i64 %img, i32 %idx, i32 %x, i32 %y, i32 %z);
  %1 = tail call <2 x i32> @__clc_struct32_to_vector2({i32,i32} %0)
  ret <2 x i32> %1
}

; - @llvm.nvvm.suld.<NDims>.array.v4i32.clamp

; - - @llvm.nvvm.suld.{1d, 2d, 3d}.array.v4i32.clamp

declare {i32,i32,i32,i32} @llvm.nvvm.suld.1d.array.v4i32.clamp(i64, i32, i32)
define <4 x i32> @__clc_llvm_nvvm_suld_1d_array_v4i32_clamp(i64 %img, i32 %idx, i32 %x) nounwind alwaysinline {
entry:
  %0 = tail call {i32,i32,i32,i32} @llvm.nvvm.suld.1d.array.v4i32.clamp(i64 %img, i32 %idx, i32 %x);
  %1 = tail call <4 x i32> @__clc_struct32_to_vector({i32,i32,i32,i32} %0)
  ret <4 x i32> %1
}

declare {i32,i32,i32,i32} @llvm.nvvm.suld.2d.array.v4i32.clamp(i64, i32, i32, i32)
define <4 x i32> @__clc_llvm_nvvm_suld_2d_array_v4i32_clamp(i64 %img, i32 %idx, i32 %x, i32 %y) nounwind alwaysinline {
entry:
  %0 = tail call {i32,i32,i32,i32} @llvm.nvvm.suld.2d.array.v4i32.clamp(i64 %img, i32 %idx, i32 %x, i32 %y);
  %1 = tail call <4 x i32> @__clc_struct32_to_vector({i32,i32,i32,i32} %0)
  ret <4 x i32> %1
}

declare {i32,i32,i32,i32} @llvm.nvvm.suld.3d.array.v4i32.clamp(i64, i32, i32, i32, i32)
define <4 x i32> @__clc_llvm_nvvm_suld_3d_array_v4i32_clamp(i64 %img, i32 %idx, i32 %x, i32 %y, i32 %z) nounwind alwaysinline {
entry:
  %0 = tail call {i32,i32,i32,i32} @llvm.nvvm.suld.3d.array.v4i32.clamp(i64 %img, i32 %idx, i32 %x, i32 %y, i32 %z);
  %1 = tail call <4 x i32> @__clc_struct32_to_vector({i32,i32,i32,i32} %0) ret <4 x i32> %1
}
