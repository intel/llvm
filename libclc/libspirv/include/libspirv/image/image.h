//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

_CLC_OVERLOAD _CLC_DECL float __spirv_ImageRead__Rfloat(image2d_t image,
                                                        int2 coord);
_CLC_OVERLOAD _CLC_DECL float __spirv_ImageRead__Rfloat(image2d_t image,
                                                        int4 coord);
_CLC_OVERLOAD _CLC_DECL float
__spirv_ImageRead__Rfloat(image2d_t image, int2 coord, int op1, int op2);
_CLC_OVERLOAD _CLC_DECL float
__spirv_ImageRead__Rfloat(image2d_t image, int4 coord, int op1, int op2);

_CLC_OVERLOAD _CLC_DECL float4 __spirv_ImageRead__Rfloat4(image1d_t image,
                                                          int coord);
_CLC_OVERLOAD _CLC_DECL float4 __spirv_ImageRead__Rfloat4(image1d_t image,
                                                          int2 coord);
_CLC_OVERLOAD _CLC_DECL float4 __spirv_ImageRead__Rfloat4(image2d_t image,
                                                          int2 coord);
_CLC_OVERLOAD _CLC_DECL float4 __spirv_ImageRead__Rfloat4(image2d_t image,
                                                          int4 coord);
_CLC_OVERLOAD _CLC_DECL float4 __spirv_ImageRead__Rfloat4(image3d_t image,
                                                          int4 coord);
_CLC_OVERLOAD _CLC_DECL float4 __spirv_ImageRead__Rfloat4(image2d_t image,
                                                          int2 coord, int op1,
                                                          int op2);
_CLC_OVERLOAD _CLC_DECL float4 __spirv_ImageRead__Rfloat4(image2d_t image,
                                                          int4 coord, int op1,
                                                          int op2);

_CLC_OVERLOAD _CLC_DECL float4 __spirv_ImageSampleExplicitcoord__Rfloat4(
    sampler_t sampler, float coord, int op1, float op2);
_CLC_OVERLOAD _CLC_DECL float4 __spirv_ImageSampleExplicitcoord__Rfloat4(
    sampler_t sampler, float2 coord, int op1, float op2);
_CLC_OVERLOAD _CLC_DECL float4 __spirv_ImageSampleExplicitcoord__Rfloat4(
    sampler_t sampler, float4 coord, int op1, float op2);
_CLC_OVERLOAD _CLC_DECL float4 __spirv_ImageSampleExplicitcoord__Rfloat4(
    sampler_t sampler, int coord, int op1, float op2);
_CLC_OVERLOAD _CLC_DECL float4 __spirv_ImageSampleExplicitcoord__Rfloat4(
    sampler_t sampler, int2 coord, int op1, float op2);
_CLC_OVERLOAD _CLC_DECL float4 __spirv_ImageSampleExplicitcoord__Rfloat4(
    sampler_t sampler, int4 coord, int op1, float op2);

_CLC_OVERLOAD _CLC_DECL int __spirv_ImageQueryFormat(image1d_t image);
_CLC_OVERLOAD _CLC_DECL int __spirv_ImageQueryFormat(image2d_t image);
_CLC_OVERLOAD _CLC_DECL int __spirv_ImageQueryFormat(image3d_t image);

_CLC_OVERLOAD _CLC_DECL int __spirv_ImageQueryOrder(image1d_t image);
_CLC_OVERLOAD _CLC_DECL int __spirv_ImageQueryOrder(image2d_t image);
_CLC_OVERLOAD _CLC_DECL int __spirv_ImageQueryOrder(image3d_t image);

_CLC_OVERLOAD _CLC_DECL int __spirv_ImageQuerySamples(image2d_t image);

_CLC_OVERLOAD _CLC_DECL uint __spirv_ImageQuerySizeLod_Ruint(image1d_t image,
                                                             int lod);

_CLC_OVERLOAD _CLC_DECL uint __spirv_ImageQuerySize_Ruint(image1d_t image);

_CLC_OVERLOAD _CLC_DECL uint2 __spirv_ImageQuerySizeLod_Ruint2(image1d_t image,
                                                               int lod);
_CLC_OVERLOAD _CLC_DECL uint2 __spirv_ImageQuerySizeLod_Ruint2(image2d_t image,
                                                               int lod);

_CLC_OVERLOAD _CLC_DECL uint3 __spirv_ImageQuerySizeLod_Ruint3(image2d_t image,
                                                               int lod);
_CLC_OVERLOAD _CLC_DECL uint3 __spirv_ImageQuerySizeLod_Ruint3(image3d_t image,
                                                               int lod);

_CLC_OVERLOAD _CLC_DECL ulong2
__spirv_ImageQuerySizeLod_Rulong2(image1d_t image, int lod);

_CLC_OVERLOAD _CLC_DECL ulong3
__spirv_ImageQuerySizeLod_Rulong3(image2d_t image, int lod);

_CLC_OVERLOAD _CLC_DECL void __spirv_ImageWrite(image1d_t image, int coord,
                                                float4 texel);
_CLC_OVERLOAD _CLC_DECL void __spirv_ImageWrite(image1d_t image, int coord,
                                                int4 texel);
_CLC_OVERLOAD _CLC_DECL void __spirv_ImageWrite(image1d_t image, int2 coord,
                                                float4 texel);
_CLC_OVERLOAD _CLC_DECL void __spirv_ImageWrite(image1d_t image, int2 coord,
                                                int4 texel);
_CLC_OVERLOAD _CLC_DECL void __spirv_ImageWrite(image2d_t image, int2 coord,
                                                float texel);
_CLC_OVERLOAD _CLC_DECL void __spirv_ImageWrite(image2d_t image, int2 coord,
                                                float4 texel);
_CLC_OVERLOAD _CLC_DECL void __spirv_ImageWrite(image2d_t image, int2 coord,
                                                int4 texel);
_CLC_OVERLOAD _CLC_DECL void __spirv_ImageWrite(image2d_t image, int4 coord,
                                                float texel);
_CLC_OVERLOAD _CLC_DECL void __spirv_ImageWrite(image2d_t image, int4 coord,
                                                float4 texel);
_CLC_OVERLOAD _CLC_DECL void __spirv_ImageWrite(image2d_t image, int4 coord,
                                                int4 texel);
_CLC_OVERLOAD _CLC_DECL void __spirv_ImageWrite(image3d_t image, int4 coord,
                                                float4 texel);
_CLC_OVERLOAD _CLC_DECL void __spirv_ImageWrite(image3d_t image, int4 coord,
                                                int4 texel);

#ifdef cl_khr_fp16
_CLC_OVERLOAD _CLC_DECL void __spirv_ImageWrite(image1d_t image, int coord,
                                                half4 texel);
_CLC_OVERLOAD _CLC_DECL void __spirv_ImageWrite(image1d_t image, int2 coord,
                                                half4 texel);
_CLC_OVERLOAD _CLC_DECL void __spirv_ImageWrite(image2d_t image, int2 coord,
                                                half4 texel);
_CLC_OVERLOAD _CLC_DECL void __spirv_ImageWrite(image2d_t image, int4 coord,
                                                half4 texel);
_CLC_OVERLOAD _CLC_DECL void __spirv_ImageWrite(image3d_t image, int4 coord,
                                                half4 texel);
#endif
