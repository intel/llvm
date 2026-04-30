//==-------- spirv_ops_image.hpp --- SPIRV image operations ---------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/__spirv/spirv_ops_builtin_decls.hpp>

#ifdef __SYCL_DEVICE_ONLY__

template <typename RetT, typename ImageT>
extern __DPCPP_SYCL_EXTERNAL RetT __spirv_ImageQueryFormat(ImageT);

template <typename RetT, typename ImageT>
extern __DPCPP_SYCL_EXTERNAL RetT __spirv_ImageQueryOrder(ImageT);

template <typename RetT, typename ImageT>
extern __DPCPP_SYCL_EXTERNAL RetT __spirv_ImageQuerySize(ImageT);

template <typename ImageT, typename CoordT, typename ValT>
extern __DPCPP_SYCL_EXTERNAL void __spirv_ImageWrite(ImageT, CoordT, ValT);

template <class RetT, typename ImageT, typename TempArgT>
extern __DPCPP_SYCL_EXTERNAL RetT __spirv_ImageRead(ImageT, TempArgT);

template <class RetT, typename ImageT, typename TempArgT>
extern __DPCPP_SYCL_EXTERNAL RetT __spirv_ImageFetch(ImageT, TempArgT);

template <class RetT, typename ImageT, typename TempArgT>
extern __DPCPP_SYCL_EXTERNAL RetT __spirv_SampledImageFetch(ImageT, TempArgT);

template <class RetT, typename ImageT, typename TempArgT>
extern __DPCPP_SYCL_EXTERNAL RetT __spirv_ImageArrayFetch(ImageT, TempArgT,
                                                          int);

template <class RetT, typename ImageT, typename TempArgT>
extern __DPCPP_SYCL_EXTERNAL RetT __spirv_SampledImageArrayFetch(ImageT,
                                                                 TempArgT, int);

template <class RetT, typename ImageT, typename TempArgT>
extern __DPCPP_SYCL_EXTERNAL RetT __spirv_SampledImageGather(ImageT, TempArgT,
                                                             unsigned);

template <class RetT, typename ImageT, typename TempArgT>
extern __DPCPP_SYCL_EXTERNAL RetT __spirv_ImageArrayRead(ImageT, TempArgT, int);

template <typename ImageT, typename CoordT, typename ValT>
extern __DPCPP_SYCL_EXTERNAL void __spirv_ImageArrayWrite(ImageT, CoordT, int,
                                                          ValT);

template <typename ImageT, typename SampledType>
extern __DPCPP_SYCL_EXTERNAL SampledType __spirv_SampledImage(ImageT,
                                                              __ocl_sampler_t);

template <typename SampledType, typename TempRetT, typename TempArgT>
extern __DPCPP_SYCL_EXTERNAL TempRetT
__spirv_ImageSampleExplicitLod(SampledType, TempArgT, int, float);

template <typename SampledType, typename TempRetT, typename TempArgT>
extern __DPCPP_SYCL_EXTERNAL TempRetT
__spirv_ImageSampleExplicitLod(SampledType, TempArgT, int, TempArgT, TempArgT);

template <typename SampledType, typename TempRetT, typename TempArgT>
extern __DPCPP_SYCL_EXTERNAL TempRetT __spirv_ImageSampleCubemap(SampledType,
                                                                 TempArgT);

template <typename RetT, class HandleT>
extern __DPCPP_SYCL_EXTERNAL RetT __spirv_ConvertHandleToImageINTEL(HandleT);

template <typename RetT, class HandleT>
extern __DPCPP_SYCL_EXTERNAL RetT __spirv_ConvertHandleToSamplerINTEL(HandleT);

template <typename RetT, class HandleT>
extern __DPCPP_SYCL_EXTERNAL
    RetT __spirv_ConvertHandleToSampledImageINTEL(HandleT);

#endif
