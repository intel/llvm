//==------ spirv_ops_subgroup.hpp --- SPIRV subgroup operations -----------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/__spirv/spirv_ops_builtin_decls.hpp>

#ifdef __SYCL_DEVICE_ONLY__

template <typename dataT>
__SYCL_CONVERGENT__ extern __DPCPP_SYCL_EXTERNAL dataT
__spirv_SubgroupBlockReadINTEL(const __attribute__((opencl_global))
                               uint8_t *Ptr) noexcept;

template <typename dataT>
__SYCL_CONVERGENT__ extern __DPCPP_SYCL_EXTERNAL dataT
__spirv_SubgroupBlockReadINTEL(const __attribute__((opencl_global))
                               uint16_t *Ptr) noexcept;

template <typename dataT>
__SYCL_CONVERGENT__ extern __DPCPP_SYCL_EXTERNAL dataT
__spirv_SubgroupBlockReadINTEL(const __attribute__((opencl_global))
                               uint32_t *Ptr) noexcept;

template <typename dataT>
__SYCL_CONVERGENT__ extern __DPCPP_SYCL_EXTERNAL dataT
__spirv_SubgroupBlockReadINTEL(const __attribute__((opencl_global))
                               uint64_t *Ptr) noexcept;

template <typename dataT>
__SYCL_CONVERGENT__ extern __DPCPP_SYCL_EXTERNAL dataT
__spirv_SubgroupBlockReadINTEL(const __attribute__((opencl_local))
                               uint8_t *Ptr) noexcept;

template <typename dataT>
__SYCL_CONVERGENT__ extern __DPCPP_SYCL_EXTERNAL dataT
__spirv_SubgroupBlockReadINTEL(const __attribute__((opencl_local))
                               uint16_t *Ptr) noexcept;

template <typename dataT>
__SYCL_CONVERGENT__ extern __DPCPP_SYCL_EXTERNAL dataT
__spirv_SubgroupBlockReadINTEL(const __attribute__((opencl_local))
                               uint32_t *Ptr) noexcept;

template <typename dataT>
__SYCL_CONVERGENT__ extern __DPCPP_SYCL_EXTERNAL dataT
__spirv_SubgroupBlockReadINTEL(const __attribute__((opencl_local))
                               uint64_t *Ptr) noexcept;

#endif
