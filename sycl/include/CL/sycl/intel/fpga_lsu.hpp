//==-------------- fpga_lsu.hpp --- SYCL FPGA Reg Extensions ---------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#pragma once

#include <CL/sycl/INTEL/fpga_lsu.hpp>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace intel {
constexpr uint8_t BURST_COALESCE = INTEL::BURST_COALESCE;
constexpr uint8_t CACHE = INTEL::CACHE;
constexpr uint8_t STATICALLY_COALESCE = INTEL::STATICALLY_COALESCE;
constexpr uint8_t PREFETCH = INTEL::PREFETCH;

template <int32_t N> using cache = INTEL::cache<N>;

template <bool B> using burst_coalesce = INTEL::burst_coalesce<B>;
template <bool B> using prefetch = INTEL::prefetch<B>;
template <bool B> using statically_coalesce = INTEL::statically_coalesce<B>;

template <class... mem_access_params>
using lsu = INTEL::lsu<mem_access_params...>;
} // namespace intel
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
