// RUN: %clangxx -fsycl -fsycl-device-only -c %s
// This test checks compile-time evaluation of functions from esimd_util.hpp

#include "CL/sycl.hpp"
#include "CL/sycl/INTEL/esimd.hpp"

static_assert(sycl::INTEL::gpu::detail::getNextPowerOf2<0>() == 0, "");
static_assert(sycl::INTEL::gpu::detail::getNextPowerOf2<1>() == 1, "");
static_assert(sycl::INTEL::gpu::detail::getNextPowerOf2<7>() == 8, "");
static_assert(sycl::INTEL::gpu::detail::getNextPowerOf2<1024>() == 1024, "");

static_assert(sycl::INTEL::gpu::detail::log2<0>() == 0, "");
static_assert(sycl::INTEL::gpu::detail::log2<1>() == 0, "");
static_assert(sycl::INTEL::gpu::detail::log2<7>() == 2, "");
static_assert(sycl::INTEL::gpu::detail::log2<1024 * 1024>() == 20, "");
