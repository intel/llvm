// RUN: %clangxx -fsycl -fsycl-device-only -c %s
// This test checks compile-time evaluation of functions from esimd_util.hpp

#include "CL/sycl.hpp"
#include "sycl/ext/intel/esimd.hpp"

using namespace sycl::ext::intel::esimd;
using namespace sycl::ext::intel::esimd::detail;

static_assert(getNextPowerOf2<0>() == 0, "");
static_assert(getNextPowerOf2<1>() == 1, "");
static_assert(getNextPowerOf2<7>() == 8, "");
static_assert(getNextPowerOf2<1024>() == 1024, "");

static_assert(log2<0>() == 0, "");
static_assert(log2<1>() == 0, "");
static_assert(log2<7>() == 2, "");
static_assert(log2<1024 * 1024>() == 20, "");

using BaseTy = simd<float, 4>;
using RegionTy = region1d_t<float, 2, 1>;
using RegionTy1 = region1d_scalar_t<float>;
static_assert(!is_simd_view_type_v<simd_view_impl<BaseTy, RegionTy>>, "");
static_assert(is_simd_view_type_v<simd_view<BaseTy, RegionTy>>, "");
static_assert(is_simd_view_type_v<simd_view<BaseTy, RegionTy1>>, "");
static_assert(!is_simd_view_type_v<BaseTy>, "");
