#pragma once
#include <sycl/accessor.hpp>
#include <sycl/kernel_bundle.hpp>

namespace SumKernel {
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY(
    (sycl::ext::oneapi::experimental::nd_range_kernel<1>))
void sum(sycl::accessor<int, 1> accA, sycl::accessor<int, 1> accB,
         sycl::accessor<int, 1> result);
} // namespace SumKernel
