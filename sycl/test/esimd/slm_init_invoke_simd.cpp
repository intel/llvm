// This test verifies call to slm_init from a function called through
// invoke_simd triggers an error.

// RUN: not %clangxx -fsycl -fno-sycl-device-code-split-esimd -Xclang -fsycl-allow-func-ptr  %s 2>&1 | FileCheck  %s

#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/oneapi/experimental/invoke_simd.hpp>
#include <sycl/ext/oneapi/experimental/uniform.hpp>
#include <sycl/sycl.hpp>

#include <functional>
#include <iostream>
#include <type_traits>

using namespace sycl::ext::oneapi::experimental;
using namespace sycl;
namespace esimd = sycl::ext::intel::esimd;

SYCL_EXTERNAL
[[intel::device_indirectly_callable]] void __regcall SIMD_CALLEE_VOID()
    SYCL_ESIMD_FUNCTION {
  esimd::slm_init<1024>();
}

int main() {
  queue Q;
  nd_range<1> NDR{range<1>{2}, range<1>{2}};
  Q.parallel_for(NDR, [=](nd_item<1> NDI) [[intel::reqd_sub_group_size(16)]] {
     sub_group sg = NDI.get_sub_group();
     invoke_simd(sg, SIMD_CALLEE_VOID);
   }).wait();
  return 0;
}
// CHECK: slm_init must be called directly from ESIMD kernel.