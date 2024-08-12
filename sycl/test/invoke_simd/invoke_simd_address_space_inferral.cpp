// RUN: %clangxx -fsycl -fsycl-device-only -Xclang -fsycl-allow-func-ptr -S %s -o %t.ll
// RUN: sycl-post-link -O2 -device-globals -properties -spec-const=native -split=auto -emit-only-kernels-as-entry-points -emit-param-info -symbols -emit-exported-symbols -emit-imported-symbols -lower-esimd -S %t.ll -o %t.table
// RUN: FileCheck %s -input-file=%t_0.ll

// The test validates proper address space inferral for a pointer passed to
// invoke_simd callee that is used for ESIMD API memory API

#include <sycl/detail/core.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/oneapi/experimental/invoke_simd.hpp>
#include <sycl/ext/oneapi/experimental/uniform.hpp>
#include <sycl/usm.hpp>

#include <functional>
#include <iostream>
#include <type_traits>

using namespace sycl::ext::oneapi::experimental;
using namespace sycl;
namespace esimd = sycl::ext::intel::esimd;

constexpr int VL = 32;

__attribute__((always_inline)) void ESIMD_CALLEE(float *A, float *B,
                                                 int i) SYCL_ESIMD_FUNCTION {
  esimd::simd<float, VL> a;
  a.copy_from(A + i);
  a.copy_to(B + i);
}

[[intel::device_indirectly_callable]] SYCL_EXTERNAL void __regcall SIMD_CALLEE1(
    float *A, float *B, int i) SYCL_ESIMD_FUNCTION {
  ESIMD_CALLEE(A, B, i);
}
bool test() {
  constexpr unsigned Size = 1024;
  constexpr unsigned GroupSize = 4 * VL;

  queue q;

  auto dev = q.get_device();
  float *A = malloc_shared<float>(Size, q);

  sycl::range<1> GlobalRange{Size};
  // Number of workitems in each workgroup.
  sycl::range<1> LocalRange{GroupSize};

  sycl::nd_range<1> Range(GlobalRange, LocalRange);

  try {
    auto e = q.submit([&](handler &cgh) {
      local_accessor<float, 1> LocalAcc(Size, cgh);
      cgh.parallel_for(Range, [=](nd_item<1> item) [[intel::reqd_sub_group_size(
                                  VL)]] {
        sycl::group<1> g = item.get_group();
        sycl::sub_group sg = item.get_sub_group();

        unsigned int i = g.get_group_id() * g.get_local_range() +
                         sg.get_group_id() * sg.get_max_local_range();

        invoke_simd(
            sg, SIMD_CALLEE1, uniform{A},
            uniform{LocalAcc.template get_multi_ptr<access::decorated::yes>()
                        .get()},
            uniform{i});
      });
    });
    e.wait();
  } catch (sycl::exception const &e) {
    std::cout << "SYCL exception caught: " << e.what() << '\n';
    sycl::free(A, q);
    return false;
  }

  sycl::free(A, q);

  return 0;
  // CHECK: addrspacecast ptr addrspace(4) %{{.*}} to ptr addrspace(1)
  // CHECK: addrspacecast ptr addrspace(4) %{{.*}} to ptr addrspace(3)
}

int main() {
  test();

  return 0;
}
