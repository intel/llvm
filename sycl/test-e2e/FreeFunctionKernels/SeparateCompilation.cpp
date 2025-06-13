// RUN: %clangxx -fsycl %s %S/SumKernel.cpp %S/ProductKernel.cpp -o %t.out
// RUN: %{run} %t.out

#include "ProductKernel.hpp"
#include "SumKernel.hpp"
#include <cassert>
#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/experimental/free_function_traits.hpp>
#include <sycl/kernel_bundle.hpp>

using namespace sycl;

// Add declarations again to test the compiler with multiple declarations of the
// same free function kernel in the translation unit.

SYCL_EXT_ONEAPI_FUNCTION_PROPERTY(
    (ext::oneapi::experimental::nd_range_kernel<1>))
void SumKernel::sum(accessor<int, 1> accA, accessor<int, 1> accB,
                    accessor<int, 1> result);

constexpr size_t SIZE = 16;

int main() {
  int data[SIZE];
  int result[SIZE];
  for (int i = 0; i < SIZE; ++i) {
    data[i] = i;
  }
  queue Q;
  kernel_bundle bundle =
      get_kernel_bundle<bundle_state::executable>(Q.get_context());
  kernel_id sumId = ext::oneapi::experimental::get_kernel_id<SumKernel::sum>();
  kernel_id productId = ext::oneapi::experimental::get_kernel_id<product>();
  kernel sumKernel = bundle.get_kernel(sumId);
  kernel productKernel = bundle.get_kernel(productId);

  {
    buffer<int, 1> databuf{&data[0], SIZE};
    buffer<int, 1> resultbuf{&result[0], SIZE};

    Q.submit([&](handler &h) {
      accessor<int, 1> accdata(databuf, h);
      accessor<int, 1> accresult(resultbuf, h);
      h.set_args(accdata, accdata, accresult);
      h.parallel_for(nd_range{{SIZE}, {SIZE}}, sumKernel);
    });
  }

  for (int i = 0; i < SIZE; ++i) {
    assert(result[i] == 2 * data[i]);
  }

  {
    buffer<int, 1> databuf{&data[0], SIZE};
    buffer<int, 1> resultbuf{&result[0], SIZE};

    Q.submit([&](handler &h) {
      accessor<int, 1> accdata(databuf, h);
      accessor<int, 1> accresult(resultbuf, h);
      h.set_args(accdata, accdata, accresult);
      h.parallel_for(nd_range{{SIZE}, {SIZE}}, productKernel);
    });
  }

  for (int i = 0; i < SIZE; ++i) {
    assert(result[i] == data[i] * data[i]);
  }

  return 0;
}
