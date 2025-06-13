#include "ProductKernel.hpp"
#include "SumKernel.hpp"
#include <cassert>
#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/experimental/free_function_traits.hpp>
#include <sycl/kernel_bundle.hpp>

using namespace sycl;

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
  kernel_id sumId = ext::oneapi::experimental::get_kernel_id<sum>();
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
