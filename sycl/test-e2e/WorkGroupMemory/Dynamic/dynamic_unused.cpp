// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
//

// UNSUPPORTED: gpu-intel-gen12
// UNSUPPORTED-TRACKER: https://github.com/intel/llvm/issues/16072

#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/work_group_scratch_memory.hpp>
#include <sycl/usm.hpp>

constexpr size_t Size = 1024;
using DataType = int;

namespace sycl_ext = sycl::ext::oneapi::experimental;

int main() {
  sycl::queue queue;
  DataType *a = sycl::malloc_device<DataType>(Size, queue);
  DataType *b = sycl::malloc_device<DataType>(Size, queue);
  std::vector<DataType> a_host(Size, 1.0);
  std::vector<DataType> b_host(Size, -5.0);

  queue.copy(a_host.data(), a, Size).wait_and_throw();

  queue
      .submit([&](sycl::handler &cgh) {
        cgh.parallel_for(sycl::nd_range<1>({Size}, {Size}),
                         sycl_ext::properties{sycl_ext::work_group_scratch_size(
                             Size * sizeof(DataType))},
                         [=](sycl::nd_item<1> it) {
                           b[it.get_local_linear_id()] =
                               a[it.get_local_linear_id()];
                         });
      })
      .wait_and_throw();

  queue.copy(b, b_host.data(), Size).wait_and_throw();
  for (size_t i = 0; i < b_host.size(); i++) {
    assert(b_host[i] == a_host[i]);
  }
  sycl::free(a, queue);
  sycl::free(b, queue);
}
