// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
//

// UNSUPPORTED: gpu-intel-gen12
// UNSUPPORTED-TRACKER: https://github.com/intel/llvm/issues/16072

#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/work_group_scratch_memory.hpp>
#include <sycl/usm.hpp>

using DataType = int;

namespace sycl_ext = sycl::ext::oneapi::experimental;

template <typename T> struct KernelFunctor {
  T m_props;
  DataType *m_a;
  DataType *m_b;
  KernelFunctor(T props, DataType *a, DataType *b)
      : m_props(props), m_a(a), m_b(b) {}

  void operator()(sycl::nd_item<1> it) const {
    m_b[it.get_local_linear_id()] = m_a[it.get_local_linear_id()];
  }
  auto get(sycl_ext::properties_tag) const { return m_props; }
};

int main() {
  sycl::queue queue;
  auto size = std::min(
      queue.get_device().get_info<sycl::info::device::max_work_group_size>(),
      size_t{1024});

  DataType *a = sycl::malloc_device<DataType>(size, queue);
  DataType *b = sycl::malloc_device<DataType>(size, queue);
  std::vector<DataType> a_host(size, 1.0);
  std::vector<DataType> b_host(size, -5.0);

  queue.copy(a_host.data(), a, size).wait_and_throw();

  queue
      .submit([&](sycl::handler &cgh) {
        cgh.parallel_for(
            sycl::nd_range<1>({size}, {size}),
            KernelFunctor(
                sycl_ext::properties{
                    sycl_ext::work_group_scratch_size(size * sizeof(DataType))},
                a, b));
      })
      .wait_and_throw();

  queue.copy(b, b_host.data(), size).wait_and_throw();
  for (size_t i = 0; i < b_host.size(); i++) {
    assert(b_host[i] == a_host[i]);
  }
  sycl::free(a, queue);
  sycl::free(b, queue);
}
