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
  DataType *a = sycl::malloc_device<DataType>(Size, queue);
  DataType *b = sycl::malloc_device<DataType>(Size, queue);
  std::vector<DataType> a_host(Size, 1.0);
  std::vector<DataType> b_host(Size, -5.0);

  queue.copy(a_host.data(), a, Size).wait_and_throw();

  queue
      .submit([&](sycl::handler &cgh) {
        cgh.parallel_for(
            sycl::nd_range<1>({Size}, {Size}),
            KernelFunctor(
                sycl_ext::properties{
                    sycl_ext::work_group_scratch_size(Size * sizeof(DataType))},
                a, b));
      })
      .wait_and_throw();

  queue.copy(b, b_host.data(), Size).wait_and_throw();
  for (size_t i = 0; i < b_host.size(); i++) {
    assert(b_host[i] == a_host[i]);
  }
  sycl::free(a, queue);
  sycl::free(b, queue);
}
