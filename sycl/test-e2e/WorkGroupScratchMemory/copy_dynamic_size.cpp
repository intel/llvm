// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// UNSUPPORTED: gpu-intel-gen12
// UNSUPPORTED-TRACKER: https://github.com/intel/llvm/issues/16072

#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/work_group_scratch_memory.hpp>
#include <sycl/group_barrier.hpp>
#include <sycl/usm.hpp>

using DataType = int;

namespace sycl_ext = sycl::ext::oneapi::experimental;

void copy_via_smem(DataType *a, DataType *b, sycl::nd_item<1> it) {
  // one work-group copies data to shared memory from A
  // And then puts in back into B

  DataType *smem_ptr =
      reinterpret_cast<DataType *>(sycl_ext::get_work_group_scratch_memory());
  auto threadIdx_x = it.get_local_linear_id();

  smem_ptr[threadIdx_x] = a[threadIdx_x];
  sycl::group_barrier(it.get_group());

  b[threadIdx_x] = smem_ptr[threadIdx_x];
}

template <typename T> struct KernelFunctor {
  T m_props;
  DataType *m_a;
  DataType *m_b;
  KernelFunctor(T props, DataType *a, DataType *b)
      : m_props(props), m_a(a), m_b(b) {}

  void operator()(sycl::nd_item<1> it) const { copy_via_smem(m_a, m_b, it); }
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
