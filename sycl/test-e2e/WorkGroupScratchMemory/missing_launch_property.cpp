// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
//

// XFAIL: target-native-cpu
// XFAIL-TRACKER: https://github.com/intel/llvm/issues/20142

// This test checks that when a kernel uses the get_work_group_scratch_memory
// function and a corresponding work_group_scratch_size launch property has not
// been specified during submission of the kernel, a runtime exception occurs
// with error code errc::memory_allocation. The test exercises different ways of
// submission across a variety of kernels such as functors, lambdas, and free
// function kernels.

#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/experimental/enqueue_functions.hpp>
#include <sycl/ext/oneapi/experimental/free_function_traits.hpp>
#include <sycl/ext/oneapi/experimental/root_group.hpp>
#include <sycl/ext/oneapi/free_function_queries.hpp>
#include <sycl/ext/oneapi/work_group_static.hpp>
#include <sycl/group_barrier.hpp>
#include <sycl/kernel_bundle.hpp>

using namespace sycl;

namespace syclex = sycl::ext::oneapi::experimental;

template <typename T1> struct KernelWithRootGroup {
  T1 m_props;
  KernelWithRootGroup(T1 props) : m_props(props) {}

  void operator()(nd_item<1> Item) const {
    int *Ptr = reinterpret_cast<int *>(syclex::get_work_group_scratch_memory());
    auto Root = Item.ext_oneapi_get_root_group();
    sycl::group_barrier(Root);
  }
  auto get(syclex::properties_tag) const { return m_props; }
};

template <typename T1> struct KernelWithoutRootGroup {
  T1 m_props;
  KernelWithoutRootGroup(T1 props) : m_props(props) {}

  void operator()(nd_item<1> Item) const {
    int *Ptr = reinterpret_cast<int *>(syclex::get_work_group_scratch_memory());
  }
  auto get(syclex::properties_tag) const { return m_props; }
};

SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclex::nd_range_kernel<1>))
void freeFuncKernelWithRootGroup() {
  int *LocalMem =
      reinterpret_cast<int *>(syclex::get_work_group_scratch_memory());
  auto Root = sycl::ext::oneapi::this_work_item::get_nd_item<1>()
                  .ext_oneapi_get_root_group();
  sycl::group_barrier(Root);
}

SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclex::single_task_kernel))
void freeFuncKernelWithoutRootGroup() {
  int *LocalMem =
      reinterpret_cast<int *>(syclex::get_work_group_scratch_memory());
}

constexpr int SIZE = 32;

int main() {
  queue Q;
  nd_range ndr{range<1>(SIZE), range<1>(SIZE)};
  syclex::properties emptyProps{};
  syclex::properties rootsyncProps{syclex::use_root_sync};
  try {
    Q.submit([&](handler &cgh) {
      cgh.parallel_for(ndr, KernelWithoutRootGroup(emptyProps));
    });
    assert(false && "Expected exception not seen!");
  } catch (sycl::exception const &e) {
    assert(e.code() == errc::memory_allocation);
  }

  try {
    Q.submit([&](handler &cgh) {
      cgh.parallel_for(ndr, KernelWithRootGroup(rootsyncProps));
    });
    assert(false && "Expected exception not seen!");

  } catch (sycl::exception const &e) {
    assert(e.code() == errc::memory_allocation);
  }

  try {
    Q.submit([&](handler &cgh) {
      cgh.parallel_for(ndr, [=](nd_item<1> id) {
        int *Ptr =
            reinterpret_cast<int *>(syclex::get_work_group_scratch_memory());
      });
    });
    assert(false && "Expected exception not seen!");

  } catch (sycl::exception const &e) {
    assert(e.code() == errc::memory_allocation);
  }

  try {
    Q.submit([&](handler &cgh) {
      cgh.single_task([=]() {
        int *Ptr =
            reinterpret_cast<int *>(syclex::get_work_group_scratch_memory());
      });
    });
    assert(false && "Expected exception not seen!");

  } catch (sycl::exception const &e) {
    assert(e.code() == errc::memory_allocation);
  }

  auto freeFuncKernelWithRootGroupBndl =
      syclex::get_kernel_bundle<freeFuncKernelWithRootGroup,
                                sycl::bundle_state::executable>(
          Q.get_context());

  sycl::kernel RootGroupKern =
      freeFuncKernelWithRootGroupBndl
          .template ext_oneapi_get_kernel<freeFuncKernelWithRootGroup>();
  syclex::launch_config RootGroupConfig{ndr, rootsyncProps};
  syclex::launch_config EmptyConfig{ndr, emptyProps};

  try {
    syclex::nd_launch(Q, RootGroupConfig, RootGroupKern);
    assert(false && "Expected exception not seen!");

  } catch (sycl::exception const &e) {
    assert(e.code() == errc::memory_allocation);
  }

  auto freeFuncKernelWithoutRootGroupBndl =
      syclex::get_kernel_bundle<freeFuncKernelWithoutRootGroup,
                                sycl::bundle_state::executable>(
          Q.get_context());

  sycl::kernel NoRootGroupKern =
      freeFuncKernelWithoutRootGroupBndl
          .template ext_oneapi_get_kernel<freeFuncKernelWithoutRootGroup>();
  try {
    syclex::single_task(Q, NoRootGroupKern);
    assert(false && "Expected exception not seen!");

  } catch (sycl::exception const &e) {
    assert(e.code() == errc::memory_allocation);
  }

  try {
    syclex::single_task(
        Q, syclex::kernel_function<freeFuncKernelWithoutRootGroup>);
  } catch (sycl::exception const &e) {
    assert(e.code() == errc::memory_allocation);
  }

  try {
    syclex::nd_launch(Q, RootGroupConfig,
                      syclex::kernel_function<freeFuncKernelWithRootGroup>);
    assert(false && "Expected exception not seen!");

  } catch (sycl::exception const &e) {
    assert(e.code() == errc::memory_allocation);
  }

  try {
    Q.submit([&](sycl::handler &Cgh) {
      syclex::nd_launch(Cgh, RootGroupConfig,
                        syclex::kernel_function<freeFuncKernelWithRootGroup>);
    });
    assert(false && "Expected exception not seen!");

  } catch (sycl::exception const &e) {
    assert(e.code() == errc::memory_allocation);
  }

  try {
    Q.submit([&](sycl::handler &Cgh) {
      syclex::single_task(
          Cgh, syclex::kernel_function<freeFuncKernelWithoutRootGroup>);
    });
    assert(false && "Expected exception not seen!");
  } catch (sycl::exception const &e) {
    assert(e.code() == errc::memory_allocation);
  }
}
