// REQUIRES: gpu, level_zero, aspect-usm_shared_allocations

// UNSUPPORTED: windows && gpu-intel-gen12
// UNSUPPORTED-TRACKER: https://github.com/intel/llvm/issues/21556

// RUN: %{build} -Wno-deprecated-declarations -o %t.out
// RUN: env SYCL_UR_TRACE=-1 UR_L0_DEBUG=1 %{run} %t.out 2>&1 | FileCheck %s

// This test verifies that the `cache_config` runtime launch property from
// sycl_ext_intel_cache_config is correctly propagated to the UR (observed
// via the `urKernelSetExecInfo` trace) across the main
// supported kernel invocation paths:
//   * `sycl_ext_oneapi_enqueue_functions` `parallel_for` and `nd_launch`.
//   * `sycl_ext_oneapi_free_function_kernels` via `nd_launch`.
//   * A negative case: the property is not set when no property is passed.
//   * Deprecated APIs: SYCL `queue::single_task`, `queue::parallel_for` and
//   `queue::parallel_for_work_group` with a property list.

#include <numeric>
#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/experimental/enqueue_functions.hpp>
#include <sycl/ext/oneapi/free_function_queries.hpp>
#include <sycl/kernel_bundle.hpp>
#include <sycl/properties/all_properties.hpp>
#include <sycl/reduction.hpp>
#include <sycl/usm.hpp>

using namespace sycl;
namespace syclext = sycl::ext::oneapi;
namespace syclexp = sycl::ext::oneapi::experimental;
namespace intelexp = sycl::ext::intel::experimental;

struct KernelFunctor {
  void operator()() const {}
};

struct KernelFunctorND {
  void operator()(nd_item<2> i) const {}
};

struct NegativeKernelFunctor {
  void operator()(nd_item<2> i) const {}
};

struct RangeKernelFunctor {
  void operator()(id<2> i) const {}
};

struct WorkGroupFunctor {
  void operator()(group<1> g) const {
    g.parallel_for_work_item([&](h_item<1>) {});
  }
};

SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<1>))
void free_function_kernel(int *) {}

template <typename T1> struct ReductionKernelFunctor {
  T1 mInput_values;
  ReductionKernelFunctor(T1 &Input_values) : mInput_values(Input_values) {}

  template <typename sumT> void operator()(id<1> idx, sumT &sum) const {
    sum += mInput_values[idx];
  }
};

int main() {
  sycl::queue q{sycl::property::queue::in_order{}};

  syclexp::properties large_slm_props{
      intelexp::cache_config{intelexp::large_slm}};
  syclexp::properties large_data_props{
      intelexp::cache_config{intelexp::large_data}};

  // Recommended APIs with launch_config.

  // CHECK: enqueue_functions::parallel_for with launch_config + large_slm
  // CHECK: urKernelSetExecInfo{{.*}}UR_KERNEL_CACHE_CONFIG_LARGE_SLM
  std::cout << "enqueue_functions::parallel_for with launch_config + large_slm"
            << std::endl;
  syclexp::parallel_for(
      q, syclexp::launch_config{sycl::range<1>{16}, large_slm_props},
      [=](sycl::item<1>) {});
  q.wait();

  // CHECK: enqueue_functions::nd_launch with launch_config + large_data
  // CHECK: urKernelSetExecInfo{{.*}}UR_KERNEL_CACHE_CONFIG_LARGE_DATA
  std::cout << "enqueue_functions::nd_launch with launch_config + large_data"
            << std::endl;
  syclexp::nd_launch(
      q,
      syclexp::launch_config{
          sycl::nd_range<1>{sycl::range<1>{16}, sycl::range<1>{4}},
          large_data_props},
      [=](sycl::nd_item<1>) {});
  q.wait();

  // CHECK: free function kernel via nd_launch + large_slm
  // CHECK: urKernelSetExecInfo{{.*}}UR_KERNEL_CACHE_CONFIG_LARGE_SLM
  std::cout << "free function kernel via nd_launch + large_slm" << std::endl;
  int *ptr = sycl::malloc_shared<int>(16, q);
  syclexp::nd_launch(
      q,
      syclexp::launch_config{
          sycl::nd_range<1>{sycl::range<1>{16}, sycl::range<1>{4}},
          large_slm_props},
      syclexp::kernel_function<free_function_kernel>, ptr);
  q.wait();

  // Same kernel launched twice with different cache_config values.
  // CHECK: kernel_bundle: same kernel with large_slm then large_data
  // CHECK: urKernelSetExecInfo{{.*}}UR_KERNEL_CACHE_CONFIG_LARGE_SLM
  // CHECK: urKernelSetExecInfo{{.*}}UR_KERNEL_CACHE_CONFIG_LARGE_DATA
  std::cout << "kernel_bundle: same kernel with large_slm then large_data"
            << std::endl;
  {
    sycl::context ctxt = q.get_context();
    auto bundle =
        syclexp::get_kernel_bundle<free_function_kernel,
                                   sycl::bundle_state::executable>(ctxt);
    sycl::kernel k = bundle.ext_oneapi_get_kernel<free_function_kernel>();
    sycl::nd_range<1> ndr{sycl::range<1>{16}, sycl::range<1>{4}};

    syclexp::nd_launch(q, syclexp::launch_config{ndr, large_slm_props}, k, ptr);
    syclexp::nd_launch(q, syclexp::launch_config{ndr, large_data_props}, k,
                       ptr);
    q.wait();
  }
  sycl::free(ptr, q);

  // Negative case: launch with no cache_config property.
  // CHECK: negative case with no cache_config
  // CHECK-NOT: urKernelSetExecInfo{{.*}}UR_KERNEL_CACHE_CONFIG
  std::cout << "negative case with no cache_config" << std::endl;
  syclexp::nd_launch(q,
                     syclexp::launch_config{sycl::nd_range<1>{
                         sycl::range<1>{16}, sycl::range<1>{4}}},
                     [=](sycl::nd_item<1>) {});
  q.wait();

  // Depracated APIs.

  // CHECK: KernelFunctor single_task
  // CHECK: urKernelSetExecInfo{{.*}}UR_KERNEL_CACHE_CONFIG_LARGE_SLM
  std::cout << "KernelFunctor single_task" << std::endl;
  q.single_task(large_slm_props, KernelFunctor{}).wait();

  // CHECK: KernelFunctor parallel_for
  // CHECK: urKernelSetExecInfo{{.*}}UR_KERNEL_CACHE_CONFIG_LARGE_DATA
  std::cout << "KernelFunctor parallel_for" << std::endl;
  q.parallel_for(nd_range<2>{range<2>(4, 4), range<2>(2, 2)}, large_data_props,
                 KernelFunctorND{})
      .wait();

  // CHECK: parallel_for with sycl::range
  // CHECK: urKernelSetExecInfo{{.*}}UR_KERNEL_CACHE_CONFIG_LARGE_DATA
  std::cout << "parallel_for with sycl::range" << std::endl;
  q.parallel_for(range<2>{16, 16}, large_data_props, RangeKernelFunctor{})
      .wait();

  // CHECK: parallel_for_work_group(range, func)
  // CHECK: urKernelSetExecInfo{{.*}}UR_KERNEL_CACHE_CONFIG_LARGE_SLM
  std::cout << "parallel_for_work_group(range, func)" << std::endl;
  q.submit([&](handler &cgh) {
     cgh.parallel_for_work_group<class hpar_range>(range<1>(8), large_slm_props,
                                                   WorkGroupFunctor{});
   }).wait();
  ;

  // CHECK: parallel_for_work_group(range, range, func)
  // CHECK: urKernelSetExecInfo{{.*}}UR_KERNEL_CACHE_CONFIG_LARGE_SLM
  std::cout << "parallel_for_work_group(range, range, func)" << std::endl;
  q.submit([&](handler &cgh) {
     cgh.parallel_for_work_group<class hpar_range_range>(
         range<1>(8), range<1>(4), large_slm_props, WorkGroupFunctor{});
   }).wait();
  ;

  buffer<int> values_buf{1024};
  {
    host_accessor a{values_buf};
    std::iota(a.begin(), a.end(), 0);
  }

  int sum_result = 0;
  buffer<int> sum_buf{&sum_result, 1};

  // CHECK: parallel_for with reduction
  // CHECK: urKernelSetExecInfo{{.*}}UR_KERNEL_CACHE_CONFIG_LARGE_SLM
  std::cout << "parallel_for with reduction" << std::endl;
  q.submit([&](handler &cgh) {
     auto input_values = values_buf.get_access<access_mode::read>(cgh);
     auto sum_reduction = reduction(sum_buf, cgh, std::plus<>());
     cgh.parallel_for(range<1>{1024}, large_slm_props, sum_reduction,
                      ReductionKernelFunctor(input_values));
   }).wait();
  ;

  // CHECK: negative parallel_for with KernelFunctor
  // CHECK-NOT: urKernelSetExecInfo{{.*}}UR_KERNEL_CACHE_CONFIG
  std::cout << "negative parallel_for with KernelFunctor" << std::endl;
  q.parallel_for(nd_range<2>{range<2>(4, 4), range<2>(2, 2)},
                 NegativeKernelFunctor{})
      .wait();

  return 0;
}
