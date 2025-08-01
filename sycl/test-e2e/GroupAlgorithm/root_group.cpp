// Fails with opencl non-cpu, enable when fixed.
// XFAIL: (opencl && !cpu)
// XFAIL-TRACKER: https://github.com/intel/llvm/issues/14641

// RUN: %{build} -I . -o %t.out %if target-nvidia %{ -Xsycl-target-backend=nvptx64-nvidia-cuda --cuda-gpu-arch=sm_70 %}
// RUN: %{run} %t.out

// Disabled temporarily while investigation into the failure is ongoing.
// UNSUPPORTED: gpu-intel-dg2

#include <cassert>
#include <cstdlib>
#include <type_traits>

#include <sycl/builtins.hpp>
#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/experimental/root_group.hpp>
#include <sycl/group_barrier.hpp>
#include <sycl/kernel_bundle.hpp>

static constexpr int WorkGroupSize = 32;

void testFeatureMacro() {
  static_assert(SYCL_EXT_ONEAPI_ROOT_GROUP == 1,
                "SYCL_EXT_ONEAPI_ROOT_GROUP must have a value of 1");
}

void testQueriesAndProperties() {
  sycl::queue q;
  const auto bundle =
      sycl::get_kernel_bundle<sycl::bundle_state::executable>(q.get_context());
  const auto kernel = bundle.get_kernel<class QueryKernel>();
  const auto local_range = sycl::range<1>(1);
  const auto maxWGs =
      kernel
          .ext_oneapi_get_info<sycl::ext::oneapi::experimental::info::
                                   kernel_queue_specific::max_num_work_groups>(
              q, local_range, 0);
  const auto wgRange = sycl::range<3>{WorkGroupSize, 1, 1};
  const auto maxWGsWithLimits =
      kernel
          .ext_oneapi_get_info<sycl::ext::oneapi::experimental::info::
                                   kernel_queue_specific::max_num_work_groups>(
              q, wgRange, wgRange.size() * sizeof(int));
  struct TestKernel0 {
    void operator()() const {}
    auto get(sycl::ext::oneapi::experimental::properties_tag) const {
      return sycl::ext::oneapi::experimental::properties{
          sycl::ext::oneapi::experimental::use_root_sync};
    }
  };
  q.single_task<class QueryKernel>(TestKernel0{});

  static auto check_max_num_work_group_sync = [](auto Result) {
    static_assert(std::is_same_v<std::remove_cv_t<decltype(Result)>, size_t>,
                  "max_num_work_group_sync query must return size_t");
    assert(Result >= 1 && "max_num_work_group_sync query failed");
  };
  check_max_num_work_group_sync(maxWGs);
  check_max_num_work_group_sync(maxWGsWithLimits);
}

template <typename T> struct TestKernel1 {
  T m_data;
  TestKernel1(T &data_) : m_data(data_) {}
  void operator()(sycl::nd_item<1> it) const {
    volatile float X = 1.0f;
    volatile float Y = 1.0f;
    auto root = it.ext_oneapi_get_root_group();
    m_data[root.get_local_id()] = root.get_local_id();
    sycl::group_barrier(root);
    // Delay half of the workgroups with extra work to check that the barrier
    // synchronizes the whole device.
    if (it.get_group(0) % 2 == 0) {
      X += sycl::sin(X);
      Y += sycl::cos(Y);
    }
    root = sycl::ext::oneapi::experimental::this_work_item::get_root_group<1>();
    int sum = m_data[root.get_local_id()] +
              m_data[root.get_local_range() - root.get_local_id() - 1];
    sycl::group_barrier(root);
    m_data[root.get_local_id()] = sum;
  }
  auto get(sycl::ext::oneapi::experimental::properties_tag) const {
    return sycl::ext::oneapi::experimental::properties{
        sycl::ext::oneapi::experimental::use_root_sync};
    ;
  }
};

void testRootGroup() {
  sycl::queue q;
  const auto bundle =
      sycl::get_kernel_bundle<sycl::bundle_state::executable>(q.get_context());
  const auto kernel = bundle.get_kernel<class RootGroupKernel>();
  const auto maxWGs =
      kernel
          .ext_oneapi_get_info<sycl::ext::oneapi::experimental::info::
                                   kernel_queue_specific::max_num_work_groups>(
              q, WorkGroupSize, 0);
  const auto props = sycl::ext::oneapi::experimental::properties{
      sycl::ext::oneapi::experimental::use_root_sync};
  sycl::buffer<int> dataBuf{sycl::range{maxWGs * WorkGroupSize}};
  const auto range = sycl::nd_range<1>{maxWGs * WorkGroupSize, WorkGroupSize};
  q.submit([&](sycl::handler &h) {
    sycl::accessor data{dataBuf, h};
    h.parallel_for<class RootGroupKernel>(range, TestKernel1(data));
  });
  sycl::host_accessor data{dataBuf};
  const int workItemCount = static_cast<int>(range.get_global_range().size());
  for (int i = 0; i < workItemCount; i++) {
    assert(data[i] == (workItemCount - 1));
  }
}

template <typename T> struct TestKernel2 {
  T m_testResults;
  TestKernel2(T &testResults_) : m_testResults(testResults_) {}
  void operator()(sycl::nd_item<1> it) const {
    const auto root = it.ext_oneapi_get_root_group();
    if (root.leader() || root.get_local_id() == 3) {
      m_testResults[0] = root.get_group_id() == sycl::id<1>(0);
      m_testResults[1] = root.leader() ? root.get_local_id() == sycl::id<1>(0)
                                       : root.get_local_id() == sycl::id<1>(3);
      m_testResults[2] = root.get_group_range() == sycl::range<1>(1);
      m_testResults[3] = root.get_local_range() == it.get_global_range();
      m_testResults[4] = root.get_max_local_range() == root.get_local_range();
      m_testResults[5] = root.get_group_linear_id() == 0;
      m_testResults[6] =
          root.get_local_linear_id() == root.get_local_id().get(0);
      m_testResults[7] = root.get_group_linear_range() == 1;
      m_testResults[8] =
          root.get_local_linear_range() == root.get_local_range().size();
    }
  }
  auto get(sycl::ext::oneapi::experimental::properties_tag) const {
    return sycl::ext::oneapi::experimental::properties{
        sycl::ext::oneapi::experimental::use_root_sync};
  }
};

void testRootGroupFunctions() {
  sycl::queue q;
  const auto bundle =
      sycl::get_kernel_bundle<sycl::bundle_state::executable>(q.get_context());
  const auto kernel = bundle.get_kernel<class RootGroupFunctionsKernel>();
  const auto maxWGs =
      kernel
          .ext_oneapi_get_info<sycl::ext::oneapi::experimental::info::
                                   kernel_queue_specific::max_num_work_groups>(
              q, WorkGroupSize, 0);
  constexpr int testCount = 9;
  sycl::buffer<bool> testResultsBuf{sycl::range{testCount}};
  const auto range = sycl::nd_range<1>{maxWGs * WorkGroupSize, WorkGroupSize};
  q.submit([&](sycl::handler &h) {
    sycl::accessor testResults{testResultsBuf, h};
    h.parallel_for<class RootGroupFunctionsKernel>(range,
                                                   TestKernel2(testResults));
  });
  sycl::host_accessor testResults{testResultsBuf};
  for (int i = 0; i < testCount; i++) {
    assert(testResults[i]);
  }
}

int main() {
  testFeatureMacro();
  testQueriesAndProperties();
  testRootGroup();
  testRootGroupFunctions();
  return EXIT_SUCCESS;
}
