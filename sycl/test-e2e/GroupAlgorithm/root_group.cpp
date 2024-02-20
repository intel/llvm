// RUN: %{build} -I . -o %t.out
// RUN: %{run} %t.out

#include <cassert>
#include <cstdlib>
#include <type_traits>

#include <sycl/ext/oneapi/experimental/root_group.hpp>
#include <sycl/sycl.hpp>

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
  const auto maxWGs = kernel.ext_oneapi_get_info<
      sycl::ext::oneapi::experimental::info::kernel_queue_specific::
          max_num_work_group_sync>(q);
  const auto props = sycl::ext::oneapi::experimental::properties{
      sycl::ext::oneapi::experimental::use_root_sync};
  q.single_task<class QueryKernel>(props, []() {});
  static_assert(std::is_same_v<std::remove_cv<decltype(maxWGs)>::type, size_t>,
                "max_num_work_group_sync query must return size_t");
  assert(maxWGs >= 1 && "max_num_work_group_sync query failed");
}

void testRootGroup() {
  sycl::queue q;
  const auto bundle =
      sycl::get_kernel_bundle<sycl::bundle_state::executable>(q.get_context());
  const auto kernel = bundle.get_kernel<class RootGroupKernel>();
  const auto maxWGs = kernel.ext_oneapi_get_info<
      sycl::ext::oneapi::experimental::info::kernel_queue_specific::
          max_num_work_group_sync>(q);
  const auto props = sycl::ext::oneapi::experimental::properties{
      sycl::ext::oneapi::experimental::use_root_sync};
  sycl::buffer<int> dataBuf{sycl::range{maxWGs * WorkGroupSize}};
  const auto range = sycl::nd_range<1>{maxWGs * WorkGroupSize, WorkGroupSize};
  q.submit([&](sycl::handler &h) {
    sycl::accessor data{dataBuf, h};
    h.parallel_for<class RootGroupKernel>(
        range, props, [=](sycl::nd_item<1> it) {
          auto root = it.ext_oneapi_get_root_group();
          data[root.get_local_id()] = root.get_local_id();
          sycl::group_barrier(root);

          root =
              sycl::ext::oneapi::experimental::this_kernel::get_root_group<1>();
          int sum = data[root.get_local_id()] +
                    data[root.get_local_range() - root.get_local_id() - 1];
          sycl::group_barrier(root);
          data[root.get_local_id()] = sum;
        });
  });
  sycl::host_accessor data{dataBuf};
  const int workItemCount = static_cast<int>(range.get_global_range().size());
  for (int i = 0; i < workItemCount; i++) {
    assert(data[i] == (workItemCount - 1));
  }
}

void testRootGroupFunctions() {
  sycl::queue q;
  const auto bundle =
      sycl::get_kernel_bundle<sycl::bundle_state::executable>(q.get_context());
  const auto kernel = bundle.get_kernel<class RootGroupFunctionsKernel>();
  const auto maxWGs = kernel.ext_oneapi_get_info<
      sycl::ext::oneapi::experimental::info::kernel_queue_specific::
          max_num_work_group_sync>(q);
  const auto props = sycl::ext::oneapi::experimental::properties{
      sycl::ext::oneapi::experimental::use_root_sync};

  constexpr int testCount = 22;
  sycl::buffer<bool> testResultsBuf{sycl::range{testCount}};
  const auto range = sycl::nd_range<1>{maxWGs * WorkGroupSize, WorkGroupSize};
  q.submit([&](sycl::handler &h) {
    sycl::accessor testResults{testResultsBuf, h};
    h.parallel_for<class RootGroupFunctionsKernel>(
        range, props, [=](sycl::nd_item<1> it) {
          const auto root = it.ext_oneapi_get_root_group();
          if (root.leader() || root.get_local_id() == 3) {
            testResults[0] = root.get_group_id() == sycl::id<1>(0);
            testResults[1] = root.leader()
                                 ? root.get_local_id() == sycl::id<1>(0)
                                 : root.get_local_id() == sycl::id<1>(3);
            testResults[2] = root.get_group_range() == sycl::range<1>(1);
            testResults[3] =
                root.get_local_range() == sycl::range<1>(WorkGroupSize);
            testResults[4] =
                root.get_max_local_range() == sycl::range<1>(WorkGroupSize);
            testResults[5] = root.get_group_linear_id() == 0;
            testResults[6] =
                root.get_local_linear_id() == root.get_local_id().get(0);
            testResults[7] = root.get_group_linear_range() == 1;
            testResults[8] = root.get_local_linear_range() == WorkGroupSize;

            const auto child =
                sycl::ext::oneapi::experimental::get_child_group(root);
            const auto grandchild =
                sycl::ext::oneapi::experimental::get_child_group(child);
            testResults[9] = child == it.get_group();
            static_assert(
                std::is_same_v<std::remove_cv<decltype(grandchild)>::type,
                               sycl::sub_group>,
                "get_child_group(sycl::group) must return a sycl::sub_group");
          }

          auto SG = it.get_sub_group();
          size_t SGSize = SG.get_local_linear_range();
          using execution_scope =
              sycl::ext::oneapi::experimental::execution_scope;

          if (root.leader()) {
            size_t NumSGPerWG = SG.get_group_linear_range();

            testResults[10] =
                root.template get_range<execution_scope::work_group>()[0] ==
                maxWGs;
            testResults[11] =
                root.template get_range<execution_scope::sub_group>()[0] ==
                NumSGPerWG * maxWGs;
            testResults[12] =
                root.template get_range<execution_scope::work_item>()[0] ==
                maxWGs * WorkGroupSize;

            testResults[13] =
                root.template get_linear_range<execution_scope::work_group>() ==
                maxWGs;
            testResults[14] =
                root.template get_linear_range<execution_scope::sub_group>() ==
                NumSGPerWG * maxWGs;
            testResults[15] =
                root.template get_linear_range<execution_scope::work_item>() ==
                maxWGs * WorkGroupSize;
          }

          if (root.get_local_id() == 3) {
            testResults[16] =
                root.template get_id<execution_scope::work_group>() ==
                it.get_global_linear_id() / WorkGroupSize;
            testResults[17] =
                root.template get_id<execution_scope::sub_group>() ==
                it.get_global_linear_id() / SGSize;
            testResults[18] =
                root.template get_id<execution_scope::work_item>() ==
                it.get_global_linear_id();

            testResults[19] =
                root.template get_linear_id<execution_scope::work_group>() ==
                it.get_global_linear_id() / WorkGroupSize;
            testResults[20] =
                root.template get_linear_id<execution_scope::sub_group>() ==
                it.get_global_linear_id() / SGSize;
            testResults[21] =
                root.template get_linear_id<execution_scope::work_item>() ==
                it.get_global_linear_id();
          }
        });
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
