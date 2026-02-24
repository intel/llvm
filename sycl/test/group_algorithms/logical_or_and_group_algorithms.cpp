// RUN: %clangxx -fsycl -Xclang -verify=expected -Xclang -verify-ignore-unexpected=note -fsyntax-only -fsycl-device-only -ferror-limit=0 %s

#include <sycl/group_algorithm.hpp>

using namespace sycl;

constexpr size_t N = 8;
range<2> global{16, 32};
range<2> local{4, 8};
id<2> groupId{1, 2};
group<2> g = detail::Builder::createGroup(global, local, groupId);
int *rawIn = nullptr;
int *rawOut = nullptr;

void ExclusiveScanOverGroup() {
  // expected-error@sycl/group_algorithm.hpp:* {{Result type of binary_op must match scan accumulation type}}
  exclusive_scan_over_group(g, 0, sycl::logical_and<int>{});
  // expected-error@sycl/group_algorithm.hpp:* {{Result type of binary_op must match scan accumulation type}}
  exclusive_scan_over_group(g, 0, 0, sycl::logical_and<int>{});
  // expected-error@sycl/group_algorithm.hpp:* {{Result type of binary_op must match scan accumulation type}}
  exclusive_scan_over_group(g, 0, sycl::logical_or<int>{});
  // expected-error@sycl/group_algorithm.hpp:* {{Result type of binary_op must match scan accumulation type}}
  exclusive_scan_over_group(g, 0, 0, sycl::logical_or<int>{});
}

void JointExclusiveScan() {
  // expected-error@sycl/group_algorithm.hpp:* {{Result type of binary_op must match scan accumulation type}}
  joint_exclusive_scan(g, rawIn, rawIn + N, rawOut, sycl::logical_and<int>{});
  // expected-error@sycl/group_algorithm.hpp:* {{Result type of binary_op must match scan accumulation type}}
  joint_exclusive_scan(g, rawIn, rawIn + N, rawOut, sycl::logical_or<int>{});
  // expected-error@sycl/group_algorithm.hpp:* {{Result type of binary_op must match scan accumulation type}}
  joint_exclusive_scan(g, rawIn, rawIn + N, rawOut, 0,
                       sycl::logical_and<int>{});
  // expected-error@sycl/group_algorithm.hpp:* {{Result type of binary_op must match scan accumulation type}}
  joint_exclusive_scan(g, rawIn, rawIn + N, rawOut, 0, sycl::logical_or<int>{});
}

void InclusiveScanOverGroup() {
  // expected-error@sycl/group_algorithm.hpp:* {{Result type of binary_op must match scan accumulation type}}
  inclusive_scan_over_group(g, 0, sycl::logical_and<int>{});
  // expected-error@sycl/group_algorithm.hpp:* {{Result type of binary_op must match scan accumulation type}}
  inclusive_scan_over_group(g, 0, sycl::logical_and<int>{}, 0);
  // expected-error@sycl/group_algorithm.hpp:* {{Result type of binary_op must match scan accumulation type}}
  inclusive_scan_over_group(g, 0, sycl::logical_or<int>{});
  // expected-error@sycl/group_algorithm.hpp:* {{Result type of binary_op must match scan accumulation type}}
  inclusive_scan_over_group(g, 0, sycl::logical_or<int>{}, 0);
}

void JointInclusiveScan() {
  // expected-error@sycl/group_algorithm.hpp:* {{Result type of binary_op must match scan accumulation type}}
  joint_inclusive_scan(g, rawIn, rawIn + N, rawOut, sycl::logical_and<int>{});
  // expected-error@sycl/group_algorithm.hpp:* {{Result type of binary_op must match scan accumulation type}}
  joint_inclusive_scan(g, rawIn, rawIn + N, rawOut, sycl::logical_or<int>{});
  // expected-error@sycl/group_algorithm.hpp:* {{Result type of binary_op must match scan accumulation type}}
  joint_inclusive_scan(g, rawIn, rawIn + N, rawOut, sycl::logical_and<int>{},
                       0);
  // expected-error@sycl/group_algorithm.hpp:* {{Result type of binary_op must match scan accumulation type}}
  joint_inclusive_scan(g, rawIn, rawIn + N, rawOut, sycl::logical_or<int>{}, 0);
}

void ReduceOverGroup() {
  // expected-error@sycl/group_algorithm.hpp:* {{Result type of binary_op must match reduction accumulation type}}
  reduce_over_group(g, 0, sycl::logical_and<int>{});
  // expected-error@sycl/group_algorithm.hpp:* {{Result type of binary_op must match reduction accumulation type}}
  reduce_over_group(g, 0, 0, sycl::logical_and<int>{});
  // expected-error@sycl/group_algorithm.hpp:* {{Result type of binary_op must match reduction accumulation type}}
  reduce_over_group(g, 0, sycl::logical_or<int>{});
  // expected-error@sycl/group_algorithm.hpp:* {{Result type of binary_op must match reduction accumulation type}}
  reduce_over_group(g, 0, 0, sycl::logical_or<int>{});
}

void JointReduce() {
  // expected-error@sycl/group_algorithm.hpp:* {{Result type of binary_op must match reduction accumulation type}}
  joint_reduce(g, rawIn, rawIn + N, 0, sycl::logical_and<int>{});
  // expected-error@sycl/group_algorithm.hpp:* {{Result type of binary_op must match reduction accumulation type}}
  joint_reduce(g, rawIn, rawIn + N, 0, sycl::logical_or<int>{});
}
