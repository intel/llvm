// RUN: %clangxx -fsycl -Xclang -verify=expected -Xclang -verify-ignore-unexpected=note -fpreview-breaking-changes -fsyntax-only -ferror-limit=0 %s


#include <sycl/functional.hpp>
#include <sycl/group_algorithm.hpp>
#include <sycl/handler.hpp>
#include <sycl/nd_range.hpp>
#include <sycl/queue.hpp>

using namespace sycl;

void TestExclusiveScanOverGroup(sycl::queue &q) {
  q.submit([&](handler &cgh) {
    cgh.parallel_for<class ExclusiveScanOverGroup>(
        nd_range<1>(1, 1), [=](nd_item<1> it) {
          group<1> g = it.get_group();
          // expected-error@sycl/group_algorithm.hpp:* {{static assertion failed}}{{Result type of binary_op must match scan accumulation type}}
          exclusive_scan_over_group(g, 0, sycl::logical_and<int>{});
          // expected-error@sycl/group_algorithm.hpp:* {{static assertion failed}}{{Result type of binary_op must match scan accumulation type}}
          exclusive_scan_over_group(g, 0, 0, sycl::logical_and<int>{});
          // expected-error@sycl/group_algorithm.hpp:* {{static assertion failed}}{{Result type of binary_op must match scan accumulation type}}
          exclusive_scan_over_group(g, 0, sycl::logical_or<int>{});
          // expected-error@sycl/group_algorithm.hpp:* {{static assertion failed}}{{Result type of binary_op must match scan accumulation type}}
          exclusive_scan_over_group(g, 0, 0, sycl::logical_or<int>{});
        });
  });
}

void TestJointExclusiveScan(sycl::queue &q) {
  constexpr size_t N = 8;
  int input[N] = {1, 2, 3, 4, 5, 6, 7, 8};
  int output[N] = {};

  sycl::buffer<int, 1> inBuf(input, sycl::range<1>(N));
  sycl::buffer<int, 1> outBuf(output, sycl::range<1>(N));

  q.submit([&](sycl::handler &cgh) {
     auto in = inBuf.get_access<sycl::access::mode::read>(cgh);
     auto out = outBuf.get_access<sycl::access::mode::write>(cgh);

     cgh.parallel_for<class JointExclusiveScan>(
         sycl::nd_range<1>(sycl::range<1>(N), sycl::range<1>(N)),
         [=](sycl::nd_item<1> it) {
           auto g = it.get_group();
           auto inPtr = in.get_multi_ptr<sycl::access::decorated::no>();
           auto outPtr = out.get_multi_ptr<sycl::access::decorated::no>();
           // expected-error@sycl/group_algorithm.hpp:* {{static assertion failed}}{{Result type of binary_op must match scan accumulation type}}
           joint_exclusive_scan(g, inPtr, inPtr + N, outPtr,
                                sycl::logical_and<int>{});
           // expected-error@sycl/group_algorithm.hpp:* {{static assertion failed}}{{Result type of binary_op must match scan accumulation type}}                     
           joint_exclusive_scan(g, inPtr, inPtr + N, outPtr,
                                sycl::logical_or<int>{});
           // expected-error@sycl/group_algorithm.hpp:* {{static assertion failed}}{{Result type of binary_op must match scan accumulation type}}
           joint_exclusive_scan(g, inPtr, inPtr + N, outPtr, 0,
                                sycl::logical_and<int>{});
           // expected-error@sycl/group_algorithm.hpp:* {{static assertion failed}}{{Result type of binary_op must match scan accumulation type}}
           joint_exclusive_scan(g, inPtr, inPtr + N, outPtr, 0,
                                sycl::logical_or<int>{});
         });
   }).wait();
}

void TestInclusiveScanOverGroup(sycl::queue &q) {
  q.submit([&](handler &cgh) {
    cgh.parallel_for<class InclusiveScanOverGroup>(
        nd_range<1>(1, 1), [=](nd_item<1> it) {
          group<1> g = it.get_group();
          // expected-error@sycl/group_algorithm.hpp:* {{static assertion failed}}{{Result type of binary_op must match scan accumulation type}}
          inclusive_scan_over_group(g, 0, sycl::logical_and<int>{});
          // expected-error@sycl/group_algorithm.hpp:* {{static assertion failed}}{{Result type of binary_op must match scan accumulation type}}
          inclusive_scan_over_group(g, 0, sycl::logical_and<int>{}, 0);
          // expected-error@sycl/group_algorithm.hpp:* {{static assertion failed}}{{Result type of binary_op must match scan accumulation type}}
          inclusive_scan_over_group(g, 0, sycl::logical_or<int>{});
          // expected-error@sycl/group_algorithm.hpp:* {{static assertion failed}}{{Result type of binary_op must match scan accumulation type}}
          inclusive_scan_over_group(g, 0, sycl::logical_or<int>{}, 0);
        });
  });
}

void TestJointInclusiveScan(sycl::queue &q) {
  constexpr size_t N = 8;
  int input[N] = {1, 2, 3, 4, 5, 6, 7, 8};
  int output[N] = {};

  sycl::buffer<int, 1> inBuf(input, sycl::range<1>(N));
  sycl::buffer<int, 1> outBuf(output, sycl::range<1>(N));

  q.submit([&](sycl::handler &cgh) {
     auto in = inBuf.get_access<sycl::access::mode::read>(cgh);
     auto out = outBuf.get_access<sycl::access::mode::write>(cgh);

     cgh.parallel_for<class JointInclusiveScan>(
         sycl::nd_range<1>(sycl::range<1>(N), sycl::range<1>(N)),
         [=](sycl::nd_item<1> it) {
           auto g = it.get_group();
           auto inPtr = in.get_multi_ptr<sycl::access::decorated::no>();
           auto outPtr = out.get_multi_ptr<sycl::access::decorated::no>();

           // expected-error@sycl/group_algorithm.hpp:* {{static assertion failed}}{{Result type of binary_op must match scan accumulation type}}
           joint_inclusive_scan(g, inPtr, inPtr + N, outPtr,
                                sycl::logical_and<int>{});
           // expected-error@sycl/group_algorithm.hpp:* {{static assertion failed}}{{Result type of binary_op must match scan accumulation type}}
           joint_inclusive_scan(g, inPtr, inPtr + N, outPtr,
                                sycl::logical_or<int>{});
           // expected-error@sycl/group_algorithm.hpp:* {{static assertion failed}}{{Result type of binary_op must match scan accumulation type}}
           joint_inclusive_scan(g, inPtr, inPtr + N, outPtr,
                                sycl::logical_and<int>{}, 0);
           // expected-error@sycl/group_algorithm.hpp:* {{static assertion failed}}{{Result type of binary_op must match scan accumulation type}}
           joint_inclusive_scan(g, inPtr, inPtr + N, outPtr,
                                sycl::logical_or<int>{}, 0);
         });
   }).wait();
}

void TestReduceOverGroup(sycl::queue &q) {
  q.submit([&](handler &cgh) {
    cgh.parallel_for<class ReduceOverGroup>(
        nd_range<1>(1, 1), [=](nd_item<1> it) {
          group<1> g = it.get_group();
          // expected-error@sycl/group_algorithm.hpp:* {{Result type of binary_op must match reduction accumulation type}}
          reduce_over_group(g, 0, sycl::logical_and<int>{});
          // expected-error@sycl/group_algorithm.hpp:* {{Result type of binary_op must match reduction accumulation type}}
          reduce_over_group(g, 0, 0, sycl::logical_and<int>{});
          // expected-error@sycl/group_algorithm.hpp:* {{Result type of binary_op must match reduction accumulation type}}
          reduce_over_group(g, 0, sycl::logical_or<int>{});
          // expected-error@sycl/group_algorithm.hpp:* {{Result type of binary_op must match reduction accumulation type}}
          reduce_over_group(g, 0, 0, sycl::logical_or<int>{});
        });
  });
}

void TestJointReduce(sycl::queue &q) {
  constexpr size_t N = 8;
  int input[N] = {1, 2, 3, 4, 5, 6, 7, 8};
  int output[N] = {};

  sycl::buffer<int, 1> inBuf(input, sycl::range<1>(N));
  sycl::buffer<int, 1> outBuf(output, sycl::range<1>(N));

  q.submit([&](sycl::handler &cgh) {
     auto in = inBuf.get_access<sycl::access::mode::read>(cgh);
     auto out = outBuf.get_access<sycl::access::mode::write>(cgh);

     cgh.parallel_for<class JointReduce>(
         sycl::nd_range<1>(sycl::range<1>(N), sycl::range<1>(N)),
         [=](sycl::nd_item<1> it) {
           auto g = it.get_group();
           auto inPtr = in.get_multi_ptr<sycl::access::decorated::no>();
           auto outPtr = out.get_multi_ptr<sycl::access::decorated::no>();

           // expected-error@sycl/group_algorithm.hpp:* {{Result type of binary_op must match reduction accumulation type}}
           joint_reduce(g, inPtr, inPtr + N, 0, sycl::logical_and<int>{});
           // expected-error@sycl/group_algorithm.hpp:* {{Result type of binary_op must match reduction accumulation type}}
           joint_reduce(g, inPtr, inPtr + N, 0, sycl::logical_or<int>{});
         });
   }).wait();
}

int main() {
  sycl::queue q;
  TestExclusiveScanOverGroup(q);
  TestJointExclusiveScan(q);
  TestInclusiveScanOverGroup(q);
  TestJointInclusiveScan(q);
  TestReduceOverGroup(q);
  TestJointReduce(q);
  return 0;
}
