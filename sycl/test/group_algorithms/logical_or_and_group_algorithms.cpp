// RUN: %clangxx -fsycl -Xclang -verify=expected -Xclang -verify-ignore-unexpected=note -fpreview-breaking-changes -fsyntax-only -ferror-limit=0 %s

// expected-error@sycl/group_algorithm.hpp:* 16 {{Result type of binary_op must match scan accumulation type}}
// expected-error@sycl/group_algorithm.hpp:* 6 {{Result type of binary_op must match reduction accumulation type}}

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
          exclusive_scan_over_group(g, 0, sycl::logical_and<int>{});
          exclusive_scan_over_group(g, 0, 0, sycl::logical_and<int>{});
          exclusive_scan_over_group(g, 0, sycl::logical_or<int>{});
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
           joint_exclusive_scan(g, inPtr, inPtr + N, outPtr,
                                sycl::logical_and<int>{});
           joint_exclusive_scan(g, inPtr, inPtr + N, outPtr,
                                sycl::logical_or<int>{});
           joint_exclusive_scan(g, inPtr, inPtr + N, outPtr, 0,
                                sycl::logical_and<int>{});
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
          inclusive_scan_over_group(g, 0, sycl::logical_and<int>{});
          inclusive_scan_over_group(g, 0, sycl::logical_and<int>{}, 0);
          inclusive_scan_over_group(g, 0, sycl::logical_or<int>{});
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

           joint_inclusive_scan(g, inPtr, inPtr + N, outPtr,
                                sycl::logical_and<int>{});
           joint_inclusive_scan(g, inPtr, inPtr + N, outPtr,
                                sycl::logical_or<int>{});
           joint_inclusive_scan(g, inPtr, inPtr + N, outPtr,
                                sycl::logical_and<int>{}, 0);
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
          reduce_over_group(g, 0, sycl::logical_and<int>{});
          reduce_over_group(g, 0, 0, sycl::logical_and<int>{});
          reduce_over_group(g, 0, sycl::logical_or<int>{});
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

           joint_reduce(g, inPtr, inPtr + N, 0, sycl::logical_and<int>{});
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
