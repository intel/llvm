// RUN: %clangxx -fsycl -Xclang -verify=expected -fpreview-breaking-changes -fsyntax-only -ferror-limit=0 %s

// expected-error@sycl/group_algorithm.hpp:* 16 {{Result type of binary_op must match scan accumulation type}}
// expected-error@sycl/group_algorithm.hpp:* 6 {{Result type of binary_op must match reduction accumulation type}}

#include <sycl/queue.hpp>
#include <sycl/handler.hpp>
#include <sycl/nd_range.hpp>
#include <sycl/group_algorithm.hpp>
#include <sycl/functional.hpp>

using namespace sycl;

void TestExclusiveScanOverGroup(sycl::queue &q) {
  q.submit([&](handler &cgh) {
    cgh.parallel_for<class ExclusiveScanOverGroup>(
        nd_range<1>(1, 1), [=](nd_item<1> it) {
          group<1> g = it.get_group();
          // expected-note@+1 {{in instantiation of function template specialization 'sycl::exclusive_scan_over_group<sycl::group<>, int, sycl::logical_and<int>>' requested here}}
          exclusive_scan_over_group(g, 0, sycl::logical_and<int>{});
          // expected-note@+1 {{in instantiation of function template specialization 'sycl::exclusive_scan_over_group<sycl::group<>, int, int, sycl::logical_and<int>>' requested here}}
          exclusive_scan_over_group(g, 0, 0, sycl::logical_and<int>{});
          // expected-note@+1 {{in instantiation of function template specialization 'sycl::exclusive_scan_over_group<sycl::group<>, int, sycl::logical_or<int>>' requested here}}
          exclusive_scan_over_group(g, 0, sycl::logical_or<int>{});
          // expected-note@+1 {{in instantiation of function template specialization 'sycl::exclusive_scan_over_group<sycl::group<>, int, int, sycl::logical_or<int>>' requested here}}
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
           // expected-note@+1 {{in instantiation of function template specialization 'sycl::joint_exclusive_scan<sycl::group<>, sycl::multi_ptr<const int, sycl::access::address_space::global_space, sycl::access::decorated::no>, sycl::multi_ptr<int, sycl::access::address_space::global_space, sycl::access::decorated::no>, sycl::logical_and<int>>' requested here}}
           joint_exclusive_scan(g, inPtr, inPtr + N, outPtr,
                                sycl::logical_and<int>{});
           // expected-note@+1 {{in instantiation of function template specialization 'sycl::joint_exclusive_scan<sycl::group<>, sycl::multi_ptr<const int, sycl::access::address_space::global_space, sycl::access::decorated::no>, sycl::multi_ptr<int, sycl::access::address_space::global_space, sycl::access::decorated::no>, sycl::logical_or<int>>' requested here}}
           joint_exclusive_scan(g, inPtr, inPtr + N, outPtr,
                                sycl::logical_or<int>{});

           // expected-note@+1 {{in instantiation of function template specialization 'sycl::joint_exclusive_scan<sycl::group<>, sycl::multi_ptr<const int, sycl::access::address_space::global_space, sycl::access::decorated::no>, sycl::multi_ptr<int, sycl::access::address_space::global_space, sycl::access::decorated::no>, int, sycl::logical_and<int>>' requested here}}
           joint_exclusive_scan(g, inPtr, inPtr + N, outPtr, 0,
                                sycl::logical_and<int>{});
           // expected-note@+1 {{in instantiation of function template specialization 'sycl::joint_exclusive_scan<sycl::group<>, sycl::multi_ptr<const int, sycl::access::address_space::global_space, sycl::access::decorated::no>, sycl::multi_ptr<int, sycl::access::address_space::global_space, sycl::access::decorated::no>, int, sycl::logical_or<int>>' requested here}}
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
          // expected-note@+1 {{in instantiation of function template specialization 'sycl::inclusive_scan_over_group<sycl::group<>, int, sycl::logical_and<int>>' requested here}}
          inclusive_scan_over_group(g, 0, sycl::logical_and<int>{});
          // expected-note@+1 {{in instantiation of function template specialization 'sycl::inclusive_scan_over_group<sycl::group<>, int, sycl::logical_and<int>, int>' requested here}}
          inclusive_scan_over_group(g, 0, sycl::logical_and<int>{}, 0);
          // expected-note@+1 {{in instantiation of function template specialization 'sycl::inclusive_scan_over_group<sycl::group<>, int, sycl::logical_or<int>>' requested here}}
          inclusive_scan_over_group(g, 0, sycl::logical_or<int>{});
          // expected-note@+1 {{in instantiation of function template specialization 'sycl::inclusive_scan_over_group<sycl::group<>, int, sycl::logical_or<int>, int>' requested here}}
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

           // expected-note@+1 {{in instantiation of function template specialization 'sycl::joint_inclusive_scan<sycl::group<>, sycl::multi_ptr<const int, sycl::access::address_space::global_space, sycl::access::decorated::no>, sycl::multi_ptr<int, sycl::access::address_space::global_space, sycl::access::decorated::no>, sycl::logical_and<int>>' requested here}}
           joint_inclusive_scan(g, inPtr, inPtr + N, outPtr,
                                sycl::logical_and<int>{});
           // expected-note@+1 {{in instantiation of function template specialization 'sycl::joint_inclusive_scan<sycl::group<>, sycl::multi_ptr<const int, sycl::access::address_space::global_space, sycl::access::decorated::no>, sycl::multi_ptr<int, sycl::access::address_space::global_space, sycl::access::decorated::no>, sycl::logical_or<int>>' requested here}}
           joint_inclusive_scan(g, inPtr, inPtr + N, outPtr,
                                sycl::logical_or<int>{});

           // expected-note@+1 {{in instantiation of function template specialization 'sycl::joint_inclusive_scan<sycl::group<>, sycl::multi_ptr<const int, sycl::access::address_space::global_space, sycl::access::decorated::no>, sycl::multi_ptr<int, sycl::access::address_space::global_space, sycl::access::decorated::no>, sycl::logical_and<int>, int>' requested here}}
           joint_inclusive_scan(g, inPtr, inPtr + N, outPtr,
                                sycl::logical_and<int>{}, 0);
           // expected-note@+1 {{in instantiation of function template specialization 'sycl::joint_inclusive_scan<sycl::group<>, sycl::multi_ptr<const int, sycl::access::address_space::global_space, sycl::access::decorated::no>, sycl::multi_ptr<int, sycl::access::address_space::global_space, sycl::access::decorated::no>, sycl::logical_or<int>, int>' requested here}}
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
          // expected-note@+1 {{in instantiation of function template specialization 'sycl::reduce_over_group<sycl::group<>, int, sycl::logical_and<int>>' requested here}}
          reduce_over_group(g, 0, sycl::logical_and<int>{});
          // expected-note@+1 {{in instantiation of function template specialization 'sycl::reduce_over_group<sycl::group<>, int, int, sycl::logical_and<int>>' requested here}}
          reduce_over_group(g, 0, 0, sycl::logical_and<int>{});
          // expected-note@+1 {{in instantiation of function template specialization 'sycl::reduce_over_group<sycl::group<>, int, sycl::logical_or<int>>' requested here}}
          reduce_over_group(g, 0, sycl::logical_or<int>{});
          // expected-note@+1 {{in instantiation of function template specialization 'sycl::reduce_over_group<sycl::group<>, int, int, sycl::logical_or<int>>' requested here}}
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

           // expected-note@+1 {{in instantiation of function template specialization 'sycl::joint_reduce<sycl::group<>, sycl::multi_ptr<const int, sycl::access::address_space::global_space, sycl::access::decorated::no>, int, sycl::logical_and<int>>' requested here}}
           joint_reduce(g, inPtr, inPtr + N, 0, sycl::logical_and<int>{});
           // expected-note@+1 {{in instantiation of function template specialization 'sycl::joint_reduce<sycl::group<>, sycl::multi_ptr<const int, sycl::access::address_space::global_space, sycl::access::decorated::no>, int, sycl::logical_or<int>>' requested here}}
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
