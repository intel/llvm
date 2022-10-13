// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple -fPIC -DCASE1 %s -c -o %t.1.o
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple -fPIC -DCASE2 %s -c -o %t.2.o
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple -shared %t.1.o %t.2.o -o %t.so
//
// Some of the above compiler options will not work on Windows.
// UNSUPPORTED: windows

// Tests that creating a shared library with multiple object files using joint
// group operations does not cause conflicting definitions.

#include <sycl/sycl.hpp>

#ifdef CASE1
#define FNAME test1
#define KNAME kernel1
#else
#define FNAME test2
#define KNAME kernel2
#endif

void FNAME(sycl::queue &Q, bool *In, bool *Out) {
  sycl::nd_range<1> WorkRange(sycl::range<1>(8), sycl::range<1>(16));
  Q.parallel_for<class KNAME>(WorkRange, [=](sycl::nd_item<1> It) {
     sycl::group<1> Group = It.get_group();
     size_t I = It.get_global_linear_id();
     Out[I] = sycl::joint_any_of(Group, In, In, [](bool B) { return B; });
   }).wait();
}
