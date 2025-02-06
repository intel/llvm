// RUN: %clangxx -fsycl -fsyntax-only -ferror-limit=0 -Xclang -verify -Xclang -verify-ignore-unexpected=note,warning %s

#include <sycl/sycl.hpp>

using namespace sycl;
namespace syclexp = sycl::ext::oneapi::experimental;

queue Q;

// This test verifies the type restrictions on the two non-default constructors
// of work group memory.

template <typename DataT> void convertToDataT(DataT &data) {}

template <typename DataT> void test_bounded_arr() {
  Q.submit([&](sycl::handler &cgh) {
    nd_range<1> ndr{1, 1};
    // expected-error-re@+1 5{{no matching constructor for initialization of 'syclexp::work_group_memory<{{.*}}>'}}
    syclexp::work_group_memory<DataT[1]> mem{1, cgh};
    // expected-error@+1 5{{no viable overloaded '='}}
    cgh.parallel_for(ndr, [=](nd_item<1> it) { mem = {DataT{}}; });
  });
}

template <typename DataT> void test_unbounded_arr() {
  Q.submit([&](sycl::handler &cgh) {
    nd_range<1> ndr{1, 1};
    // expected-error-re@+1 5{{no matching constructor for initialization of 'syclexp::work_group_memory<{{.*}}>'}}
    syclexp::work_group_memory<DataT[]> mem{cgh};
    // expected-error@+1 5{{no viable overloaded '='}}
    cgh.parallel_for(ndr, [=](nd_item<1> it) { mem = {DataT{}}; });
  });
}

template <typename... DataTs> void test() {
  (test_bounded_arr<DataTs>(), ...);
  (test_unbounded_arr<DataTs>(), ...);
}

int main() {
  test<char, int16_t, int[1], double[2], half[3]>();
  return 0;
}
