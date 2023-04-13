// This test is intended to check that creating vec<DataT, N> with unsupported
// DataT or N produces a verbose error message.
//
// RUN: %clangxx %fsycl-host-only -fsyntax-only -Xclang -verify -Xclang -verify-ignore-unexpected=note,error %s
// RUN: %clangxx %fsycl-host-only  -D__NO_EXT_VECTOR_TYPE_ON_HOST__ -fsyntax-only -Xclang -verify -Xclang -verify-ignore-unexpected=note,error %s
//
// Note: there is one more error being emitted: "requested alignemnt is not a
// power of 2" It happens because in all cases above we weren't able to select
// underlying data type for vec and therefore it screwed up other code realying
// on it. This error message is not they key thing we want to test here and
// that's why it is ok to ignore it.

#include <sycl/sycl.hpp>

struct CustomT {
  int a;
  float b;
};

void unsupported_data_type() {
  // expected-error@sycl/types.hpp:* {{Incorrect data type for sycl::vec}}
  sycl::vec<CustomT, 4> v;
}

void unsupported_size() {
  // expected-error@sycl/types.hpp:* {{Incorrect number of elements for sycl::vec}}
  sycl::vec<int, 15> v;
}
