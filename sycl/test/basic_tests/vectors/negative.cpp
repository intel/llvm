// This test is intended to check that creating vec<DataT, N> with unsupported
// DataT or N produces a verbose error message.
//
// RUN: %clangxx %fsycl-host-only -fsyntax-only -Xclang -verify -Xclang -verify-ignore-unexpected=note,error %s
// RUN: %if preview-breaking-changes-supported %{%clangxx %fsycl-host-only -fsyntax-only -fpreview-breaking-changes -Xclang -verify -Xclang -verify-ignore-unexpected=note,error %s%}

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
  // expected-error@detail/vec_* {{Incorrect data type for sycl::vec}}
  sycl::vec<CustomT, 4> v;
}

void unsupported_size() {
  // expected-error@detail/vec_* {{Incorrect number of elements for sycl::vec}}
  sycl::vec<int, 15> v;
}
