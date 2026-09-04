// RUN: %clangxx -fsycl-device-only -Xclang -fsycl-is-device -fsyntax-only -ferror-limit=0 -Xclang -verify -Xclang -verify-ignore-unexpected=note,warning %s

#include <sycl/sycl.hpp>

struct Kernel {
  void operator()(sycl::id<1> i) const {}

  auto get(sycl::ext::oneapi::experimental::properties_tag) const {
    return sycl::ext::oneapi::experimental::properties{
        sycl::ext::intel::experimental::maximum_registers<768>};
  }
};

int main() {
  // expected-error-re@sycl/ext/oneapi/properties.hpp:* {{static assertion failed due to requirement {{.+}}: Conflicting properties in property list.}}
  // expected-note-re@+1 {{in instantiation of function template specialization {{.+}}}}
  sycl::ext::oneapi::experimental::properties{
      sycl::ext::intel::experimental::grf_size<128>,
      sycl::ext::intel::experimental::maximum_registers<128>};

  // expected-error-re@sycl/ext/oneapi/properties.hpp:* {{static assertion failed due to requirement {{.+}}: Conflicting properties in property list.}}
  // expected-note-re@+1 {{in instantiation of function template specialization {{.+}}}}
  sycl::ext::oneapi::experimental::properties{
      sycl::ext::intel::experimental::grf_size<128>,
      sycl::ext::intel::experimental::grf_size_automatic};

  // expected-error-re@sycl/ext/oneapi/properties.hpp:* {{static assertion failed due to requirement {{.+}}: Conflicting properties in property list.}}
  // expected-note-re@+1 {{in instantiation of function template specialization {{.+}}}}
  sycl::ext::oneapi::experimental::properties{
      sycl::ext::intel::experimental::grf_size<128>,
      sycl::ext::intel::experimental::maximum_registers_automatic};

  // expected-error-re@sycl/ext/oneapi/properties.hpp:* {{static assertion failed due to requirement {{.+}}: Conflicting properties in property list.}}
  // expected-note-re@+1 {{in instantiation of function template specialization {{.+}}}}
  sycl::ext::oneapi::experimental::properties{
      sycl::ext::intel::experimental::grf_size_automatic,
      sycl::ext::intel::experimental::maximum_registers<128>};

  // expected-error-re@sycl/ext/oneapi/properties.hpp:* {{static assertion failed due to requirement {{.+}}: Conflicting properties in property list.}}
  // expected-note-re@+1 {{in instantiation of function template specialization {{.+}}}}
  sycl::ext::oneapi::experimental::properties{
      sycl::ext::intel::experimental::grf_size_automatic,
      sycl::ext::intel::experimental::maximum_registers_automatic};

  // expected-error-re@sycl/ext/oneapi/properties.hpp:* {{static assertion failed due to requirement {{.+}}: Conflicting properties in property list.}}
  // expected-note-re@+1 {{in instantiation of function template specialization {{.+}}}}
  sycl::ext::oneapi::experimental::properties{
      sycl::ext::intel::experimental::maximum_registers<128>,
      sycl::ext::intel::experimental::maximum_registers_automatic};

  sycl::queue q;
  // expected-error-re@sycl/ext/intel/experimental/maximum_registers_properties.hpp:* {{static assertion failed due to requirement {{.+}}: Unsupported maximum registers}}
  q.submit([&](sycl::handler &cgh) {
    cgh.parallel_for(sycl::range<1>(1), Kernel{});
  });
  return 0;
}
