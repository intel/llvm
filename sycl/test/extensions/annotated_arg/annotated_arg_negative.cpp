// RUN: %clangxx -fsycl -ferror-limit=0 -fsycl-targets=%sycl_triple -fsyntax-only -Xclang -verify -Xclang -verify-ignore-unexpected=note,warning %s

#include "sycl/sycl.hpp"
#include <sycl/ext/intel/fpga_extensions.hpp>

using namespace sycl;
using namespace ext::oneapi::experimental;
using namespace ext::intel::experimental;

void check_conduit_and_register_map_properties() {
  // check for conduit and register_map properties specified together
  // expected-error@sycl/ext/oneapi/experimental/annotated_arg/annotated_arg.hpp:* {{The properties conduit and register_map cannot be specified at the same time.}}
  annotated_arg<int, decltype(properties{conduit, register_map})> a;
  // expected-error@sycl/ext/oneapi/experimental/annotated_arg/annotated_arg.hpp:* {{The properties conduit and register_map cannot be specified at the same time.}}
  annotated_arg<int *, decltype(properties{conduit, register_map})> b;
  // expected-error@sycl/ext/oneapi/experimental/annotated_ptr/annotated_ptr.hpp:* {{The properties conduit and register_map cannot be specified at the same time.}}
  annotated_ptr<int, decltype(properties{conduit, register_map})> c;
}

void check_invalid_properties_on_non_pointer_types() {
  // check buffer location property specified on non pointer type
  // expected-error@sycl/ext/oneapi/experimental/annotated_arg/annotated_arg.hpp:* {{Property buffer_location cannot be specified for annotated_arg<T> when T is a non pointer type.}}
  annotated_arg<int, decltype(properties{buffer_location<0>})> a;

  // check awidth property specified on non pointer type
  // expected-error@sycl/ext/oneapi/experimental/annotated_arg/annotated_arg.hpp:* {{Property awidth cannot be specified for annotated_arg<T> when T is a non pointer type.}}
  annotated_arg<int, decltype(properties{awidth<32>})> b;

  // check dwidth property specified on non pointer type
  // expected-error@sycl/ext/oneapi/experimental/annotated_arg/annotated_arg.hpp:* {{Property dwidth cannot be specified for annotated_arg<T> when T is a non pointer type.}}
  annotated_arg<int, decltype(properties{dwidth<32>})> c;

  // check latency property specified on non pointer type
  // expected-error@sycl/ext/oneapi/experimental/annotated_arg/annotated_arg.hpp:* {{Property latency cannot be specified for annotated_arg<T> when T is a non pointer type.}}
  annotated_arg<int, decltype(properties{latency<1>})> d;

  // check read_write_mode property specified on non pointer type
  // expected-error@sycl/ext/oneapi/experimental/annotated_arg/annotated_arg.hpp:* {{Property read_write_mode cannot be specified for annotated_arg<T> when T is a non pointer type.}}
  annotated_arg<int, decltype(properties{read_write_mode_readwrite})> e;

  // check maxburst property specified on non pointer type
  // expected-error@sycl/ext/oneapi/experimental/annotated_arg/annotated_arg.hpp:* {{Property maxburst cannot be specified for annotated_arg<T> when T is a non pointer type.}}
  annotated_arg<int, decltype(properties{maxburst<1>})> f;

  // check wait_request property specified on non pointer type
  // expected-error@sycl/ext/oneapi/experimental/annotated_arg/annotated_arg.hpp:* {{Property wait_request cannot be specified for annotated_arg<T> when T is a non pointer type.}}
  annotated_arg<int, decltype(properties{wait_request_requested})> g;

  // check alignment property specified on non pointer type
  // expected-error@sycl/ext/oneapi/experimental/annotated_arg/annotated_arg.hpp:* {{Property alignment cannot be specified for annotated_arg<T> when T is a non pointer type.}}
  annotated_arg<int, decltype(properties{alignment<256>})> h;
}

int main() {
  check_invalid_properties_on_non_pointer_types();
  check_conduit_and_register_map_properties();
  return 0;
}
