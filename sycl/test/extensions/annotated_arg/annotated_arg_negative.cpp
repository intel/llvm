// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple -fsyntax-only -Xclang -verify -Xclang -verify-ignore-unexpected=note,warning %s

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

int main() {
  check_conduit_and_register_map_properties();
  return 0;
}
