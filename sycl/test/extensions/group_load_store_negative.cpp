// RUN: %clangxx -fsycl -fsycl-device-only -fsyntax-only -Xclang -verify -Xclang -verify-ignore-unexpected=warning,note %s

#include <sycl/sycl.hpp>

using namespace sycl;
using namespace sycl::ext::oneapi::experimental;

void foo(sub_group sg, int *p, vec<int, 4> &v) {
  // expected-error@+2 {{no matching function for call to 'group_load'}}
  // expected-note-re@*:* {{candidate template ignored: requirement 'is_property_list_v<{{.*data_placement.*}}>' was not satisfied {{.*}}}}
  group_load(sg, p, v, data_placement_blocked);

  // This is ok:
  group_load(sg, p, v, properties{data_placement_blocked});
}
