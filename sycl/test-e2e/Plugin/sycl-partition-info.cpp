// RUN:  %clangxx -fsycl -fsycl-targets=%sycl_triple  %s -o %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out

// various plugins may return a larger choice of partition properties than SYCL
// supports ensure we are only returning SYCL standard  partition properties.

#include <cassert>
#include <sycl/sycl.hpp>

using namespace sycl;

int main(void) {
  device d;

  auto pp = d.get_info<info::device::partition_properties>();

  for (auto &p : pp) {
    switch (p) {
    case info::partition_property::no_partition:
    case info::partition_property::partition_equally:
    case info::partition_property::partition_by_counts:
    case info::partition_property::partition_by_affinity_domain:
      break;
    default:
      assert(false && "Unrecognized partition property");
    }
  }

  return 0;
}
