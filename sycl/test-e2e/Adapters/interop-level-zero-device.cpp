// REQUIRES: level_zero
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include <numeric>
#include <sycl/backend.hpp>
#include <sycl/detail/core.hpp>
#include <sycl/properties/all_properties.hpp>

template <typename FuncTy>
void for_each_descendent_dev(sycl::device dev, FuncTy F) {
  F(dev);

  std::vector<sycl::info::partition_property> partition_props =
      dev.get_info<sycl::info::device::partition_properties>();

  auto supports = [&](auto prop) {
    return std::find(partition_props.begin(), partition_props.end(), prop) !=
           partition_props.end();
  };

  if (supports(sycl::info::partition_property::partition_by_affinity_domain)) {
    std::cout << "Affinity" << std::endl;
    auto sub_devices = dev.create_sub_devices<
        sycl::info::partition_property::partition_by_affinity_domain>(
        sycl::info::partition_affinity_domain::next_partitionable);
    for (auto d : sub_devices)
      for_each_descendent_dev(d, F);
  }

  // I'm not sure if remaining partitioning schems are actually supported by any
  // of the existing Level-Zero devices. Make sure we still cover that
  // possibility in this test to accomodate any future situation.

  if (supports(sycl::info::partition_property::partition_equally)) {
    std::cout << "Equally" << std::endl;
    auto max_compute_units =
        dev.get_info<sycl::info::device::max_compute_units>();
    for (int count = 1; count < max_compute_units; ++count) {
      auto sub_devices = dev.create_sub_devices<
          sycl::info::partition_property::partition_equally>(count);
      for (auto d : sub_devices)
        for_each_descendent_dev(d, F);
    }
  }

  if (supports(sycl::info::partition_property::partition_by_counts)) {
    std::cout << "By counts" << std::endl;
    auto max_compute_units =
        dev.get_info<sycl::info::device::max_compute_units>();

    // Iterating over all possible sub-devices with this partitioning scheme
    // explodes combinatorially, yet Level-Zero backend specificaiton states
    // that device produced by `make_device` has to be a copy of a device from
    // the existing fixed number of (sub-)devices enumerated by
    // get_devices/create_sub_devices. As such, it wouldn't be practical to face
    // Level-Zero devices going through code path unless the specification is
    // changed.
    assert(max_compute_units <= 8 && "Don't expect L0 devices like that.");

    auto fill_counts_and_invoke = [&](auto self, std::vector<size_t> counts) {
      size_t used = std::accumulate(counts.begin(), counts.end(), 0);

      if (used == max_compute_units) {
        std::cout << "counts:";
        for (auto c : counts)
          std::cout << " " << c;
        std::cout << ", total: " << used << std::endl;

        auto sub_devices = dev.create_sub_devices<
            sycl::info::partition_property::partition_by_counts>(counts);
        for (auto d : sub_devices)
          for_each_descendent_dev(d, F);
        return;
      }
      for (size_t i = 1; i <= max_compute_units - used; ++i) {
        std::vector<size_t> new_counts{counts};
        new_counts.push_back(i);
        self(self, new_counts);
      }
    };
    fill_counts_and_invoke(fill_counts_and_invoke, {});
  }
}

int main() {
  auto root_devices = sycl::device::get_devices();

  for (auto d : root_devices)
    for_each_descendent_dev(d, [](sycl::device d) {
      int level = 0;
      sycl::device tmp = d;
      while (tmp.get_info<sycl::info::device::partition_type_property>() !=
             sycl::info::partition_property::no_partition) {
        ++level;
        tmp = tmp.template get_info<sycl::info::device::parent_device>();
      }
      std::cout << "Device at level " << level << std::endl;

      constexpr auto be = sycl::backend::ext_oneapi_level_zero;

      auto native = sycl::get_native<be>(d);
      auto from_native = sycl::make_device<be>(native);
      assert(d == from_native);
      std::hash<sycl::device> hash;
      assert(hash(d) == hash(from_native));
    });

  return 0;
}
