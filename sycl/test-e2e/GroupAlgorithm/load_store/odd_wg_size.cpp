// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/experimental/group_load_store.hpp>

#include <numeric>

using namespace sycl;
namespace sycl_exp = sycl::ext::oneapi::experimental;

// Similar to partial_sg.cpp, but check group (vs. sub_group) loads/stores when
// WG_SIZE isn't equally divisible by SG_SIZE.

template <int SG_SIZE, int WG_SIZE> void test(queue &q) {
  constexpr std::size_t wg_size = WG_SIZE;
  constexpr std::size_t n_wgs = 2;
  constexpr std::size_t global_size = n_wgs * wg_size;
  constexpr std::size_t elems_per_wi = 4;
  constexpr std::size_t n = global_size * elems_per_wi;

  buffer<int, 1> input_buf{n};

  {
    host_accessor acc{input_buf};
    std::iota(acc.begin(), acc.end(), 0);
  }

  buffer<int, 1> load_blocked_buf{n};
  buffer<int, 1> load_striped_buf{n};
  buffer<int, 1> store_blocked_buf{n};
  buffer<int, 1> store_striped_buf{n};

  q.submit([&](handler &cgh) {
    accessor input{input_buf, cgh};

    accessor load_blocked{load_blocked_buf, cgh};
    accessor load_striped{load_striped_buf, cgh};
    accessor store_blocked{store_blocked_buf, cgh};
    accessor store_striped{store_striped_buf, cgh};

    cgh.parallel_for(
        nd_range<1>{global_size, wg_size},
        [=](nd_item<1> ndi) [[intel::reqd_sub_group_size(SG_SIZE)]] {
          auto gid = ndi.get_global_id(0);
          auto g = ndi.get_group();
          auto offset = g.get_group_id(0) * g.get_local_range(0) * elems_per_wi;

          int data[elems_per_wi];

          auto blocked = sycl_exp::properties{sycl_exp::data_placement_blocked};
          auto striped = sycl_exp::properties{sycl_exp::data_placement_striped};

          // blocked
          sycl_exp::group_load(g, input.begin() + offset, span{data}, blocked);
          for (int i = 0; i < elems_per_wi; ++i)
            load_blocked[gid * elems_per_wi + i] = data[i];

          // striped
          sycl_exp::group_load(g, input.begin() + offset, span{data}, striped);
          for (int i = 0; i < elems_per_wi; ++i)
            load_striped[gid * elems_per_wi + i] = data[i];

          // Stores:

          std::iota(std::begin(data), std::end(data), gid * elems_per_wi);

          sycl_exp::group_store(g, span{data}, store_blocked.begin() + offset,
                                blocked);
          sycl_exp::group_store(g, span{data}, store_striped.begin() + offset,
                                striped);
        });
  });

  host_accessor load_blocked{load_blocked_buf};
  host_accessor load_striped{load_striped_buf};
  host_accessor store_blocked{store_blocked_buf};
  host_accessor store_striped{store_striped_buf};

  // Check blocked.
  for (int i = 0; i < global_size * elems_per_wi; ++i) {
    assert(load_blocked[i] == i);
    assert(store_blocked[i] == i);
  }

  // Check striped.
  for (int wi = 0; wi < global_size; ++wi) {
    auto group = wi / wg_size;
    auto lid = wi % wg_size;

    for (auto elem = 0; elem < elems_per_wi; ++elem) {
      auto striped_idx = group * wg_size * elems_per_wi + elem * wg_size + lid;
      assert(load_striped[wi * elems_per_wi + elem] == striped_idx);

      auto value_stored = wi * elems_per_wi + elem;
      assert(store_striped[striped_idx] == value_stored);
    }
  }
}

int main() {
  queue q;
  auto device_sg_sizes =
      q.get_device().get_info<info::device::sub_group_sizes>();

  constexpr std::size_t sg_sizes[] = {4, 8, 16, 32};

  detail::loop<std::size(sg_sizes)>([&](auto sg_size_idx) {
    constexpr auto sg_size = sg_sizes[sg_size_idx];
    if (std::none_of(device_sg_sizes.begin(), device_sg_sizes.end(),
                     [](auto x) { return x == sg_size; }))
      return;
    test<sg_size, sg_size / 2>(q);
    test<sg_size, sg_size * 3 / 2>(q);
  });

  return 0;
}
