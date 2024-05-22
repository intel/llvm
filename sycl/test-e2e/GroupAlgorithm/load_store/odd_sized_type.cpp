// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/experimental/group_load_store.hpp>

#include <numeric>

using namespace sycl;
namespace sycl_exp = sycl::ext::oneapi::experimental;

struct __attribute__((packed)) S {
  S() = default;
  S(const S &) = default;
  S(S &&) = default;
  S &operator=(const S &) = default;
  S &operator=(S &&) = default;

  S(int i) : x(i), y(i), z(i) {}
  S &operator=(int i) {
    *this = S{i};
    return *this;
  }

  int32_t x;
  int16_t y;
  int8_t z;
};

static_assert(sizeof(S) == 7);
static_assert(std::is_trivially_copyable_v<S>);

int main() {

  constexpr std::size_t wg_size = 32;
  constexpr std::size_t elems_per_wi = 2;
  constexpr std::size_t n = wg_size * elems_per_wi;

  queue q;

  buffer<S, 1> input_buf{n};

  {
    host_accessor acc{input_buf};
    std::iota(acc.begin(), acc.end(), 0);
  }

  buffer<S, 1> load_blocked_buf{n};
  buffer<S, 1> load_striped_buf{n};
  buffer<S, 1> store_blocked_buf{n};
  buffer<S, 1> store_striped_buf{n};

  q.submit([&](handler &cgh) {
    accessor input{input_buf, cgh};

    accessor load_blocked{load_blocked_buf, cgh};
    accessor load_striped{load_striped_buf, cgh};
    accessor store_blocked{store_blocked_buf, cgh};
    accessor store_striped{store_striped_buf, cgh};

    cgh.parallel_for(nd_range<1>{wg_size, wg_size}, [=](nd_item<1> ndi) {
      auto gid = ndi.get_global_id(0);
      auto g = ndi.get_group();

      S data[elems_per_wi];

      auto blocked = sycl_exp::properties{sycl_exp::data_placement_blocked};
      auto striped = sycl_exp::properties{sycl_exp::data_placement_striped};

      // blocked
      sycl_exp::group_load(g, input.begin(), span{data}, blocked);
      for (int i = 0; i < elems_per_wi; ++i)
        load_blocked[gid * elems_per_wi + i] = data[i];

      // striped
      sycl_exp::group_load(g, input.begin(), span{data}, striped);
      for (int i = 0; i < elems_per_wi; ++i)
        load_striped[gid * elems_per_wi + i] = data[i];

      // Stores:

      std::iota(std::begin(data), std::end(data), gid * elems_per_wi);

      sycl_exp::group_store(g, span{data}, store_blocked.begin(), blocked);
      sycl_exp::group_store(g, span{data}, store_striped.begin(), striped);
    });
  });

  host_accessor load_blocked{load_blocked_buf};
  host_accessor load_striped{load_striped_buf};
  host_accessor store_blocked{store_blocked_buf};
  host_accessor store_striped{store_striped_buf};

  auto Check = [](S s, int v) {
    assert(s.x == v);
    assert(s.y == v);
    assert(s.z == v);
  };

  // Check blocked.
  for (int i = 0; i < wg_size * elems_per_wi; ++i) {
    Check(load_blocked[i], i);
    Check(store_blocked[i], i);
  }

  // Check striped.
  for (int wi = 0; wi < wg_size; ++wi) {
    auto group = wi / wg_size;
    auto lid = wi % wg_size;

    for (auto elem = 0; elem < elems_per_wi; ++elem) {
      auto striped_idx = group * wg_size * elems_per_wi + elem * wg_size + lid;
      Check(load_striped[wi * elems_per_wi + elem], striped_idx);

      auto value_stored = wi * elems_per_wi + elem;
      Check(store_striped[striped_idx], value_stored);
    }
  }

  return 0;
}
