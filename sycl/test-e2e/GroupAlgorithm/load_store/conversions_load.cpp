// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/experimental/group_load_store.hpp>

#include <numeric>

struct S {
  S() : i(-1) {}
  S(int i) : i(i + 42) {}

  int i;
};

int main() {
  using namespace sycl;
  namespace sycl_exp = sycl::ext::oneapi::experimental;

  constexpr std::size_t wg_size = 16;

  queue q;

  buffer<int, 1> input_buf{wg_size * 2};
  {
    host_accessor acc{input_buf};
    std::iota(acc.begin(), acc.end(), 0);
  }
  buffer<bool, 1> success_buf{wg_size};

  q.submit([&](handler &cgh) {
    accessor input{input_buf, cgh};
    accessor success{success_buf, cgh};
    cgh.parallel_for(nd_range<1>{wg_size, wg_size}, [=](nd_item<1> ndi) {
      auto gid = ndi.get_global_id(0);
      auto g = ndi.get_group();

      S data[2];
      sycl_exp::group_load(g, input.begin(), span{data});

      bool ok = true;
      ok &= (data[0].i == gid * 2 + 0 + 42);
      ok &= (data[1].i == gid * 2 + 1 + 42);
      success[gid] = ok;
    });
  });

  for (bool wi_success : host_accessor{success_buf})
    assert(wi_success);

  return 0;
}
