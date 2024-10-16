// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/experimental/group_load_store.hpp>

struct S {
  operator int() const { return i + 42; }

  int i = 0;
};
static_assert(std::is_trivially_copyable_v<S>);

int main() {
  using namespace sycl;
  namespace sycl_exp = sycl::ext::oneapi::experimental;

  constexpr std::size_t wg_size = 16;

  queue q;

  buffer<int, 1> output_buf{wg_size * 2};

  q.submit([&](handler &cgh) {
    accessor output{output_buf, cgh};
    cgh.parallel_for(nd_range<1>{wg_size, wg_size}, [=](nd_item<1> ndi) {
      auto gid = ndi.get_global_id(0);
      auto g = ndi.get_group();

      S data[2];
      data[0].i = gid * 2 + 0;
      data[1].i = gid * 2 + 1;
      sycl_exp::group_store(g, span{data}, output.begin());
    });
  });

  host_accessor output{output_buf};
  for (int i = 0; i < wg_size * 2; ++i) {
    assert(output[i] == i + 42);
  }

  return 0;
}
