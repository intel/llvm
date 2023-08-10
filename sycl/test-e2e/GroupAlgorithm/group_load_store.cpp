// Use per-kernel split as a workaroud for a miscompilation bug in IGC.
// RUN: %clangxx -fsycl -fsycl-targets=%{sycl_triple} -fsycl-device-code-split=per_kernel %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out

#include <sycl/sycl.hpp>

#include <algorithm>
#include <iomanip>

using namespace sycl;
namespace sycl_ext = sycl::ext::oneapi::experimental;

constexpr int SG_SIZE = 16;
constexpr int N_WGS = 3;

template <typename MemType, typename SpanType, int ELEMS_PER_WI,

          sycl_ext::group_algorithm_data_placement data_placement,
          memory_scope scope>
void test(size_t wg_size) {
  std::cout << __PRETTY_FUNCTION__ << std::endl;

  size_t global_size = wg_size * N_WGS;
  queue q;

  buffer<MemType, 1> b(global_size * ELEMS_PER_WI);
  buffer<bool, 1> result(global_size);

  q.submit([&](handler &cgh) {
    accessor acc{b, cgh};
    accessor res{result, cgh};

    static constexpr auto scratch_mem_per_wi = ELEMS_PER_WI * sizeof(MemType);
    // sycl_ext::memory_required<MemType, ELEMS_PER_WI>(scope, block_size);
    local_accessor<std::byte> scratch(scratch_mem_per_wi * wg_size, cgh);

    cgh.parallel_for(
        nd_range{range<1>{global_size}, range<1>{wg_size}},
        [=](nd_item<1> ndi) [[intel::reqd_sub_group_size(SG_SIZE)]] {
          bool success = true;

          auto gid = ndi.get_global_id(0);
          auto init = static_cast<long long>(gid) %
                      (1 << (std::min<size_t>(sizeof(MemType), 4) * 8 - 2));
          auto *global_mem =
              acc.template get_multi_ptr<access::decorated::yes>()
                  .get_decorated();
          auto *group_mem =
              global_mem + ndi.get_group(0) * wg_size * ELEMS_PER_WI;

          auto *scratch_mem = &scratch[0];
          if constexpr (scope == memory_scope::sub_group) {
            auto sg = ndi.get_sub_group();
            group_mem += sg.get_group_id() * SG_SIZE * ELEMS_PER_WI;
            scratch_mem += sg.get_group_id() * SG_SIZE * scratch_mem_per_wi;
          }
          auto g = [&]() {
            if constexpr (scope == memory_scope::sub_group)
              return ndi.get_sub_group();
            else
              return ndi.get_group();
          }();

          auto group_lid = g.get_local_linear_id();

          constexpr bool blocked =
              data_placement ==
              sycl_ext::group_algorithm_data_placement::blocked;

          auto idx = [&](int elem_idx) {
            if constexpr (blocked) {
              return group_lid * ELEMS_PER_WI + elem_idx;
            } else {
              return group_lid + g.get_local_range().size() * elem_idx;
            }
          };

          SpanType arr[ELEMS_PER_WI];
          auto s = span<SpanType, ELEMS_PER_WI>(arr, ELEMS_PER_WI);

          auto data_placement_prop =
              sycl_ext::property::data_placement<data_placement>;
          auto props = sycl_ext::properties(data_placement_prop);

          {
            for (int i = 0; i < ELEMS_PER_WI; ++i)
              group_mem[idx(i)] = init - i;

            group_load(g, group_mem, s, props);

            for (int i = 0; i < ELEMS_PER_WI; ++i)
              success &= (s[i] == init - i);
          }

          {
            for (int i = 0; i < ELEMS_PER_WI; ++i)
              s[i] = init - i + 1;

            group_store(g, s, group_mem, props);

            for (int i = 0; i < ELEMS_PER_WI; ++i)
              success &= group_mem[idx(i)] == init - i + 1;
          }

          sycl_ext::group_with_scratchpad gh{
              g, span(scratch_mem,
                      g.get_local_range().size() * scratch_mem_per_wi)};

          {
            for (int i = 0; i < ELEMS_PER_WI; ++i)
              group_mem[idx(i)] = init - i;

            group_load(gh, group_mem, s, props);

            for (int i = 0; i < ELEMS_PER_WI; ++i)
              success &= (s[i] == init - i);
          }

          {
            for (int i = 0; i < ELEMS_PER_WI; ++i)
              s[i] = init - i + 1;

            group_store(gh, s, group_mem, props);

            for (int i = 0; i < ELEMS_PER_WI; ++i)
              success &= group_mem[idx(i)] == init - i + 1;
          }

          res[gid] = success;
        });
  });

  host_accessor res_acc{result};
  bool success =
      std::all_of(res_acc.begin(), res_acc.end(), [](bool r) { return r; });
  if constexpr (true)
    assert(success);
  else
    std::cout << "Test success: " << std::boolalpha << success << std::endl;
}

struct S1 {
  S1() = default;
  S1(int i) : i(i) {}
  operator int() { return i; }
  void operator+=(int inc) { i += inc; }
  int i = 0;
  int j = 2;
};
static_assert(sizeof(S1) == 8);

struct S2 {
  S2() = default;
  S2(int i) : i(i) {}
  operator int() { return i; }
  void operator+=(int inc) { i += inc; }
  int i = 0;
  int j = 2;
  int k = 3;
};
static_assert(sizeof(S2) == 12);

struct __attribute__((packed)) S3 {
  S3() = default;
  S3(int i) : i(i) {}
  operator int() { return i; }
  void operator+=(int inc) { i += inc; }
  int i = 42;
  int j = 2;
  char k = 3;
};
static_assert(sizeof(S3) == 9);

template <typename MemType, typename SpanType, int ELEMS_PER_WI>
void test_type_combo(size_t wg_size) {
  constexpr auto blocked = sycl_ext::group_algorithm_data_placement::blocked;
  constexpr auto striped = sycl_ext::group_algorithm_data_placement::striped;
  constexpr auto sg = memory_scope::sub_group;
  constexpr auto wg = memory_scope::work_group;

  test<MemType, SpanType, ELEMS_PER_WI, blocked, sg>(wg_size);
  test<MemType, SpanType, ELEMS_PER_WI, blocked, wg>(wg_size);
  test<MemType, SpanType, ELEMS_PER_WI, striped, sg>(wg_size);
  test<MemType, SpanType, ELEMS_PER_WI, striped, wg>(wg_size);
}

int main() {
#ifdef SINGLE
  using T = char;
  test<T, T, 3, sycl_ext::group_algorithm_data_placement::striped,
       memory_scope::work_group>(SG_SIZE * 3);
#else
  size_t wg_sizes[] = {SG_SIZE / 2, SG_SIZE, SG_SIZE * 3 / 2, SG_SIZE * 3};
  for (auto wg_size : wg_sizes) {
    std::cout << "WG_SIZE: " << wg_size << std::endl;
    constexpr int sizes[] = {1, 2, 3, 4, 7, 8, 16, 17, 31, 32, 64, 67};
    sycl::detail::loop<std::size(sizes)>([&](auto i) {
      constexpr int size = sizes[i];
      test_type_combo<char, char, size>(wg_size);
      test_type_combo<short, short, size>(wg_size);
      test_type_combo<int, int, size>(wg_size);
      test_type_combo<long long, long long, size>(wg_size);
      test_type_combo<S1, S1, size>(wg_size);
      test_type_combo<S2, S2, size>(wg_size);

      // Disabled due to an IGC bug resulting in miscompilations.
      // test_type_combo<S3, S3, size>(wg_size);
    });
  }
#endif

  return 0;
}
