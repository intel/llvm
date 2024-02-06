// Test fusion works with reductions. Only algorithms automatically selected by
// `sycl::reduction` are supported.

#include <sycl/sycl.hpp>

#include "../helpers.hpp"
#include "sycl/detail/reduction_forward.hpp"

using namespace sycl;

constexpr inline size_t globalSize = 512;

template <detail::reduction::strategy Strategy> struct is_fusion_supported {
  constexpr static inline bool value =
      detail::reduction::strategy::group_reduce_and_last_wg_detection <=
          Strategy &&
      Strategy < detail::reduction::strategy::group_reduce_and_atomic_cross_wg;
};

template <detail::reduction::strategy Strategy>
constexpr inline bool is_fusion_supported_v =
    is_fusion_supported<Strategy>::value;

template <detail::reduction::strategy Strategy, bool Fuse>
void test(nd_range<1> ndr) {
  static_assert(is_fusion_supported_v<Strategy>,
                "Testing unsupported algorithm");
  std::array<int, globalSize> data;
  int sumRes = 0;
  int maxRes = 0;

  {
    queue q{ext::codeplay::experimental::property::queue::enable_fusion{}};

    buffer<int> dataBuf{data};
    buffer<int> sumBuf{&sumRes, 1};
    buffer<int> maxBuf{&maxRes, 1};

    ext::codeplay::experimental::fusion_wrapper fw{q};

    fw.start_fusion();
    iota(q, dataBuf, 0);

    q.submit([&](handler &cgh) {
      accessor in(dataBuf, cgh, read_only);
      auto sumRed = reduction(sumBuf, cgh, plus<>{},
                              property::reduction::initialize_to_identity{});
      detail::reduction_parallel_for<detail::auto_name, Strategy>(
          cgh, ndr, ext::oneapi::experimental::empty_properties_t{}, sumRed,
          [=](nd_item<1> Item, auto &Red) {
            Red.combine(in[Item.get_global_id()]);
          });
    });

    q.submit([&](handler &cgh) {
      accessor in(dataBuf, cgh, read_only);
      auto maxRed = reduction(maxBuf, cgh, maximum<>{},
                              property::reduction::initialize_to_identity{});
      detail::reduction_parallel_for<detail::auto_name, Strategy>(
          cgh, ndr, ext::oneapi::experimental::empty_properties_t{}, maxRed,
          [=](nd_item<1> Item, auto &Red) {
            Red.combine(in[Item.get_global_id()]);
          });
    });

    if constexpr (Fuse) {
      fw.complete_fusion(ext::codeplay::experimental::property::no_barriers{});
    } else {
      fw.cancel_fusion();
    }
  }

  constexpr int expectedMax = globalSize - 1;
  constexpr int expectedSum = globalSize * expectedMax / 2;

  assert(sumRes == expectedSum);
  assert(maxRes == expectedMax);
}

template <detail::reduction::strategy Strategy> void test() {
  for (size_t localSize = 1; localSize <= globalSize; localSize *= 2) {
    nd_range<1> ndr{globalSize, localSize};
    test<Strategy, true>(ndr);
    test<Strategy, false>(ndr);
  }
}
