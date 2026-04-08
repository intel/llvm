// REQUIRES: cpu
// RUN: %{build} -Wno-error=deprecated-declarations -fsycl-id-queries-range=none -o %t.out
// RUN: %{run} %t.out
// Test legacy compatibility flag
// RUN: %{build} -Wno-error=deprecated-declarations -fno-sycl-id-queries-fit-in-int -o %t_legacy.out
// RUN: %{run} %t_legacy.out

#include <climits>
#include <iostream>
#include <sycl/detail/core.hpp>

namespace S = sycl;

int main(void) {
  auto EH = [](S::exception_list EL) {
    for (const std::exception_ptr &E : EL) {
      throw E;
    }
  };

  S::queue Queue(EH);

  int Data = 0;
  S::buffer<int, 1> Buf{&Data, 1};

  // In "none" mode, no range validation should occur
  // These large values would throw in INT or UINT mode, but should succeed here

  // Test 1: Range exceeding INT_MAX should succeed
  static constexpr size_t LargeSize = static_cast<size_t>(INT_MAX) + 1024;
  S::range<1> LargeRange{LargeSize};

  try {
    Queue.submit([&](S::handler &CGH) {
      auto Acc = Buf.get_access<sycl::access::mode::read_write>(CGH);

      CGH.parallel_for<class PF_LARGE_RANGE>(LargeRange, [Acc](S::id<1> Id) {
        if (Id[0] == 0)
          Acc[0] += 1;
      });
    });
  } catch (S::exception &E) {
    std::cerr << "Unexpected exception: " << E.what() << std::endl;
    assert(false &&
           "No exception should be thrown in 'none' mode for large ranges");
  }

  // Test 2: Offset exceeding INT_MAX should succeed
  S::id<1> LargeOffset{LargeSize};
  S::range<1> SmallRange{1};

  try {
    Queue.submit([&](S::handler &CGH) {
      auto Acc = Buf.get_access<sycl::access::mode::read_write>(CGH);

      CGH.parallel_for<class PF_LARGE_OFFSET>(
          SmallRange, LargeOffset, [Acc](S::id<1> Id) { Acc[0] += 1; });
    });
  } catch (S::exception &E) {
    std::cerr << "Unexpected exception: " << E.what() << std::endl;
    assert(false &&
           "No exception should be thrown in 'none' mode for large offsets");
  }

  // Test 3: Range exceeding UINT_MAX should also succeed
  if constexpr (sizeof(size_t) > sizeof(unsigned int)) {
    static constexpr size_t VeryLargeSize =
        static_cast<size_t>(UINT_MAX) + 1024;
    S::range<1> VeryLargeRange{VeryLargeSize};

    try {
      Queue.submit([&](S::handler &CGH) {
        auto Acc = Buf.get_access<sycl::access::mode::read_write>(CGH);

        CGH.parallel_for<class PF_VERY_LARGE_RANGE>(VeryLargeRange,
                                                    [Acc](S::id<1> Id) {
                                                      if (Id[0] == 0)
                                                        Acc[0] += 1;
                                                    });
      });
    } catch (S::exception &E) {
      std::cerr << "Unexpected exception: " << E.what() << std::endl;
      assert(
          false &&
          "No exception should be thrown in 'none' mode for very large ranges");
    }
  }

  // Test 4: Product of dimensions exceeding limits should succeed
  static constexpr size_t ModerateSize = static_cast<size_t>(INT_MAX) / 2 + 1;
  S::range<2> ProductExceedsLimits{ModerateSize, 3};

  try {
    Queue.submit([&](S::handler &CGH) {
      auto Acc = Buf.get_access<sycl::access::mode::read_write>(CGH);

      CGH.parallel_for<class PF_PRODUCT_LARGE>(ProductExceedsLimits,
                                               [Acc](S::id<2> Id) {
                                                 if (Id[0] == 0 && Id[1] == 0)
                                                   Acc[0] += 1;
                                               });
    });
  } catch (S::exception &E) {
    std::cerr << "Unexpected exception: " << E.what() << std::endl;
    assert(false && "No exception should be thrown in 'none' mode");
  }
  return 0;
}
