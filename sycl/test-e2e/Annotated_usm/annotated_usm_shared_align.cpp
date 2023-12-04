// REQUIRES: aspect-usm_shared_allocations
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// UNSUPPORTED: gpu

// This e2e test checks the alignment of the annotated shared USM allocation in
// various cases

#include <sycl/sycl.hpp>

#include <algorithm>
#include <complex>
#include <numeric>

// clang-format on

using namespace sycl::ext::oneapi::experimental;
using namespace sycl::ext::intel::experimental;
using alloc = sycl::usm::alloc;

template <typename T> void testAlign(sycl::queue &q, size_t align) {
  const sycl::context &Ctx = q.get_context();
  auto dev = q.get_device();

  properties PL{buffer_location<0>, conduit};

  auto check_align = [&q](size_t Alignment, auto AllocFn,
                          int Line = __builtin_LINE(), int Case = 0) {
    // First allocation might naturally be over-aligned. Do several of them to
    // do the verification;
    decltype(AllocFn()) Arr[10];
    for (auto *&Elem : Arr)
      Elem = AllocFn();
    for (auto *Ptr : Arr) {
      auto v = reinterpret_cast<uintptr_t>(Ptr);
      if ((v & (Alignment - 1)) != 0) {
        std::cout << "Failed at line " << Line << ", case " << Case
                  << std::endl;
        assert(false && "Not properly aligned!");
        break; // To be used with commented out assert above.
      }
    }
    for (auto *Ptr : Arr)
      free(Ptr, q);
  };

  auto CheckAlignAll = [&](size_t Expected, auto Funcs,
                           int Line = __builtin_LINE()) {
    std::apply(
        [&](auto... Fs) {
          int Case = 0;
          (void)std::initializer_list<int>{
              (check_align(Expected, Fs, Line, Case++), 0)...};
        },
        Funcs);
  };

  constexpr int N = 10;
  assert(align > 0 || (align & (align - 1)) == 0);

  auto MShared = [&](auto... args) {
    return malloc_shared_annotated(N, args...).get();
  };
  auto MAnnotated = [&](auto... args) {
    return malloc_annotated(N, args...).get();
  };

  auto AShared = [&](size_t align, auto... args) {
    return aligned_alloc_shared_annotated(align, N, args...).get();
  };
  auto AAnnotated = [&](size_t align, auto... args) {
    return aligned_alloc_annotated(align, N, args...).get();
  };

  CheckAlignAll(
      8,
      std::tuple{
          // Case: `malloc_xxx` with no alignment constraint, this includes
          // no alignment property
          [&]() { return MShared(q); }, [&]() { return MShared(q, PL); },
          [&]() { return MShared(dev, Ctx); },
          [&]() { return MShared(dev, Ctx, PL); },
          [&]() { return MAnnotated(dev, Ctx, alloc::shared); },
          [&]() { return MAnnotated(dev, Ctx, alloc::shared, PL); },
          [&]() { return MAnnotated(dev, Ctx, properties{usm_kind_shared}); },
          [&]() { return MAnnotated(dev, Ctx, properties{usm_kind_shared}); }

          // Case: `aligned_alloc_xxx` with no alignment constraint, this
          // includes
          // 1. no alignment property or
          // 2. alignment argument is 0
          ,
          [&]() { return AShared(0, q); }, [&]() { return AShared(0, q, PL); },
          [&]() { return AShared(0, dev, Ctx); },
          [&]() { return AShared(0, dev, Ctx, PL); },
          [&]() { return AAnnotated(0, dev, Ctx, alloc::shared); },
          [&]() { return AAnnotated(0, dev, Ctx, alloc::shared, PL); }});

  // Case: `aligned_alloc_xxx` with alignment argument (power of 2) and no
  // alignment property
  CheckAlignAll(
      align,
      std::tuple{[&]() { return AShared(align, q); },
                 [&]() { return AShared(align, dev, Ctx); },
                 [&]() { return AAnnotated(align, dev, Ctx, alloc::shared); }});

  // Case: `malloc_xxx<T>` with no alignment constraint, the alignment is
  // alignof(T)
  auto TShared = [&](auto... args) {
    return malloc_shared_annotated<T>(N, args...).get();
  };
  auto TAnnotated = [&](auto... args) {
    return malloc_annotated<T>(N, args...).get();
  };

  auto ATShared = [&](size_t align, auto... args) {
    return aligned_alloc_shared_annotated<T>(align, N, args...).get();
  };
  auto ATAnnotated = [&](size_t align, auto... args) {
    return aligned_alloc_annotated<T>(align, N, args...).get();
  };

  CheckAlignAll(
      alignof(T),
      std::tuple{
          // Case: `malloc_xxx<T>` with no alignment constraint, this includes
          // no alignment property
          [&]() { return TShared(q); }, [&]() { return TShared(q, PL); },
          [&]() { return TShared(dev, Ctx); },
          [&]() { return TShared(dev, Ctx, PL); },
          [&]() { return TAnnotated(dev, Ctx, alloc::shared); },
          [&]() { return TAnnotated(dev, Ctx, alloc::shared, PL); },
          [&]() { return TAnnotated(dev, Ctx, properties{usm_kind_shared}); }

          // Case: `aligned_alloc_xxx<T>` with no alignment constraint, this
          // includes
          // 1. no alignment property
          // 2. alignment argument is 0
          ,
          [&]() { return ATShared(0, q); },
          [&]() { return ATShared(0, q, PL); },
          [&]() { return ATShared(0, dev, Ctx); },
          [&]() { return ATShared(0, dev, Ctx, PL); },
          [&]() { return ATAnnotated(0, dev, Ctx, alloc::shared); },
          [&]() { return ATAnnotated(0, dev, Ctx, alloc::shared, PL); }});

  // Case: aligned_alloc_xxx<T> with alignment argument K (K is a power of 2)
  // and no alignment properties, the alignment is the least-common-multiple of
  // K and alignof(T)
  CheckAlignAll(std::max(align, alignof(T)),
                std::tuple{[&]() { return ATShared(align, q); },
                           [&]() { return ATShared(align, dev, Ctx); },
                           [&]() {
                             return ATAnnotated(align, dev, Ctx, alloc::shared);
                           }});
}

int main() {
  sycl::queue q;
  testAlign<char>(q, 4);
  testAlign<int>(q, 128);
  testAlign<std::complex<double>>(q, 4);
  return 0;
}