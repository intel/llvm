// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// UNSUPPORTED: gpu

// This e2e test checks the alignment of the annotated USM allocation (host &
// device) in various cases

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

  auto MDevice = [&](auto... args) {
    return malloc_device_annotated(N, args...).get();
  };
  auto MHost = [&](auto... args) {
    return malloc_host_annotated(N, args...).get();
  };

  auto MAnnotated = [&](auto... args) {
    return malloc_annotated(N, args...).get();
  };

  auto ADevice = [&](size_t align, auto... args) {
    return aligned_alloc_device_annotated(align, N, args...).get();
  };
  auto AHost = [&](size_t align, auto... args) {
    return aligned_alloc_host_annotated(align, N, args...).get();
  };

  auto AAnnotated = [&](size_t align, auto... args) {
    return aligned_alloc_annotated(align, N, args...).get();
  };

  CheckAlignAll(
      8,
      std::tuple{
          // Case: `malloc_xxx` with no alignment constraint, this includes
          // no alignment property
          [&]() { return MDevice(q); },
          [&]() { return MDevice(q, PL); },
          [&]() { return MDevice(dev, Ctx); },
          [&]() { return MDevice(dev, Ctx, PL); },
          [&]() { return MHost(q); },
          [&]() { return MHost(q, PL); },
          [&]() { return MHost(Ctx); },
          [&]() { return MHost(Ctx, PL); },
          [&]() { return MAnnotated(dev, Ctx, alloc::device); },
          [&]() { return MAnnotated(dev, Ctx, alloc::device, PL); },
          [&]() { return MAnnotated(dev, Ctx, alloc::host); },
          [&]() { return MAnnotated(dev, Ctx, alloc::host, PL); },
          [&]() { return MAnnotated(dev, Ctx, properties{usm_kind_device}); },
          [&]() { return MAnnotated(dev, Ctx, properties{usm_kind_host}); }

          // Case: `aligned_alloc_xxx` with no alignment constraint, this
          // includes
          // 1. no alignment property
          // 2. alignment argument is 0
          ,
          [&]() { return ADevice(0, q); },
          [&]() { return ADevice(0, q, PL); },
          [&]() { return ADevice(0, dev, Ctx); },
          [&]() { return ADevice(0, dev, Ctx, PL); },
          [&]() { return AHost(0, q); },
          [&]() { return AHost(0, q, PL); },
          [&]() { return AHost(0, Ctx); },
          [&]() { return AHost(0, Ctx, PL); },
          [&]() { return AAnnotated(0, dev, Ctx, alloc::device); },
          [&]() { return AAnnotated(0, dev, Ctx, alloc::device, PL); },
          [&]() { return AAnnotated(0, dev, Ctx, alloc::host); },
          [&]() { return AAnnotated(0, dev, Ctx, alloc::host, PL); }});

  // Case: `aligned_alloc_xxx` with alignment argument (power of 2) and no
  // alignment property
  CheckAlignAll(
      std::max(align, (size_t)8),
      std::tuple{[&]() { return ADevice(align, q); },
                 [&]() { return ADevice(align, dev, Ctx); },
                 [&]() { return AHost(align, q); },
                 [&]() { return AHost(align, Ctx); },
                 [&]() { return AAnnotated(align, dev, Ctx, alloc::device); },
                 [&]() { return AAnnotated(align, dev, Ctx, alloc::host); }});

  // Case: `malloc_xxx<T>` with no alignment constraint, the alignment is
  // alignof(T)
  auto TDevice = [&](auto... args) {
    return malloc_device_annotated<T>(N, args...).get();
  };
  auto THost = [&](auto... args) {
    return malloc_host_annotated<T>(N, args...).get();
  };
  auto TAnnotated = [&](auto... args) {
    return malloc_annotated<T>(N, args...).get();
  };

  auto ATDevice = [&](size_t align, auto... args) {
    return aligned_alloc_device_annotated<T>(align, N, args...).get();
  };
  auto ATHost = [&](size_t align, auto... args) {
    return aligned_alloc_host_annotated<T>(align, N, args...).get();
  };
  auto ATAnnotated = [&](size_t align, auto... args) {
    return aligned_alloc_annotated<T>(align, N, args...).get();
  };

  auto check_align_max = [&q](std::vector<size_t> Alignments, auto AllocFn,
                              int Line = __builtin_LINE(), int Case = 0) {
    assert(Alignments.size() > 0);
    size_t MaxVal = Alignments[0];
    for (int i = 1; i < Alignments.size(); i++)
      MaxVal = std::max(MaxVal, Alignments[i]);

    // First allocation might naturally be over-aligned. Do several of them to
    // do the verification;
    decltype(AllocFn()) Arr[10];
    for (auto *&Elem : Arr)
      Elem = AllocFn();
    for (auto *Ptr : Arr) {
      auto v = reinterpret_cast<uintptr_t>(Ptr);
      if ((v & (MaxVal - 1)) != 0) {
        std::cout << "Failed at line " << Line << ", case " << Case
                  << std::endl;
        assert(false && "Not properly aligned!");
        break; // To be used with commented out assert above.
      }
    }
    for (auto *Ptr : Arr)
      free(Ptr, q);
  };

  auto CheckAlignMaxOf = [&](std::vector<std::vector<size_t>> AlignVec,
                             auto Funcs, int Line = __builtin_LINE()) {
    std::apply(
        [&](auto... Fs) {
          int Case = 0, Idx = 0;
          (void)std::initializer_list<int>{
              (check_align_max(AlignVec[Idx++], Fs, Line, Case++), 0)...};
        },
        Funcs);
  };

  CheckAlignAll(
      alignof(T),
      std::tuple{
          // Case: `malloc_xxx<T>` with no alignment constraint, this includes
          // no alignment property
          [&]() { return TDevice(q); },
          [&]() { return TDevice(q, PL); },
          [&]() { return TDevice(dev, Ctx); },
          [&]() { return TDevice(dev, Ctx, PL); },
          [&]() { return THost(q); },
          [&]() { return THost(q, PL); },
          [&]() { return THost(Ctx); },
          [&]() { return THost(Ctx, PL); },
          [&]() { return TAnnotated(dev, Ctx, alloc::device); },
          [&]() { return TAnnotated(dev, Ctx, alloc::device, PL); },
          [&]() { return TAnnotated(dev, Ctx, alloc::host); },
          [&]() { return TAnnotated(dev, Ctx, alloc::host, PL); },
          [&]() { return TAnnotated(dev, Ctx, properties{usm_kind_device}); },
          [&]() { return TAnnotated(dev, Ctx, properties{usm_kind_host}); }

          // Case: `aligned_alloc_xxx<T>` with no alignment constraint, this
          // includes
          // 1. no alignment property
          // 2. alignment argument is 0
          ,
          [&]() { return ATDevice(0, q); },

          [&]() { return ATDevice(0, dev, Ctx); },

          [&]() { return ATHost(0, q); },

          [&]() { return ATHost(0, Ctx); },

          [&]() { return ATAnnotated(0, dev, Ctx, alloc::device); },

          [&]() { return ATAnnotated(0, dev, Ctx, alloc::host); },
      });

  // Case: aligned_alloc_xxx<T> with alignment argument K (K is a power of 2)
  // and no alignment properties, the alignment is the least-common-multiple of
  // K and alignof(T)
  CheckAlignAll(
      std::max(align, alignof(T)),
      std::tuple{[&]() { return ATDevice(align, q); },
                 [&]() { return ATDevice(align, dev, Ctx); },
                 [&]() { return ATHost(align, q); },
                 [&]() { return ATHost(align, Ctx); },
                 [&]() { return ATAnnotated(align, dev, Ctx, alloc::device); },
                 [&]() { return ATAnnotated(align, dev, Ctx, alloc::host); }});

  // Case: aligned_alloc_xxx<T> with alignment argument K (K is a power of 2)
  // and no alignment properties, the alignment is the maximum value between
  // K and alignof(T)
  CheckAlignMaxOf(
      {
          {2, alignof(T)},
          {4, alignof(T)},
          {8, alignof(T)},
          {16, alignof(T)},
          {128, alignof(T)},
          {256, alignof(T)},
      },
      std::tuple{
          [&]() { return ATDevice(2, q); }, [&]() { return ATDevice(4, q); },
          [&]() { return ATDevice(8, q); }, [&]() { return ATDevice(16, q); },
          [&]() { return ATDevice(128, q); },
          [&]() { return ATDevice(256, q); }});

  // Test cases that are expected to return null
  auto check_null = [&q](auto AllocFn, int Line = __builtin_LINE(),
                         int Case = 0) {
    decltype(AllocFn()) Ptr = AllocFn();
    auto v = reinterpret_cast<uintptr_t>(Ptr);
    if (v != 0) {
      free(Ptr, q);
      std::cout << "Failed at line " << Line << ", case " << Case << std::endl;
      assert(false && "The return is not null!");
    }
  };

  auto CheckNullAll = [&](auto Funcs, int Line = __builtin_LINE()) {
    std::apply(
        [&](auto... Fs) {
          int Case = 0;
          (void)std::initializer_list<int>{
              (check_null(Fs, Line, Case++), 0)...};
        },
        Funcs);
  };

  CheckNullAll(std::tuple{
      // Case: aligned_alloc_xxx with no alignment property, and the alignment
      // argument is not a power of 2, the result is nullptr
      [&]() { return ADevice(3, q); }, [&]() { return ADevice(7, dev, Ctx); },
      [&]() { return AHost(7, q); }, [&]() { return AHost(9, Ctx); },
      [&]() { return AAnnotated(15, q, alloc::device); },
      [&]() { return AAnnotated(17, dev, Ctx, alloc::host); }

      // Case: aligned_alloc_xxx<T> with no alignment property, and the
      // alignment
      // argument is not a power of 2, the result is nullptr
      ,
      [&]() { return ATDevice(65, q); },
      [&]() { return ATDevice(70, dev, Ctx); },
      [&]() { return ATHost(174, q); }, [&]() { return ATHost(299, Ctx); },
      [&]() { return ATAnnotated(550, q, alloc::device); },
      [&]() { return ATAnnotated(1700, dev, Ctx, alloc::host); }});
}

struct alignas(64) MyStruct {
  int x;
};

int main() {
  sycl::queue q;
  testAlign<char>(q, 4);
  testAlign<int>(q, 128);
  testAlign<std::complex<double>>(q, 4);
  testAlign<MyStruct>(q, 4);
  return 0;
}