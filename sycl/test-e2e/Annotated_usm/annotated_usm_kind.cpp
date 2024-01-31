// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// This e2e test checks the usm kind of the pointer returned by annotated USM
// allocation

#include <sycl/sycl.hpp>

#include <complex>
#include <numeric>

// clang-format on

using namespace sycl::ext::oneapi::experimental;
using namespace sycl::ext::intel::experimental;
using alloc = sycl::usm::alloc;

template <typename T> void testUsmKind(sycl::queue &q) {
  const sycl::context &Ctx = q.get_context();
  auto dev = q.get_device();

  auto check_usm_kind = [&](alloc Expected, auto AllocFn,
                            int Line = __builtin_LINE(), int Case = 0) {
    // First allocation might naturally be over-aligned. Do several of them to
    // do the verification;
    decltype(AllocFn()) Ptr = AllocFn();
    if (sycl::get_pointer_type(Ptr, Ctx) != Expected) {
      std::cout << "Failed at line " << Line << ", case " << Case << std::endl;
      assert(false && "Incorrect usm_kind!");
    }
    free(Ptr, q);
  };

  auto CheckUsmKindAll = [&](alloc Expected, auto Funcs,
                             int Line = __builtin_LINE()) {
    std::apply(
        [&](auto... Fs) {
          int Case = 0;
          (void)std::initializer_list<int>{
              (check_usm_kind(Expected, Fs, Line, Case++), 0)...};
        },
        Funcs);
  };

  constexpr int N = 11;

  auto MDevice = [&](auto... args) {
    return malloc_device_annotated(N, args...).get();
  };
  auto MHost = [&](auto... args) {
    return malloc_host_annotated(N, args...).get();
  };
  auto MShared = [&](auto... args) {
    return malloc_shared_annotated(N, args...).get();
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
  auto AShared = [&](size_t align, auto... args) {
    return aligned_alloc_shared_annotated(align, N, args...).get();
  };
  auto AAnnotated = [&](size_t align, auto... args) {
    return aligned_alloc_annotated(align, N, args...).get();
  };

  auto TDevice = [&](auto... args) {
    return malloc_device_annotated<T>(N, args...).get();
  };
  auto THost = [&](auto... args) {
    return malloc_host_annotated<T>(N, args...).get();
  };
  auto TShared = [&](auto... args) {
    return malloc_shared_annotated<T>(N, args...).get();
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
  auto ATShared = [&](size_t align, auto... args) {
    return aligned_alloc_shared_annotated<T>(align, N, args...).get();
  };
  auto ATAnnotated = [&](size_t align, auto... args) {
    return aligned_alloc_annotated<T>(align, N, args...).get();
  };

  CheckUsmKindAll(
      alloc::device,
      std::tuple{
          [&]() { return MDevice(q); }, [&]() { return MDevice(dev, Ctx); },
          [&]() { return MAnnotated(dev, Ctx, alloc::device); },
          [&]() { return MAnnotated(dev, Ctx, properties{usm_kind_device}); },
          [&]() { return ADevice(1, q); },
          [&]() { return ADevice(1, dev, Ctx); },
          [&]() { return AAnnotated(1, dev, Ctx, alloc::device); },
          [&]() { return TDevice(q); }, [&]() { return TDevice(dev, Ctx); },
          [&]() { return TAnnotated(dev, Ctx, alloc::device); },
          [&]() { return TAnnotated(dev, Ctx, properties{usm_kind_device}); },
          [&]() { return ATDevice(1, q); },
          [&]() { return ATDevice(1, dev, Ctx); },
          [&]() { return ATAnnotated(1, dev, Ctx, alloc::device); }});
  CheckUsmKindAll(
      alloc::host,
      std::tuple{
          [&]() { return MHost(q); }, [&]() { return MHost(Ctx); },
          [&]() { return MAnnotated(dev, Ctx, alloc::host); },
          [&]() { return MAnnotated(dev, Ctx, properties{usm_kind_host}); },
          [&]() { return AHost(1, q); }, [&]() { return AHost(1, Ctx); },
          [&]() { return AAnnotated(1, dev, Ctx, alloc::host); },
          [&]() { return THost(q); }, [&]() { return THost(Ctx); },
          [&]() { return TAnnotated(dev, Ctx, alloc::host); },
          [&]() { return TAnnotated(dev, Ctx, properties{usm_kind_host}); },
          [&]() { return ATHost(1, q); }, [&]() { return ATHost(1, Ctx); },
          [&]() { return ATAnnotated(1, dev, Ctx, alloc::host); }});

  if (q.get_device().has(sycl::aspect::usm_shared_allocations)) {
    CheckUsmKindAll(
        alloc::shared,
        std::tuple{
            [&]() { return MShared(q); }, [&]() { return MShared(dev, Ctx); },
            [&]() { return MAnnotated(dev, Ctx, alloc::shared); },
            [&]() { return MAnnotated(dev, Ctx, properties{usm_kind_shared}); },
            [&]() { return AShared(1, q); },
            [&]() { return AShared(1, dev, Ctx); },
            [&]() { return AAnnotated(1, dev, Ctx, alloc::shared); },
            [&]() { return TShared(q); }, [&]() { return TShared(dev, Ctx); },
            [&]() { return TAnnotated(dev, Ctx, alloc::shared); },
            [&]() { return TAnnotated(dev, Ctx, properties{usm_kind_shared}); },
            [&]() { return ATShared(1, q); },
            [&]() { return ATShared(1, dev, Ctx); },
            [&]() { return ATAnnotated(1, dev, Ctx, alloc::shared); }});
  }
}

int main() {
  sycl::queue q;
  testUsmKind<int>(q);
  return 0;
}
