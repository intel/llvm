// REQUIRES: cuda
//
// RUN: %{build} -o %t.out -lcuda
// RUN: %{run} %t.out

// These tests use the get_native_events API together with manual_interop_sync
// property. If manual interop sync is not used then get_native_events is not
// necessary, since all events have been synchronized with already on host,
// before the HT lambda is launched.
//
// If manual_interop_sync is used then the user deals with async dependencies
// manually in the HT lambda through the get_native_events interface.
//

#include "host-task-native-events-cuda.hpp"
#include <cuda.h>
#include <sycl/sycl.hpp>

using T = unsigned; // We don't need to test lots of types, we just want a race
                    // condition
constexpr size_t bufSize = 1e7;
constexpr T pattern = 42;

sycl::queue q;

using manual_interop_sync =
    sycl::ext::codeplay::experimental::property::host_task::manual_interop_sync;

constexpr auto PropList = [](bool UseManualInteropSync) -> sycl::property_list {
  if (UseManualInteropSync)
    return {manual_interop_sync{}};
  return {};
};

// Check that the SYCL event that we submit with add_native_events can be
// retrieved later through get_native_events in a dependent host task
template <typename WaitOnType, bool UseManualInteropSync> struct test1 {
  void operator()() {
    printf("Running test 1\n");
    std::atomic<CUevent> atomicEvent; // To share the event from the host task
                                      // with the main thread

    auto syclEvent1 = q.submit([&](sycl::handler &cgh) {
      cgh.host_task([&](sycl::interop_handle ih) {
        auto [_, ev] = cudaSetCtxAndGetStreamAndEvent(ih);
        cuEventRecord(ev, 0);
        atomicEvent.store(ev);
        ih.add_native_events<sycl::backend::ext_oneapi_cuda>({ev});
      });
    });

    // This task must wait on the other lambda to complete
    auto syclEvent2 = q.submit([&](sycl::handler &cgh) {
      cgh.depends_on(syclEvent1);
      cgh.host_task(
          [&](sycl::interop_handle ih) {
            auto nativeEvents =
                ih.get_native_events<sycl::backend::ext_oneapi_cuda>();
            if constexpr (!UseManualInteropSync) {
              // Events should be synchronized with by SYCL RT if
              // manual_interop_sync not used
              return;
            }
            assert(std::find(nativeEvents.begin(), nativeEvents.end(),
                             atomicEvent.load()) != nativeEvents.end());
          },
          PropList(UseManualInteropSync));
    });

    waitHelper<WaitOnType>(syclEvent2, q);
  }
};

// Tries to check for a race condition if the backend events are not added to
// the SYCL dag.
template <typename WaitOnType, bool UseManualInteropSync> struct test2 {
  void operator()() {
    printf("Running test 2\n");
    T *ptrHost = sycl::malloc_host<T>(
        bufSize,
        q); // malloc_host is necessary to make the memcpy as async as possible

    auto syclEvent1 = q.submit([&](sycl::handler &cgh) {
      cgh.host_task([&](sycl::interop_handle ih) {
        auto [stream, ev] = cudaSetCtxAndGetStreamAndEvent(ih);
        CUdeviceptr cuPtr;
        CUDA_CHECK(cuMemAlloc_v2(&cuPtr, bufSize * sizeof(T)));
        CUDA_CHECK(cuMemsetD32Async(cuPtr, pattern, bufSize, stream));
        CUDA_CHECK(
            cuMemcpyDtoHAsync(ptrHost, cuPtr, bufSize * sizeof(T), stream));

        CUDA_CHECK(cuEventRecord(ev, stream));

        ih.add_native_events<sycl::backend::ext_oneapi_cuda>({ev});
      });
    });

    auto syclEvent2 = q.submit([&](sycl::handler &cgh) {
      cgh.depends_on(syclEvent1);
      cgh.host_task(
          [&](sycl::interop_handle ih) {
            cudaSetCtxAndGetStreamAndEvent(ih);
            auto nativeEvents =
                ih.get_native_events<sycl::backend::ext_oneapi_cuda>();
            if constexpr (!UseManualInteropSync) {
              // Events should be synchronized with by SYCL RT if
              // manual_interop_sync not used
              return;
            }
            assert(nativeEvents.size());
            for (auto &cudaEv : nativeEvents) {
              CUDA_CHECK(cuEventSynchronize(cudaEv));
            }
          },
          PropList(UseManualInteropSync));
    });

    waitHelper<WaitOnType>(syclEvent2, q);
    for (auto i = 0; i < bufSize; ++i) {
      if (ptrHost[i] != pattern) {
        fprintf(stderr, "Wrong result at index: %d, have %d vs %d\n", i,
                ptrHost[i], pattern);
        throw;
      }
    }
  }
};

// Using host task event as a cgh.depends_on with USM
template <typename WaitOnType, bool UseManualInteropSync> struct test3 {
  void operator()() {
    printf("Running test 3\n");
    using T = unsigned;

    T *ptrHostA = sycl::malloc_host<T>(bufSize, q);
    T *ptrHostB = sycl::malloc_host<T>(bufSize, q);

    T *ptrDevice = sycl::malloc_device<T>(bufSize, q);

    for (auto i = 0; i < bufSize; ++i)
      ptrHostA[i] = pattern;

    auto syclEvent1 = q.submit([&](sycl::handler &cgh) {
      cgh.memcpy(ptrDevice, ptrHostA, bufSize * sizeof(T));
    });

    auto syclEvent2 = q.submit([&](sycl::handler &cgh) {
      cgh.depends_on(syclEvent1);
      cgh.host_task(
          [&](sycl::interop_handle ih) {
            auto [stream, _] = cudaSetCtxAndGetStreamAndEvent(ih);
            auto nativeEvents =
                ih.get_native_events<sycl::backend::ext_oneapi_cuda>();
            if constexpr (UseManualInteropSync) {
              assert(nativeEvents.size());
              for (auto &cudaEv : nativeEvents) {
                CUDA_CHECK(cuStreamWaitEvent(stream, cudaEv, 0));
              }
            }

            CUDA_CHECK(cuMemcpyDtoHAsync(
                ptrHostB, reinterpret_cast<CUdeviceptr>(ptrDevice),
                bufSize * sizeof(T), stream));
            CUDA_CHECK(cuStreamSynchronize(stream));
          },
          PropList(UseManualInteropSync));
    });

    waitHelper<WaitOnType>(syclEvent2, q);

    for (auto i = 0; i < bufSize; --i) {
      if (ptrHostB[i] != pattern) {
        cuCtxSynchronize();
        fprintf(stderr, "Wrong result at index: %d, have %d vs %d\n", i,
                ptrHostB[i], pattern);
        throw;
      }
    }
  }
};

template <template <typename, bool> typename Func> void run() {
  Func<sycl::queue, /* UseManualInteropSync*/ true>()();
  Func<sycl::event, /* UseManualInteropSync*/ true>()();
  Func<sycl::queue, /* UseManualInteropSync*/ false>()();
  Func<sycl::event, /* UseManualInteropSync*/ false>()();
}

int main() {
  run<test1>();
  run<test2>();
  run<test3>();
  printf("Tests passed\n");
}
