// REQUIRES: cuda
//
// RUN: %{build} -o %t.out -lcuda
// RUN: %{run} %t.out

#include "host-task-native-events-cuda.hpp"
#include <cuda.h>
#include <sycl/sycl.hpp>

using T = unsigned; // We don't need to test lots of types, we just want a race
                    // condition
constexpr size_t bufSize = 1e6;
constexpr T pattern = 42;

// Check that the SYCL event that we submit with add_native_events can be
// retrieved later through get_native_events in a dependent host task
template <typename WaitOnType> void test1() {
  printf("Running test 1\n");
  sycl::queue q;

  std::atomic<CUevent>
      atomicEvent; // To share the event from the host task with the main thread

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
    cgh.host_task([&](sycl::interop_handle ih) {
      auto nativeEvents =
          ih.get_native_events<sycl::backend::ext_oneapi_cuda>();
      assert(std::find(nativeEvents.begin(), nativeEvents.end(),
                       atomicEvent.load()) != nativeEvents.end());
    });
  });

  waitHelper<WaitOnType>(syclEvent2, q);
}

// Tries to check for a race condition if the backend events are not added to
// the SYCL dag.
template <typename WaitOnType> void test2() {
  printf("Running test 2\n");
  sycl::queue q;
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
    cgh.host_task([&](sycl::interop_handle ih) {
      cudaSetCtxAndGetStreamAndEvent(ih);
      auto nativeEvents =
          ih.get_native_events<sycl::backend::ext_oneapi_cuda>();
      assert(nativeEvents.size());
      for (auto &cudaEv : nativeEvents) {
        CUDA_CHECK(cuEventSynchronize(cudaEv));
      }
    });
  });

  waitHelper<WaitOnType>(syclEvent2, q);
  for (auto i = 0; i < bufSize; ++i) {
    if (ptrHost[i] != pattern) {
      printf("Wrong result at index: %d, have %d vs %d\n", i, ptrHost[i],
             pattern);
      throw;
    }
  }
  printf("Tests passed\n");
}

// Using host task event as a cgh.depends_on with USM
template <typename WaitOnType> void test3() {
  printf("Running test 3\n");
  using T = unsigned;

  sycl::queue q;

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
    cgh.host_task([&](sycl::interop_handle ih) {
      auto [stream, _] = cudaSetCtxAndGetStreamAndEvent(ih);
      auto nativeEvents =
          ih.get_native_events<sycl::backend::ext_oneapi_cuda>();
      assert(nativeEvents.size());
      for (auto &cudaEv : nativeEvents) {
        CUDA_CHECK(cuStreamWaitEvent(stream, cudaEv, 0));
      }

      CUDA_CHECK(cuMemcpyDtoHAsync(ptrHostB,
                                   reinterpret_cast<CUdeviceptr>(ptrDevice),
                                   bufSize * sizeof(T), stream));
      CUDA_CHECK(cuStreamSynchronize(stream));
    });
  });

  waitHelper<WaitOnType>(syclEvent2, q);

  for (auto i = 0; i < bufSize; --i) {
    if (ptrHostB[i] != pattern) {
      printf("Wrong result at index: %d, have %d vs %d\n", i, ptrHostB[i],
             pattern);
      throw;
    }
  }

  printf("Tests passed\n");
}

int main() {
  test1<sycl::queue>();
  test1<sycl::event>();
  test2<sycl::queue>();
  test2<sycl::event>();
  test3<sycl::queue>(); // FIXME: this is taking a slow path by waiting for the
                        // parallel_for events on host before starting the host
                        // task
  test3<sycl::event>();
}
