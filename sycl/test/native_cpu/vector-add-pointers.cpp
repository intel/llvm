// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// REQUIRES: native_cpu_be
// RUN: %clangxx -fsycl -fsycl-targets=native_cpu %s -o %t
// RUN: env ONEAPI_DEVICE_SELECTOR="native_cpu:cpu" %t

#include <array>
#include <cstddef>

template <typename T, size_t N, size_t... Rest>
struct Array : std::array<Array<T, Rest...>, N> {
  using std::array<Array<T, Rest...>, N>::operator[];
  Array() = default;
  Array(T *data) { memcpy(this, data, sizeof(*this)); }
  Array(Array<T, N, Rest...> *data) { memcpy(this, data, sizeof(*this)); }
};

template <typename T, size_t N> struct Array<T, N> : std::array<T, N> {
  using std::array<T, N>::operator[];
  Array() = default;
  Array(T *data) { memcpy(this, data, sizeof(*this)); }
  Array(Array<T, N> *data) { memcpy(this, data, sizeof(*this)); }
};

#include <sycl/sycl.hpp>

using namespace ::sycl;

#define N 16

using _Array = Array<int, N>;

int main() {
  queue deviceQueue(default_selector_v);

  const device dev = deviceQueue.get_device();
  const context ctx = deviceQueue.get_context();

  auto A_acc = (_Array *)malloc_shared(sizeof(_Array), dev, ctx);
  auto B_acc = (_Array *)malloc_shared(sizeof(_Array), dev, ctx);
  auto OUT_acc = (_Array *)malloc_shared(sizeof(_Array), dev, ctx);

  deviceQueue.submit([&](handler &cgh) {
    auto kern = [=]() {
      for (int i = 0; i < N; i++) {
        (*OUT_acc)[i] = (*A_acc)[i] + (*B_acc)[i];
      }
    };
    cgh.single_task<class vec_add>(kern);
  });

  deviceQueue.wait();

  for (int i = 0; i < N; i++) {
    if ((*OUT_acc)[i] != (*A_acc)[i] + (*B_acc)[i])
      return 1;
  }

  sycl::free(OUT_acc, deviceQueue);
  sycl::free(B_acc, deviceQueue);
  sycl::free(A_acc, deviceQueue);

  return 0;
}
