// REQUIRES: aspect-ext_oneapi_atomic16

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// Checks atomic_ref<sycl::half> add/sub, which are emulated via a
// compare-exchange loop so they also work on devices without native half
// atomic add (e.g. devices that only support half atomic load/store and
// min/max).

#include <algorithm>
#include <iostream>
#include <vector>

#include <sycl/atomic_ref.hpp>
#include <sycl/detail/core.hpp>
#include <sycl/group_barrier.hpp>
#include <sycl/half_type.hpp>

using namespace sycl;

// All values stay below 2048, so every integer value is exactly representable
// in half and the results do not depend on the order of the updates.
constexpr size_t N = 1024;
constexpr size_t WGSize = 256;

template <memory_scope Scope = memory_scope::device,
          access::address_space Space = access::address_space::global_space>
using half_ref = atomic_ref<half, memory_order::relaxed, Scope, Space>;

int Errors = 0;

void check(const char *Name, float Got, float Expected) {
  if (Got != Expected) {
    std::cout << "FAILED " << Name << ": got " << Got << ", expected "
              << Expected << std::endl;
    ++Errors;
  }
}

// Every work-item must observe a distinct old value: {First, ..., First+N-1}.
void check_unique(const char *Name, std::vector<half> &Old, float First) {
  std::sort(Old.begin(), Old.end(),
            [](half A, half B) { return float(A) < float(B); });
  for (size_t I = 0; I < Old.size(); ++I) {
    if (float(Old[I]) != First + I) {
      std::cout << "FAILED " << Name << ": old values are not unique"
                << std::endl;
      ++Errors;
      return;
    }
  }
}

void test_fetch_add_sub(queue &Q) {
  half Values[2] = {0, N};
  std::vector<half> OldAdd(N), OldSub(N);
  {
    buffer<half> ValuesBuf(Values, 2);
    buffer<half> OldAddBuf(OldAdd.data(), N);
    buffer<half> OldSubBuf(OldSub.data(), N);
    Q.submit([&](handler &CGH) {
      accessor Val{ValuesBuf, CGH};
      accessor OldA{OldAddBuf, CGH, write_only, no_init};
      accessor OldS{OldSubBuf, CGH, write_only, no_init};
      CGH.parallel_for(range<1>(N), [=](id<1> I) {
        OldA[I] = half_ref<>(Val[0]).fetch_add(half(1));
        OldS[I] = half_ref<>(Val[1]).fetch_sub(half(1));
      });
    });
  }
  check("fetch_add", float(Values[0]), N);
  check("fetch_sub", float(Values[1]), 0);
  check_unique("fetch_add", OldAdd, 0);
  check_unique("fetch_sub", OldSub, 1);
}

void test_operators(queue &Q) {
  half Values[2] = {0, 0};
  {
    buffer<half> ValuesBuf(Values, 2);
    Q.submit([&](handler &CGH) {
      accessor Val{ValuesBuf, CGH};
      CGH.parallel_for(range<1>(N), [=](id<1>) {
        half_ref<>(Val[0]) += half(0.5f);
        half_ref<>(Val[1]) -= half(0.25f);
      });
    });
  }
  check("operator+=", float(Values[0]), N * 0.5f);
  check("operator-=", float(Values[1]), N * -0.25f);
}

void test_local(queue &Q) {
  std::vector<half> Results(N / WGSize);
  {
    buffer<half> ResultsBuf(Results.data(), Results.size());
    Q.submit([&](handler &CGH) {
      accessor Res{ResultsBuf, CGH, write_only, no_init};
      local_accessor<half, 1> Loc(2, CGH);
      CGH.parallel_for(nd_range<1>(N, WGSize), [=](nd_item<1> It) {
        if (It.get_local_id(0) == 0) {
          Loc[0] = 0;
          Loc[1] = WGSize;
        }
        group_barrier(It.get_group());
        using LocalRef = half_ref<memory_scope::work_group,
                                  access::address_space::local_space>;
        LocalRef(Loc[0]).fetch_add(half(1));
        LocalRef(Loc[1]).fetch_sub(half(1));
        group_barrier(It.get_group());
        if (It.get_local_id(0) == 0)
          Res[It.get_group(0)] = Loc[0] - Loc[1];
      });
    });
  }
  for (half R : Results)
    check("local fetch_add/fetch_sub", float(R), WGSize);
}

int main() {
  queue Q;
  test_fetch_add_sub(Q);
  test_operators(Q);
  test_local(Q);
  if (Errors)
    return 1;
  std::cout << "Test passed." << std::endl;
  return 0;
}
