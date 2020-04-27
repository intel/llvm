// RUN: %clangxx -fsycl -fsycl-unnamed-lambda -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: env SYCL_DEVICE_TYPE=HOST %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out

#include <CL/sycl.hpp>
#include <algorithm>
#include <cassert>
#include <iostream>
#include <numeric>
#include <vector>
using namespace sycl;
using namespace sycl::intel;

template <typename T>
void add_fetch_test(queue q, size_t N) {
  T sum = 0;
  std::vector<T> output(N);
  std::fill(output.begin(), output.end(), 0);
  {
    buffer<T> sum_buf(&sum, 1);
    buffer<T> output_buf(output.data(), output.size());

    q.submit([&](handler &cgh) {
      auto sum = sum_buf.template get_access<access::mode::read_write>(cgh);
      auto out = output_buf.template get_access<access::mode::discard_write>(cgh);
      cgh.parallel_for(range<1>(N), [=](item<1> it) {
        int gid = it.get_id(0);
        auto atm = atomic_ref<T, intel::memory_order::relaxed, intel::memory_scope::device, access::address_space::global_space>(sum[0]);
        out[gid] = atm.fetch_add(T(1));
      });
    });
  }

  // All work-items increment by 1, so final value should be equal to N
  assert(sum == N);

  // Fetch returns original value: will be in [0, N-1]
  auto min_e = std::min_element(output.begin(), output.end());
  auto max_e = std::max_element(output.begin(), output.end());
  assert(*min_e == 0 && *max_e == N - 1);

  // Intermediate values should be unique
  std::sort(output.begin(), output.end());
  assert(std::unique(output.begin(), output.end()) == output.end());
}

template <typename T>
void add_plus_equal_test(queue q, size_t N) {
  T sum = 0;
  std::vector<T> output(N);
  std::fill(output.begin(), output.end(), 0);
  {
    buffer<T> sum_buf(&sum, 1);
    buffer<T> output_buf(output.data(), output.size());

    q.submit([&](handler &cgh) {
      auto sum = sum_buf.template get_access<access::mode::read_write>(cgh);
      auto out = output_buf.template get_access<access::mode::discard_write>(cgh);
      cgh.parallel_for(range<1>(N), [=](item<1> it) {
        int gid = it.get_id(0);
        auto atm = atomic_ref<T, intel::memory_order::relaxed, intel::memory_scope::device, access::address_space::global_space>(sum[0]);
        out[gid] = atm += T(1);
      });
    });
  }

  // All work-items increment by 1, so final value should be equal to N
  assert(sum == N);

  // += returns updated value: will be in [1, N]
  auto min_e = std::min_element(output.begin(), output.end());
  auto max_e = std::max_element(output.begin(), output.end());
  assert(*min_e == 1 && *max_e == N);

  // Intermediate values should be unique
  std::sort(output.begin(), output.end());
  assert(std::unique(output.begin(), output.end()) == output.end());
}

template <typename T>
void add_pre_inc_test(queue q, size_t N) {
  T sum = 0;
  std::vector<T> output(N);
  std::fill(output.begin(), output.end(), 0);
  {
    buffer<T> sum_buf(&sum, 1);
    buffer<T> output_buf(output.data(), output.size());

    q.submit([&](handler &cgh) {
      auto sum = sum_buf.template get_access<access::mode::read_write>(cgh);
      auto out = output_buf.template get_access<access::mode::discard_write>(cgh);
      cgh.parallel_for(range<1>(N), [=](item<1> it) {
        int gid = it.get_id(0);
        auto atm = atomic_ref<T, intel::memory_order::relaxed, intel::memory_scope::device, access::address_space::global_space>(sum[0]);
        out[gid] = ++atm;
      });
    });
  }

  // All work-items increment by 1, so final value should be equal to N
  assert(sum == N);

  // Pre-increment returns updated value: will be in [1, N]
  auto min_e = std::min_element(output.begin(), output.end());
  auto max_e = std::max_element(output.begin(), output.end());
  assert(*min_e == 1 && *max_e == N);

  // Intermediate values should be unique
  std::sort(output.begin(), output.end());
  assert(std::unique(output.begin(), output.end()) == output.end());
}

template <typename T>
void add_post_inc_test(queue q, size_t N) {
  T sum = 0;
  std::vector<T> output(N);
  std::fill(output.begin(), output.end(), 0);
  {
    buffer<T> sum_buf(&sum, 1);
    buffer<T> output_buf(output.data(), output.size());

    q.submit([&](handler &cgh) {
      auto sum = sum_buf.template get_access<access::mode::read_write>(cgh);
      auto out = output_buf.template get_access<access::mode::discard_write>(cgh);
      cgh.parallel_for(range<1>(N), [=](item<1> it) {
        int gid = it.get_id(0);
        auto atm = atomic_ref<T, intel::memory_order::relaxed, intel::memory_scope::device, access::address_space::global_space>(sum[0]);
        out[gid] = atm++;
      });
    });
  }

  // All work-items increment by 1, so final value should be equal to N
  assert(sum == N);

  // Post-increment returns original value: will be in [0, N-1]
  auto min_e = std::min_element(output.begin(), output.end());
  auto max_e = std::max_element(output.begin(), output.end());
  assert(*min_e == 0 && *max_e == N - 1);

  // Intermediate values should be unique
  std::sort(output.begin(), output.end());
  assert(std::unique(output.begin(), output.end()) == output.end());
}

template <typename T>
void add_test(queue q, size_t N) {
  add_fetch_test<T>(q, N);
  add_plus_equal_test<T>(q, N);
  add_pre_inc_test<T>(q, N);
  add_post_inc_test<T>(q, N);
}

// Floating-point types do not support pre- or post-increment
template <>
void add_test<float>(queue q, size_t N) {
  add_fetch_test<float>(q, N);
  add_plus_equal_test<float>(q, N);
}
template <>
void add_test<double>(queue q, size_t N) {
  add_fetch_test<double>(q, N);
  add_plus_equal_test<double>(q, N);
}

int main() {
  queue q;
  std::string version = q.get_device().get_info<info::device::version>();
  if (version < std::string("2.0")) {
    std::cout << "Skipping test\n";
    return 0;
  }

  constexpr int N = 32;

  // TODO: Enable missing tests when supported
  add_test<int>(q, N);
  add_test<unsigned int>(q, N);
  add_test<long>(q, N);
  add_test<unsigned long>(q, N);
  add_test<long long>(q, N);
  add_test<unsigned long long>(q, N);
  add_test<float>(q, N);
  add_test<double>(q, N);
  //add_test<char*>(q, N);

  std::cout << "Test passed." << std::endl;
}
