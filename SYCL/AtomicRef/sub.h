#pragma once

#include <CL/sycl.hpp>
#include <algorithm>
#include <cassert>
#include <numeric>
#include <vector>

using namespace sycl;
using namespace sycl::ext::oneapi;

template <template <typename, memory_order, memory_scope, access::address_space>
          class AtomicRef,
          access::address_space address_space, typename T,
          typename Difference = T>
void sub_fetch_test(queue q, size_t N) {
  T val = T(N);
  std::vector<T> output(N);
  std::fill(output.begin(), output.end(), T(0));
  {
    buffer<T> val_buf(&val, 1);
    buffer<T> output_buf(output.data(), output.size());

    q.submit([&](handler &cgh) {
      auto val = val_buf.template get_access<access::mode::read_write>(cgh);
      auto out =
          output_buf.template get_access<access::mode::discard_write>(cgh);
      cgh.parallel_for(range<1>(N), [=](item<1> it) {
        int gid = it.get_id(0);
        auto atm = AtomicRef<T, memory_order::relaxed, memory_scope::device,
                             address_space>(val[0]);
        out[gid] = atm.fetch_sub(Difference(1));
      });
    });
  }

  // All work-items decrement by 1, so final value should be equal to 0
  assert(val == T(0));

  // Fetch returns original value: will be in [1, N]
  auto min_e = std::min_element(output.begin(), output.end());
  auto max_e = std::max_element(output.begin(), output.end());
  assert(*min_e == T(1) && *max_e == T(N));

  // Intermediate values should be unique
  std::sort(output.begin(), output.end());
  assert(std::unique(output.begin(), output.end()) == output.end());
}

template <template <typename, memory_order, memory_scope, access::address_space>
          class AtomicRef,
          access::address_space address_space, typename T,
          typename Difference = T>
void sub_plus_equal_test(queue q, size_t N) {
  T val = T(N);
  std::vector<T> output(N);
  std::fill(output.begin(), output.end(), T(0));
  {
    buffer<T> val_buf(&val, 1);
    buffer<T> output_buf(output.data(), output.size());

    q.submit([&](handler &cgh) {
      auto val = val_buf.template get_access<access::mode::read_write>(cgh);
      auto out =
          output_buf.template get_access<access::mode::discard_write>(cgh);
      cgh.parallel_for(range<1>(N), [=](item<1> it) {
        int gid = it.get_id(0);
        auto atm = AtomicRef<T, memory_order::relaxed, memory_scope::device,
                             address_space>(val[0]);
        out[gid] = atm -= Difference(1);
      });
    });
  }

  // All work-items decrement by 1, so final value should be equal to 0
  assert(val == T(0));

  // -= returns updated value: will be in [0, N-1]
  auto min_e = std::min_element(output.begin(), output.end());
  auto max_e = std::max_element(output.begin(), output.end());
  assert(*min_e == T(0) && *max_e == T(N - 1));

  // Intermediate values should be unique
  std::sort(output.begin(), output.end());
  assert(std::unique(output.begin(), output.end()) == output.end());
}

template <template <typename, memory_order, memory_scope, access::address_space>
          class AtomicRef,
          access::address_space address_space, typename T,
          typename Difference = T>
void sub_pre_dec_test(queue q, size_t N) {
  T val = T(N);
  std::vector<T> output(N);
  std::fill(output.begin(), output.end(), T(0));
  {
    buffer<T> val_buf(&val, 1);
    buffer<T> output_buf(output.data(), output.size());

    q.submit([&](handler &cgh) {
      auto val = val_buf.template get_access<access::mode::read_write>(cgh);
      auto out =
          output_buf.template get_access<access::mode::discard_write>(cgh);
      cgh.parallel_for(range<1>(N), [=](item<1> it) {
        int gid = it.get_id(0);
        auto atm = AtomicRef<T, memory_order::relaxed, memory_scope::device,
                             address_space>(val[0]);
        out[gid] = --atm;
      });
    });
  }

  // All work-items decrement by 1, so final value should be equal to 0
  assert(val == T(0));

  // Pre-decrement returns updated value: will be in [0, N-1]
  auto min_e = std::min_element(output.begin(), output.end());
  auto max_e = std::max_element(output.begin(), output.end());
  assert(*min_e == T(0) && *max_e == T(N - 1));

  // Intermediate values should be unique
  std::sort(output.begin(), output.end());
  assert(std::unique(output.begin(), output.end()) == output.end());
}

template <template <typename, memory_order, memory_scope, access::address_space>
          class AtomicRef,
          access::address_space address_space, typename T,
          typename Difference = T>
void sub_post_dec_test(queue q, size_t N) {
  T val = T(N);
  std::vector<T> output(N);
  std::fill(output.begin(), output.end(), T(0));
  {
    buffer<T> val_buf(&val, 1);
    buffer<T> output_buf(output.data(), output.size());

    q.submit([&](handler &cgh) {
      auto val = val_buf.template get_access<access::mode::read_write>(cgh);
      auto out =
          output_buf.template get_access<access::mode::discard_write>(cgh);
      cgh.parallel_for(range<1>(N), [=](item<1> it) {
        int gid = it.get_id(0);
        auto atm = AtomicRef<T, memory_order::relaxed, memory_scope::device,
                             address_space>(val[0]);
        out[gid] = atm--;
      });
    });
  }

  // All work-items decrement by 1, so final value should be equal to 0
  assert(val == T(0));

  // Post-decrement returns original value: will be in [1, N]
  auto min_e = std::min_element(output.begin(), output.end());
  auto max_e = std::max_element(output.begin(), output.end());
  assert(*min_e == T(1) && *max_e == T(N));

  // Intermediate values should be unique
  std::sort(output.begin(), output.end());
  assert(std::unique(output.begin(), output.end()) == output.end());
}

template <typename T, typename Difference = T>
void sub_test(queue q, size_t N) {
  sub_fetch_test<::sycl::ext::oneapi::atomic_ref,
                 access::address_space::global_space, T, Difference>(q, N);
  sub_fetch_test<::sycl::atomic_ref, access::address_space::global_space, T,
                 Difference>(q, N);
  sub_plus_equal_test<::sycl::ext::oneapi::atomic_ref,
                      access::address_space::global_space, T, Difference>(q, N);
  sub_plus_equal_test<::sycl::atomic_ref, access::address_space::global_space,
                      T, Difference>(q, N);
  sub_pre_dec_test<::sycl::ext::oneapi::atomic_ref,
                   access::address_space::global_space, T, Difference>(q, N);
  sub_pre_dec_test<::sycl::atomic_ref, access::address_space::global_space, T,
                   Difference>(q, N);
  sub_post_dec_test<::sycl::ext::oneapi::atomic_ref,
                    access::address_space::global_space, T, Difference>(q, N);
  sub_post_dec_test<::sycl::atomic_ref, access::address_space::global_space, T,
                    Difference>(q, N);
}

template <typename T, typename Difference = T>
void sub_generic_test(queue q, size_t N) {
  sub_fetch_test<::sycl::atomic_ref, access::address_space::generic_space, T,
                 Difference>(q, N);
  sub_plus_equal_test<::sycl::atomic_ref, access::address_space::generic_space,
                      T, Difference>(q, N);
  sub_pre_dec_test<::sycl::atomic_ref, access::address_space::generic_space, T,
                   Difference>(q, N);
  sub_post_dec_test<::sycl::atomic_ref, access::address_space::generic_space, T,
                    Difference>(q, N);
}
