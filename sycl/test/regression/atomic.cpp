// RUN: %clangxx -fsycl -fsyntax-only -Wno-deprecated-declarations -Xclang -verify %s
// expected-no-diagnostics

#include <sycl/atomic.hpp>
#include <sycl/multi_ptr.hpp>

SYCL_EXTERNAL void
store(sycl::multi_ptr<int, sycl::access::address_space::global_space> mptr,
      int value) {
  sycl::atomic<int> a(mptr);
  a.store(value);
}

SYCL_EXTERNAL int
load(sycl::multi_ptr<int, sycl::access::address_space::global_space> mptr) {
  sycl::atomic<int> a(mptr);
  return a.load();
}

SYCL_EXTERNAL int
exchange(sycl::multi_ptr<int, sycl::access::address_space::global_space> mptr,
         int value) {
  sycl::atomic<int> a(mptr);
  return a.exchange(value);
}

SYCL_EXTERNAL int
fetch_add(sycl::multi_ptr<int, sycl::access::address_space::global_space> mptr,
          int value) {
  sycl::atomic<int> a(mptr);
  return a.fetch_add(value);
}

SYCL_EXTERNAL int
fetch_sub(sycl::multi_ptr<int, sycl::access::address_space::global_space> mptr,
          int value) {
  sycl::atomic<int> a(mptr);
  return a.fetch_sub(value);
}

SYCL_EXTERNAL int
fetch_and(sycl::multi_ptr<int, sycl::access::address_space::global_space> mptr,
          int value) {
  sycl::atomic<int> a(mptr);
  return a.fetch_and(value);
}

SYCL_EXTERNAL int
fetch_or(sycl::multi_ptr<int, sycl::access::address_space::global_space> mptr,
         int value) {
  sycl::atomic<int> a(mptr);
  return a.fetch_or(value);
}

SYCL_EXTERNAL int
fetch_xor(sycl::multi_ptr<int, sycl::access::address_space::global_space> mptr,
          int value) {
  sycl::atomic<int> a(mptr);
  return a.fetch_xor(value);
}

SYCL_EXTERNAL int
fetch_min(sycl::multi_ptr<int, sycl::access::address_space::global_space> mptr,
          int value) {
  sycl::atomic<int> a(mptr);
  return a.fetch_min(value);
}

SYCL_EXTERNAL int
fetch_max(sycl::multi_ptr<int, sycl::access::address_space::global_space> mptr,
          int value) {
  sycl::atomic<int> a(mptr);
  return a.fetch_max(value);
}
