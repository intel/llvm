// RUN: %clang_cc1 -fsycl-is-device -fsycl-int-header=%t.h %s -o %t.out
// RUN: FileCheck -input-file=%t.h %s

#include "Inputs/sycl.hpp"

template <typename KernelName, typename KernelType>
__attribute__((sycl_kernel)) void single_task(const KernelType &kernelFunc) {
  kernelFunc();
}

struct dummy_functor {
  void operator()() const {}
};

typedef int int_t;
using uint_t = unsigned int;

typedef const int cint_t;
using cuint_t = const unsigned int;

namespace space {
typedef long long_t;
using ulong_t = unsigned long;
typedef const long clong_t;
using culong_t = const unsigned long;
} // namespace space

// non-type template arguments cases
// CHECK: template <int N, unsigned int M> struct kernel_name1;
template <int N, unsigned int M>
struct kernel_name1 {};

// CHECK: template <int N, unsigned int M> struct kernel_name1v1;
template <int_t N, uint_t M>
struct kernel_name1v1 {};

// CHECK: template <long N, unsigned long M> struct kernel_name1v2;
template <space::long_t N, space::ulong_t M>
struct kernel_name1v2 {};

// CHECK: template <typename T, typename T2> struct kernel_name2;
template <typename T, typename T2>
struct kernel_name2;

// CHECK: class A;
class A {};
namespace space {
// CHECK: namespace space {
// CHECK-NEXT: class B;
// CHECK-NEXT: }
class B {};
using a_t = A;
using b_t = B;
} // namespace space

// partial template specialization cases
template <typename T>
struct kernel_name2<int_t, T> {};

template <typename T>
struct kernel_name2<cint_t, T> {};

template <typename T>
struct kernel_name2<space::long_t, T> {};

template <typename T>
struct kernel_name2<const space::long_t, T> {};

template <typename T>
struct kernel_name2<volatile space::clong_t, T> {};

template <typename T>
struct kernel_name2<space::a_t, T> {};

template <typename T>
struct kernel_name2<space::b_t, T> {};

// full template specialization cases
template <>
struct kernel_name2<int_t, const uint_t> {};

template <>
struct kernel_name2<space::clong_t, volatile space::culong_t> {};

template <>
struct kernel_name2<space::a_t, volatile space::b_t> {};

// CHECK: template <long T> struct kernel_name3;
template <typename space::long_t T>
struct kernel_name3;

struct foo {
  using type = long;
};

// CHECK: template <long T> struct kernel_name4;
template <typename foo::type T>
struct kernel_name4;

int main() {
  dummy_functor f;
  // non-type template arguments
  // CHECK: template <> struct KernelInfo<::kernel_name1<1, 1>> {
  single_task<kernel_name1<1, 1>>(f);
  // CHECK: template <> struct KernelInfo<::kernel_name1v1<1, 1>> {
  single_task<kernel_name1v1<1, 1>>(f);
  // CHECK: template <> struct KernelInfo<::kernel_name1v2<1, 1>> {
  single_task<kernel_name1v2<1, 1>>(f);
  // partial template specialization
  // CHECK: template <> struct KernelInfo<::kernel_name2<int, int>> {
  single_task<kernel_name2<int_t, int>>(f);
  // CHECK: template <> struct KernelInfo<::kernel_name2<const int, char>> {
  single_task<kernel_name2<cint_t, char>>(f);
  // CHECK: template <> struct KernelInfo<::kernel_name2<long, float>> {
  single_task<kernel_name2<space::long_t, float>>(f);
  // CHECK: template <> struct KernelInfo<::kernel_name2<const long, ::A>> {
  single_task<kernel_name2<const space::long_t, space::a_t>>(f);
  // CHECK: template <> struct KernelInfo<::kernel_name2<const volatile long, const ::space::B>> {
  single_task<kernel_name2<volatile space::clong_t, const space::b_t>>(f);
  // CHECK: template <> struct KernelInfo<::kernel_name2<::A, long>> {
  single_task<kernel_name2<space::a_t, space::long_t>>(f);
  // CHECK: template <> struct KernelInfo<::kernel_name2<::space::B, int>> {
  single_task<kernel_name2<space::b_t, int_t>>(f);
  // full template specialization
  // CHECK: template <> struct KernelInfo<::kernel_name2<int, const unsigned int>> {
  single_task<kernel_name2<int_t, const uint_t>>(f);
  // CHECK: template <> struct KernelInfo<::kernel_name2<const long, const volatile unsigned long>> {
  single_task<kernel_name2<space::clong_t, volatile space::culong_t>>(f);
  // CHECK: template <> struct KernelInfo<::kernel_name2<::A, volatile ::space::B>> {
  single_task<kernel_name2<space::a_t, volatile space::b_t>>(f);
  // CHECK: template <> struct KernelInfo<::kernel_name3<1>> {
  single_task<kernel_name3<1>>(f);
  // CHECK: template <> struct KernelInfo<::kernel_name4<1>> {
  single_task<kernel_name4<1>>(f);

  return 0;
}
