// RUN: %clang -I %S/Inputs --sycl -Xclang -fsycl-int-header=%t.h %s -c -o kernel.spv
// RUN: FileCheck -input-file=%t.h %s

// CHECK: // Forward declarations of templated kernel function types:
// CHECK-NEXT: template <typename T, typename T2, long N, unsigned long M> struct functor1;
// CHECK-NEXT: template <typename T, typename T2, long N, unsigned long M> struct functor2;
// CHECK-NEXT: template <typename T, typename T2> struct functor3;
//
// CHECK: // Specializations of KernelInfo for kernel function types:
// CHECK: template <> struct KernelInfo<struct functor1<long, unsigned long, 0, 1>> {
// CHECK: template <> struct KernelInfo<struct functor2<long, unsigned long, 0, 1>> {
// CHECK: template <> struct KernelInfo<struct functor3<int, int>> {
// CHECK: template <> struct KernelInfo<struct functor3<long, int>> {
// CHECK: template <> struct KernelInfo<struct functor3<int, unsigned long>> {
// CHECK: template <> struct KernelInfo<struct functor3<long, float>> {
// CHECK: template <> struct KernelInfo<struct functor3<float, unsigned long>> {
// CHECK: template <> struct KernelInfo<struct functor3<long, unsigned long>> {

#include "sycl.hpp"

template <typename KernelName, typename KernelType>
__attribute__((sycl_kernel)) void kernel_single_task(KernelType kernelFunc) {
  kernelFunc();
}

typedef signed long int signed_integer_t;

using unsigned_integer_t = unsigned long int;

template <typename T, typename T2, signed long int N, unsigned long int M>
struct functor1 { void operator()() {} };

template <typename T, typename T2, signed_integer_t N, unsigned_integer_t M>
struct functor2 { void operator()() {} };

template <typename T, typename T2>
struct functor3 { void operator()() {} };

template <typename T>
struct functor3<signed_integer_t, T> { void operator()() {} };

template <typename T>
struct functor3<T, unsigned_integer_t> { void operator()() {} };

template <>
struct functor3<signed_integer_t, float> { void operator()() {} };

template <>
struct functor3<float, unsigned_integer_t> { void operator()() {} };

template <>
struct functor3<signed_integer_t, unsigned_integer_t> { void operator()() {} };

int main() {
  functor1<signed long int, unsigned long int, 0L, 1UL> Functor1;
  kernel_single_task<decltype(Functor1)>(Functor1);

  functor2<signed_integer_t, unsigned_integer_t, 0L, 1UL> Functor2;
  kernel_single_task<decltype(Functor2)>(Functor2);

  functor3<int, int> Functor3;
  kernel_single_task<decltype(Functor3)>(Functor3);

  functor3<signed_integer_t, int> Functor4;
  kernel_single_task<decltype(Functor4)>(Functor4);

  functor3<int, unsigned_integer_t> Functor5;
  kernel_single_task<decltype(Functor5)>(Functor5);

  functor3<signed_integer_t, float> Functor6;
  kernel_single_task<decltype(Functor6)>(Functor6);

  functor3<float, unsigned_integer_t> Functor7;
  kernel_single_task<decltype(Functor7)>(Functor7);

  functor3<signed_integer_t, unsigned_integer_t> Functor8;
  kernel_single_task<decltype(Functor8)>(Functor8);

  return 0;
}
