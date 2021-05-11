// RUN: %clang_cc1 -fsycl-is-device -fsyntax-only -Wno-sycl-2017-compat -verify %s
// expected-no-diagnostics

// The kernel_single_task call is emitted as an OpenCL kernel function. The call
// to getFullyQualifiedType caused a 2nd instantiation of zip_iterator<b,b> (the
// first instantiation is on line 28 during phase 1).
// Then, the call to 'foo' in 'main' causes 'foo' to be instantiated. The best
// match for zip_iterator<j,d> is the one instantiated by the
// kernel_single_task, which is now different from the one made in phase 1.
// 'is_same<zip_iterator<b,b>' is instantiated during phase 1 at line 28, and
// 'zip_iterator<b,b>' is instantiated by getFullyQualifiedName.
// So 'is_same<zip_iterator<b,b>, zip_iterator<j,d>>::value' return false
// even though zip_iterator<b,b> and zip_iterator<j,d> have the same type
// 'zip_iterator<b,b>'.

struct b {};

template <typename T, typename U>
struct is_same { static const bool value = false; };
template <typename T>
struct is_same<T, T> { static const bool value = true; };

template <typename... Ts>
struct zip_iterator {};

template <class j, class d>
void foo(j e, d k) {
  static_assert(is_same<zip_iterator<b, b>, zip_iterator<j, d>>::value, "device_iterator");
}

template <typename name, typename Func>
__attribute__((sycl_kernel)) void kernel_single_task(const Func &kernelFunc) {
  kernelFunc();
}

int main() {
  kernel_single_task<zip_iterator<b, b>>([] {});
  foo(b{}, b{});
  return 0;
}
