// RUN: %clang_cc1 -fsycl -fsycl-is-device -ast-dump %s | FileCheck %s
// expected-no-diagnostics

struct b {};

template <typename T, typename U>
struct is_same { static const bool value = false; };
template <typename T>
struct is_same<T, T> { static const bool value = true; };

template <typename... Ts>
struct zip_iterator {};

// CHECK_NOT: static_assert(is_same<zip_iterator<b,b>, zip_iterator<j,d>>::value, "device_iterator");
template <class j, class d>
void foo(j e, d k) {
  static_assert(is_same<zip_iterator<b, b>, zip_iterator<j, d>>::value, "device_iterator");
}

template <typename name, typename Func>
__attribute__((sycl_kernel)) void kernel_single_task(Func kernelFunc) {
  kernelFunc();
}

int main() {
  kernel_single_task<zip_iterator<b, b>>([] {});
  foo(b{}, b{});
  return 0;
}

// CHECK: TemplateArgument type 'zip_iterator<b, b>'
// CHECK: TemplateArgument type 'zip_iterator<b, b>'
