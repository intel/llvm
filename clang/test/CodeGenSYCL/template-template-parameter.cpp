// RUN: %clang_cc1 -fsycl-is-device -fsycl-int-header=%t.h -sycl-std=2020 %s
// RUN: FileCheck -input-file=%t.h %s

#include "Inputs/sycl.hpp"

using namespace cl::sycl;

template <typename T> class Foo1;
// CHECK: template <typename T> class Foo1;
template <template <typename> class TT> class KernelName1;
// CHECK: template <template <typename> class TT> class KernelName1;
template <template <typename> class TT> void enqueue() {
  queue q;
  q.submit([&](handler &cgh) {
    cgh.single_task<KernelName1<TT>>([](){});
  });
}

template <typename TY> class Bar2;
// CHECK: template <typename TY> class Bar2;
template <template <typename> class TT> class Foo2;
// CHECK: template <template <typename> class TT> class Foo2;
template <class TTY> class KernelName2;
// CHECK: template <class TTY> class KernelName2;
template <class Y> void enqueue2() {
  queue q;
  q.submit([&](handler &cgh) {
    cgh.single_task< KernelName2<Y> >([](){});
  });
}

template <typename T> class Bar3;
// CHECK: template <typename T> class Bar3;
template <template <typename> class> class Baz3;
// CHECK: template <template <typename> class> class Baz3;
template <template <template <typename> class> class T> class Foo3;
// CHECK: template <template <template <typename> class> class T> class Foo3;
template <typename T , typename... Args> class Mist3;
// CHECK: template <typename T, typename ...Args> class Mist3;
template <typename T, template <typename, typename...> class, typename... Args> class Ice3;
// CHECK: template <typename T, template <typename, typename ...> class, typename ...Args> class Ice3;

int main() {
  enqueue<Foo1>();

  enqueue2<Foo2<Bar2>>();
  
  queue q;

  q.submit([&](handler &cgh) {
    cgh.single_task<Bar3<int>>([](){});
  });

  q.submit([&](handler &cgh) {
    cgh.single_task<Baz3<Bar3>>([](){});
  });

  q.submit([&](handler &cgh) {
    cgh.single_task<Foo3<Baz3>>([](){});
  });

  q.submit([&](handler &cgh) {
    cgh.single_task<Mist3<int, float, char, double>>([](){});
  });

  q.submit([&](handler &cgh) {
    cgh.single_task<Ice3<int, Mist3, char, short, float>>([](){});
  });

  return 0;
}

