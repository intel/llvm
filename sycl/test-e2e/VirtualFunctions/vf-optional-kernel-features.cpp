// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
#include <sycl/detail/core.hpp>
#include <sycl/usm.hpp>

namespace syclext = sycl::ext::oneapi::experimental;

struct set_fp64;

struct Base {
  bool fooCalled = false;
  SYCL_EXT_ONEAPI_FUNCTION_PROPERTY(syclext::indirectly_callable)
  virtual void foo() { fooCalled = true; }

  bool barCalled = false;
  SYCL_EXT_ONEAPI_FUNCTION_PROPERTY(syclext::indirectly_callable_in<set_fp64>)
  virtual void bar() {
    // this virtual function uses double
    volatile double d = 3.14;
    barCalled = true;
  }
};

class Constructor;
class Use;
class UseFP64;

int main() {
  // Selected device may not support 'fp64' aspect
  sycl::queue Q;

  Base *Obj = sycl::malloc_device<Base>(1, Q);

  Q.single_task<Constructor>([=]() {
    // Even though at LLVM IR level this kernel does reference 'Base::foo'
    // and 'Base::bar' through global variable containing `vtable` for `Base`,
    // we do not consider the kernel to be using `fp64` optional feature.
    new (Obj) Base;
  }).wait();

  Q.single_task<Use>(syclext::properties{syclext::assume_indirect_calls}, [=]() {
    // This kernel is not considered to be using any optional features, because
    // virtual functions in default set do not use any.
    Obj->foo();
  }).wait();

  if (Q.get_device().has(sycl::aspect::fp64)) {
    Q.single_task<UseFP64>(syclext::properties{syclext::assume_indirect_calls_to<set_fp64>},
        [=]() {
      // This kernel is considered to be using 'fp64' optional feature, because
      // there is a virtual function in 'set_fp64' which uses double.
      Obj->bar();
    }).wait();
  }

  int nfails = 0;
  if (!Obj->fooCalled) {
    std::cerr << "Error: 'foo' was not called\n";
    ++nfails;
  }
  if (Q.get_device().has(sycl::aspect::fp64) && !Obj->barCalled) {
    std::cerr << "Error: 'bar' was not called\n";
    ++nfails;
  }
  if (!Q.get_device().has(sycl::aspect::fp64) && Obj->barCalled) {
    std::cerr << "Error: 'bar' was called, but should not have been\n";
    ++nfails;
  }

  return 0;
}