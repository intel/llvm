// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// This test checks the behavior of `expired()` for `weak_object`.

#include "weak_object_utils.hpp"

template <typename SyclObjT> struct WeakObjectCheckExpired {
  void operator()(SyclObjT Obj) {
    sycl::ext::oneapi::weak_object<SyclObjT> WeakObj{Obj};
    sycl::ext::oneapi::weak_object<SyclObjT> NullWeakObj;

    assert(!WeakObj.expired());
    assert(NullWeakObj.expired());
  }
};

int main() {
  sycl::queue Q;
  runTest<WeakObjectCheckExpired>(Q);
  return 0;
}
