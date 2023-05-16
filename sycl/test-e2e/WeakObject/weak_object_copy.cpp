// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// This test checks the behavior of the copy ctor and assignment operator for
// `weak_object`.

#include "weak_object_utils.hpp"

template <typename SyclObjT> struct WeakObjectCheckCopy {
  void operator()(SyclObjT Obj) {
    sycl::ext::oneapi::weak_object<SyclObjT> WeakObj{Obj};

    sycl::ext::oneapi::weak_object<SyclObjT> WeakObjCopyCtor{WeakObj};
    sycl::ext::oneapi::weak_object<SyclObjT> WeakObjCopyAssign = WeakObj;

    assert(!WeakObjCopyCtor.expired());
    assert(!WeakObjCopyAssign.expired());

    assert(WeakObjCopyCtor.lock() == Obj);
    assert(WeakObjCopyAssign.lock() == Obj);
  }
};

int main() {
  sycl::queue Q;
  runTest<WeakObjectCheckCopy>(Q);
  return 0;
}
