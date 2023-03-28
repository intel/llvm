// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out

// This test checks the behavior of the copy ctor and assignment operator for
// `weak_object`.

#include "weak_object_utils.hpp"

template <typename SyclObjT> struct WeakObjectCheckMove {
  void operator()(SyclObjT Obj) {
    sycl::ext::oneapi::weak_object<SyclObjT> WeakObj1{Obj};
    sycl::ext::oneapi::weak_object<SyclObjT> WeakObj2{Obj};

    sycl::ext::oneapi::weak_object<SyclObjT> WeakObjMoveCtor{
        std::move(WeakObj1)};
    sycl::ext::oneapi::weak_object<SyclObjT> WeakObjMoveAssign =
        std::move(WeakObj2);

    assert(!WeakObjMoveCtor.expired());
    assert(!WeakObjMoveAssign.expired());

    assert(WeakObjMoveCtor.lock() == Obj);
    assert(WeakObjMoveAssign.lock() == Obj);
  }
};

int main() {
  sycl::queue Q;
  runTest<WeakObjectCheckMove>(Q);
  return 0;
}
