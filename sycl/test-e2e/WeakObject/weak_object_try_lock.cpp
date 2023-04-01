// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %BE_RUN_PLACEHOLDER %t.out

// This test checks the behavior of `try_lock()` for `weak_object`.

#include "weak_object_utils.hpp"

template <typename SyclObjT> struct WeakObjectCheckTryLock {
  void operator()(SyclObjT Obj) {
    sycl::ext::oneapi::weak_object<SyclObjT> WeakObj{Obj};
    sycl::ext::oneapi::weak_object<SyclObjT> NullWeakObj;

    std::optional<SyclObjT> TLObj = WeakObj.try_lock();
    std::optional<SyclObjT> TLNull = NullWeakObj.try_lock();

    assert(TLObj.has_value());
    assert(!TLNull.has_value());

    assert(TLObj.value() == Obj);
  }
};

int main() {
  sycl::queue Q;
  runTest<WeakObjectCheckTryLock>(Q);
  return 0;
}
