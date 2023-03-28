// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out

// This test checks the behavior of `lock()` for `weak_object`.

#include "weak_object_utils.hpp"

template <typename SyclObjT> struct WeakObjectCheckLock {
  void operator()(SyclObjT Obj) {
    sycl::ext::oneapi::weak_object<SyclObjT> WeakObj{Obj};
    sycl::ext::oneapi::weak_object<SyclObjT> NullWeakObj;

    SyclObjT LObj = WeakObj.lock();
    assert(LObj == Obj);

    try {
      SyclObjT LNull = NullWeakObj.lock();
      assert(false && "Locking empty weak object did not throw.");
    } catch (sycl::exception &E) {
      assert(E.code() == sycl::make_error_code(sycl::errc::invalid) &&
             "Unexpected thrown error code.");
    }
  }
};

int main() {
  sycl::queue Q;
  runTest<WeakObjectCheckLock>(Q);
  return 0;
}
