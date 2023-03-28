// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out

// This test checks the behavior of `reset()` for `weak_object`.

#include "weak_object_utils.hpp"

template <typename SyclObjT> struct WeakObjectCheckReset {
  void operator()(SyclObjT Obj) {
    sycl::ext::oneapi::weak_object<SyclObjT> WeakObj{Obj};
    sycl::ext::oneapi::weak_object<SyclObjT> NullWeakObj;

    WeakObj.reset();
    assert(WeakObj.expired());
    assert(!WeakObj.owner_before(NullWeakObj));
    assert(!NullWeakObj.owner_before(WeakObj));

    std::optional<SyclObjT> TLObj = WeakObj.try_lock();
    assert(!TLObj.has_value());

    try {
      SyclObjT LObj = WeakObj.lock();
      assert(false && "Locking reset weak object did not throw.");
    } catch (sycl::exception &E) {
      assert(E.code() == sycl::make_error_code(sycl::errc::invalid) &&
             "Unexpected thrown error code.");
    }
  }
};

int main() {
  sycl::queue Q;
  runTest<WeakObjectCheckReset>(Q);
  return 0;
}
