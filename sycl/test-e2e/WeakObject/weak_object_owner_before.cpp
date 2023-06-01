// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// This test checks the behavior of owner_before semantics for `weak_object`.

#include "weak_object_utils.hpp"

template <typename SyclObjT> struct WeakObjectCheckOwnerBefore {
  void operator()(SyclObjT Obj) {
    sycl::ext::oneapi::weak_object<SyclObjT> WeakObj{Obj};
    sycl::ext::oneapi::weak_object<SyclObjT> NullWeakObj;

    assert((WeakObj.owner_before(NullWeakObj) &&
            !NullWeakObj.owner_before(WeakObj)) ||
           (NullWeakObj.owner_before(WeakObj) &&
            !WeakObj.owner_before(NullWeakObj)));

    assert(!WeakObj.owner_before(Obj));
    assert(!Obj.ext_oneapi_owner_before(WeakObj));

    assert(!Obj.ext_oneapi_owner_before(Obj));
  }
};

template <typename SyclObjT> struct WeakObjectCheckOwnerBeforeMulti {
  void operator()(SyclObjT Obj1, SyclObjT Obj2) {
    sycl::ext::oneapi::weak_object<SyclObjT> WeakObj1{Obj1};
    sycl::ext::oneapi::weak_object<SyclObjT> WeakObj2{Obj2};

    assert(
        (WeakObj1.owner_before(WeakObj2) && !WeakObj2.owner_before(WeakObj1)) ||
        (WeakObj2.owner_before(WeakObj1) && !WeakObj1.owner_before(WeakObj2)));

    assert(!WeakObj1.owner_before(Obj1));
    assert(!Obj1.ext_oneapi_owner_before(WeakObj1));

    assert(!WeakObj2.owner_before(Obj2));
    assert(!Obj2.ext_oneapi_owner_before(WeakObj2));

    assert((Obj1.ext_oneapi_owner_before(Obj2) &&
            !Obj2.ext_oneapi_owner_before(Obj1)) ||
           (Obj2.ext_oneapi_owner_before(Obj1) &&
            !Obj1.ext_oneapi_owner_before(Obj2)));
  }
};

int main() {
  sycl::queue Q;
  runTest<WeakObjectCheckOwnerBefore>(Q);
  runTestMulti<WeakObjectCheckOwnerBeforeMulti>(Q);
  return 0;
}
