// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// This test checks the behavior of owner_less semantics for `weak_object`.

#include "weak_object_utils.hpp"

#include <sycl/ext/oneapi/owner_less.hpp>

#include <map>

template <typename SyclObjT> struct WeakObjectCheckOwnerLess {
  void operator()(SyclObjT Obj) {
    sycl::ext::oneapi::weak_object<SyclObjT> WeakObj{Obj};
    sycl::ext::oneapi::weak_object<SyclObjT> NullWeakObj;
    sycl::ext::oneapi::owner_less<SyclObjT> Comparator;

    assert((Comparator(WeakObj, NullWeakObj) &&
            !Comparator(NullWeakObj, WeakObj)) ||
           (Comparator(NullWeakObj, WeakObj) &&
            !Comparator(WeakObj, NullWeakObj)));

    assert(!Comparator(WeakObj, Obj));
    assert(!Comparator(Obj, WeakObj));
  }
};

template <typename SyclObjT> struct WeakObjectCheckOwnerLessMulti {
  void operator()(SyclObjT Obj1, SyclObjT Obj2) {
    sycl::ext::oneapi::weak_object<SyclObjT> WeakObj1{Obj1};
    sycl::ext::oneapi::weak_object<SyclObjT> WeakObj2{Obj2};
    sycl::ext::oneapi::owner_less<SyclObjT> Comparator;

    assert(
        (Comparator(WeakObj1, WeakObj2) && !Comparator(WeakObj2, WeakObj1)) ||
        (Comparator(WeakObj2, WeakObj1) && !Comparator(WeakObj1, WeakObj2)));

    assert(!Comparator(WeakObj1, Obj1));
    assert(!Comparator(Obj1, WeakObj1));

    assert(!Comparator(WeakObj2, Obj2));
    assert(!Comparator(Obj2, WeakObj2));
  }
};

template <typename SyclObjT> struct WeakObjectCheckOwnerLessMap {
  void operator()(SyclObjT Obj1, SyclObjT Obj2) {
    sycl::ext::oneapi::weak_object<SyclObjT> WeakObj1{Obj1};
    sycl::ext::oneapi::weak_object<SyclObjT> WeakObj2{Obj2};

    std::map<sycl::ext::oneapi::weak_object<SyclObjT>, int,
             sycl::ext::oneapi::owner_less<SyclObjT>>
        Map;
    Map[WeakObj1] = 1;
    Map[WeakObj2] = 2;

    assert(Map.size() == (size_t)2);
    assert(Map[WeakObj1] == 1);
    assert(Map[WeakObj2] == 2);
    assert(Map[Obj1] == 1);
    assert(Map[Obj2] == 2);

    Map[WeakObj1] = 2;
    Map[WeakObj2] = 3;

    assert(Map.size() == (size_t)2);
    assert(Map[WeakObj1] == 2);
    assert(Map[WeakObj2] == 3);
    assert(Map[Obj1] == 2);
    assert(Map[Obj2] == 3);

    Map[Obj1] = 5;
    Map[Obj2] = 6;

    assert(Map.size() == (size_t)2);
    assert(Map[WeakObj1] == 5);
    assert(Map[WeakObj2] == 6);
    assert(Map[Obj1] == 5);
    assert(Map[Obj2] == 6);

    Map[sycl::ext::oneapi::weak_object<SyclObjT>{Obj1}] = 10;
    Map[sycl::ext::oneapi::weak_object<SyclObjT>{Obj2}] = 13;

    assert(Map.size() == (size_t)2);
    assert(Map[WeakObj1] == 10);
    assert(Map[WeakObj2] == 13);
    assert(Map[Obj1] == 10);
    assert(Map[Obj2] == 13);
  }
};

int main() {
  sycl::queue Q;
  runTest<WeakObjectCheckOwnerLess>(Q);
  runTestMulti<WeakObjectCheckOwnerLessMulti>(Q);
  runTestMulti<WeakObjectCheckOwnerLessMap>(Q);
  return 0;
}
