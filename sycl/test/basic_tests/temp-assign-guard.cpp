// RUN: %clangxx -fsycl %s -o %t.out
// RUN: %t.out

#include <sycl/detail/util.hpp>
#include <sycl/sycl.hpp>
using namespace sycl;

struct someStruct {
  int firstValue;
  bool secondValue;
};

int main() {
  someStruct myStruct;
  myStruct.firstValue = 2;
  myStruct.secondValue = false;
  someStruct moarStruct;
  moarStruct.firstValue = 3;
  moarStruct.secondValue = false;
  someStruct *moarPtr = &moarStruct;
  int anotherValue = 4;

  { // scope to limit lifetime of TempAssignGuards

    sycl::detail::TempAssignGuard myTAG_1(myStruct.firstValue, -20);
    sycl::detail::TempAssignGuard myTAG_2(myStruct.secondValue, true);
    sycl::detail::TempAssignGuard moarTAG_1(moarPtr->firstValue, -30);
    sycl::detail::TempAssignGuard moarTAG_2(moarPtr->secondValue, true);
    sycl::detail::TempAssignGuard anotherTAG(anotherValue, -40);

    // ensure values have been temporarily assigned.
    assert(myStruct.firstValue == -20);
    assert(myStruct.secondValue == true);
    assert(moarStruct.firstValue == -30);
    assert(moarStruct.secondValue == true);
    assert(anotherValue == -40);
  }

  // ensure values have been restored.
  assert(myStruct.firstValue == 2);
  assert(myStruct.secondValue == false);
  assert(moarStruct.firstValue == 3);
  assert(moarStruct.secondValue == false);
  assert(anotherValue == 4);

  return 0;
}