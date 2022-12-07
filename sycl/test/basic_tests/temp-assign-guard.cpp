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

  { // Scope to limit lifetime of TempAssignGuards.

    sycl::detail::TempAssignGuard myTAG_1(myStruct.firstValue, -20);
    sycl::detail::TempAssignGuard myTAG_2(myStruct.secondValue, true);
    sycl::detail::TempAssignGuard moarTAG_1(moarPtr->firstValue, -30);
    sycl::detail::TempAssignGuard moarTAG_2(moarPtr->secondValue, true);
    sycl::detail::TempAssignGuard anotherTAG(anotherValue, -40);

    // Ensure values have been temporarily assigned.
    assert(myStruct.firstValue == -20);
    assert(myStruct.secondValue == true);
    assert(moarStruct.firstValue == -30);
    assert(moarStruct.secondValue == true);
    assert(anotherValue == -40);
  }

  // Ensure values have been restored.
  assert(myStruct.firstValue == 2);
  assert(myStruct.secondValue == false);
  assert(moarStruct.firstValue == 3);
  assert(moarStruct.secondValue == false);
  assert(anotherValue == 4);

  // Test exceptions
  int exceptionalValue = 5;
  try {
    sycl::detail::TempAssignGuard exceptionalTAG(exceptionalValue, -50);
    assert(exceptionalValue == -50);
    throw 7; // Baby needs a new pair of shoes.
  } catch (...) {
    assert(exceptionalValue == 5);
  }
  assert(exceptionalValue == 5);

  // Test premature exit
  int prematureValue = 6;
  {
    sycl::detail::TempAssignGuard prematureTAG(prematureValue, -60);
    assert(prematureValue == -60);
    goto dragons;
    assert(true == false);
  }
dragons:
  assert(prematureValue == 6);

  return 0;
}
