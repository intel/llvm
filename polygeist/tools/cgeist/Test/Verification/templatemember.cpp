// RUN: cgeist %s --function=* -S | FileCheck %s

class House;

template <typename T>
class Info;

template <>
class Info<House>{
public:
  static constexpr bool has_infinity = true;
};

bool add_kernel_cuda() {
  return Info<House>::has_infinity;
}

//  CHECK:   func @_Z15add_kernel_cudav() -> i8 attributes {llvm.linkage = #llvm.linkage<external>} {
//  CHECK-NEXT:     %c1_i8 = arith.constant 1 : i8
//  CHECK-NEXT:     return %c1_i8 : i8
//  CHECK-NEXT:   }
