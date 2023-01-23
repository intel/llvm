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

//  CHECK:   func @_Z15add_kernel_cudav() -> i1 attributes {llvm.linkage = #llvm.linkage<external>} {
//  CHECK-NEXT:     %true = arith.constant true
//  CHECK-NEXT:     return %true : i1
//  CHECK-NEXT:   }
