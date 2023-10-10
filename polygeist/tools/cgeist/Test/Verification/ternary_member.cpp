// RUN: cgeist  %s --function=* -S | FileCheck %s

// COM: Test ternary operator on a class member of type bool. Because bool
// members are stored as i8, but scf.if requires an i1, a type conversion must
// take place.

void keep(int *);

template <typename T> class dual_pointer {
public:
  dual_pointer(bool cond, T *first, T *second) : c{cond}, f{first}, s{second} {}

  T *get_pointer() { return c ? f : s; }

private:
  bool c;

  T *f;

  T *s;
};

template <typename T> void callee(dual_pointer<T> ptr) {
  keep(ptr.get_pointer());
}

int main() {

  dual_pointer<int> ptr{true, nullptr, nullptr};

  callee(ptr);

  return 0;
}


// CHECK-LABEL:   func.func @_ZN12dual_pointerIiE11get_pointerEv(
// CHECK-SAME:                                                   %[[VAL_0:.*]]: !llvm.ptr) -> memref<?xi32> 
// CHECK-NEXT:      %[[VAL_1:.*]] = arith.constant 0 : i8
// CHECK-NEXT:      %[[VAL_2:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i8, memref<?xi32>, memref<?xi32>)>
// CHECK-NEXT:      %[[VAL_3:.*]] = llvm.load %[[VAL_2]] : !llvm.ptr -> i8
// CHECK-NEXT:      %[[VAL_4:.*]] = arith.cmpi ne, %[[VAL_3]], %[[VAL_1]] : i8
// CHECK-NEXT:      %[[VAL_5:.*]] = scf.if %[[VAL_4]] -> (!llvm.ptr) {
// CHECK-NEXT:        %[[VAL_6:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i8, memref<?xi32>, memref<?xi32>)>
// CHECK-NEXT:        scf.yield %[[VAL_6]] : !llvm.ptr
// CHECK-NEXT:      } else {
// CHECK-NEXT:        %[[VAL_7:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i8, memref<?xi32>, memref<?xi32>)>
// CHECK-NEXT:        scf.yield %[[VAL_7]] : !llvm.ptr
// CHECK-NEXT:      }
// CHECK-NEXT:      %[[VAL_8:.*]] = llvm.load %[[VAL_5]] : !llvm.ptr -> memref<?xi32>
// CHECK-NEXT:      return %[[VAL_8]] : memref<?xi32>
// CHECK-NEXT:    }


