// RUN: cgeist %s --function=* -S | FileCheck %s
  
template<typename _Tp, _Tp __v>
    struct integral_constant
    {
      static constexpr _Tp value = __v;
    };

  template<typename _Tp, _Tp __v>
    constexpr _Tp integral_constant<_Tp, __v>::value;
   
bool failure() {
  return integral_constant<bool, true>::value;
}

unsigned char conv() {
  return integral_constant<bool, true>::value;
}


// CHECK:   func @_Z7failurev() -> i1
// CHECK-NEXT:     %true = arith.constant true
// CHECK-NEXT:     return %true : i1
// CHECK-NEXT:   }
// CHECK:   func @_Z4convv() -> i8 
// CHECK-NEXT:     %c1_i8 = arith.constant 1 : i8
// CHECK-NEXT:     return %c1_i8 : i8
// CHECK-NEXT:   }
