// RUN: cgeist  %s -O2 --function=* -S | FileCheck %s

class M {
};


struct _Alloc_hider : M
      {
	_Alloc_hider() { }

	char* _M_p; // The actual data.
      };

class A {
    public:
  int x;
  virtual void foo();
  A() : x(3) {}
};

    class mbasic_stringbuf : public A 
    {
    public:

      _Alloc_hider	_M_dataplus;
      mbasic_stringbuf()
      { }

    };


void a() {
    mbasic_stringbuf a;
}

// CHECK-LABEL:   func.func @_Z1av() attributes {llvm.linkage = #llvm.linkage<external>} {
// CHECK-NEXT:      %[[VAL_0:.*]] = arith.constant 1 : i64
// CHECK-NEXT:      %[[VAL_1:.*]] = llvm.alloca %[[VAL_0]] x !llvm.struct<(struct<packed (ptr, i32, array<4 x i8>)>, struct<(ptr)>)> : (i64) -> !llvm.ptr
// CHECK-NEXT:      call @_ZN16mbasic_stringbufC1Ev(%[[VAL_1]]) : (!llvm.ptr) -> ()
// CHECK-NEXT:      return
// CHECK-NEXT:    }

// CHECK-LABEL:   func.func @_ZN16mbasic_stringbufC1Ev(
// CHECK-SAME:                                         %[[VAL_0:.*]]: !llvm.ptr) attributes {llvm.linkage = #llvm.linkage<linkonce_odr>} {
// CHECK-NEXT:      %[[VAL_1:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(struct<packed (ptr, i32, array<4 x i8>)>, struct<(ptr)>)>
// CHECK-NEXT:      call @_ZN1AC1Ev(%[[VAL_1]]) : (!llvm.ptr) -> ()
// CHECK-NEXT:      %[[VAL_2:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(struct<packed (ptr, i32, array<4 x i8>)>, struct<(ptr)>)>
// CHECK-NEXT:      call @_ZN12_Alloc_hiderC1Ev(%[[VAL_2]]) : (!llvm.ptr) -> ()
// CHECK-NEXT:      return
// CHECK-NEXT:    }

// CHECK-LABEL:   func.func @_ZN1AC1Ev(
// CHECK-SAME:                         %[[VAL_0:.*]]: !llvm.ptr) attributes {llvm.linkage = #llvm.linkage<linkonce_odr>} {
// CHECK-NEXT:      %[[VAL_1:.*]] = arith.constant 3 : i32
// CHECK-NEXT:      %[[VAL_2:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<packed (ptr, i32, array<4 x i8>)>
// CHECK-NEXT:      llvm.store %[[VAL_1]], %[[VAL_2]] : i32, !llvm.ptr
// CHECK-NEXT:      return
// CHECK-NEXT:    }

// CHECK-LABEL:   func.func @_ZN12_Alloc_hiderC1Ev(
// CHECK-SAME:                                     %[[VAL_0:.*]]: !llvm.ptr) attributes {llvm.linkage = #llvm.linkage<linkonce_odr>} {
// CHECK-NEXT:      return
// CHECK-NEXT:    }
