// RUN: cgeist %s -O0 --function=* -S | FileCheck %s

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

// clang-format off
// CHECK:   func @_Z1av() attributes {llvm.linkage = #llvm.linkage<external>} {
// CHECK-NEXT:     %c1_i64 = arith.constant 1 : i64
// CHECK-NEXT:     %0 = llvm.alloca %c1_i64 x !llvm.struct<(struct<packed (ptr<ptr<func<i32 (...)>>>, i32, array<4 x i8>)>, struct<(ptr<i8>)>)> : (i64) -> !llvm.ptr<struct<(struct<packed (ptr<ptr<func<i32 (...)>>>, i32, array<4 x i8>)>, struct<(ptr<i8>)>)>>
// CHECK-NEXT:     call @_ZN16mbasic_stringbufC1Ev(%0) : (!llvm.ptr<struct<(struct<packed (ptr<ptr<func<i32 (...)>>>, i32, array<4 x i8>)>, struct<(ptr<i8>)>)>>) -> ()
// CHECK-NEXT:     return
// CHECK-NEXT:   }
// CHECK:   func @_ZN16mbasic_stringbufC1Ev(%arg0: !llvm.ptr<struct<(struct<packed (ptr<ptr<func<i32 (...)>>>, i32, array<4 x i8>)>, struct<(ptr<i8>)>)>>) attributes {llvm.linkage = #llvm.linkage<linkonce_odr>} {
// CHECK-NEXT:     %0 = llvm.getelementptr %arg0[0, 0] : (!llvm.ptr<struct<(struct<packed (ptr<ptr<func<i32 (...)>>>, i32, array<4 x i8>)>, struct<(ptr<i8>)>)>>) -> !llvm.ptr<struct<packed (ptr<ptr<func<i32 (...)>>>, i32, array<4 x i8>)>>
// CHECK-NEXT:     call @_ZN1AC1Ev(%0) : (!llvm.ptr<struct<packed (ptr<ptr<func<i32 (...)>>>, i32, array<4 x i8>)>>) -> ()
// CHECK:          call @_ZN12_Alloc_hiderC1Ev
// CHECK-NEXT:     return
// CHECK-NEXT:   }
// CHECK:   func @_ZN1AC1Ev(%arg0: !llvm.ptr<struct<packed (ptr<ptr<func<i32 (...)>>>, i32, array<4 x i8>)>>) attributes {llvm.linkage = #llvm.linkage<linkonce_odr>} {
// CHECK-DAG:     %c3_i32 = arith.constant 3 : i32
// CHECK-NEXT:     %0 = llvm.getelementptr %arg0[0, 1] : (!llvm.ptr<struct<packed (ptr<ptr<func<i32 (...)>>>, i32, array<4 x i8>)>>) -> !llvm.ptr<i32>
// CHECK-NEXT:     llvm.store %c3_i32, %0 : !llvm.ptr<i32>
// CHECK-NEXT:     return
// CHECK-NEXT:   }
// CHECK-NEXT:   func @_ZN12_Alloc_hiderC1Ev(%arg0: !llvm.ptr<struct<(ptr<i8>)>>) attributes {llvm.linkage = #llvm.linkage<linkonce_odr>} {
// CHECK-NEXT:     return
// CHECK-NEXT:   }
