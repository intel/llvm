// RUN: cgeist  %s --function=* --struct-abi=0 -memref-abi=0 -S | FileCheck %s

void run0(void*);
void run1(void*);
void run2(void*);

class M {
    public:
 M() { run0(this); }
};


struct _Alloc_hider : M
      {
	_Alloc_hider() { run1(this); }

      };
  
    class basic_ostringstream 
    {
    public:
      _Alloc_hider _M_stringbuf;
      basic_ostringstream() { run2(this); }
    };

void a() {
    ::basic_ostringstream a;
}

// CHECK-LABEL:   func.func @_Z1av() attributes {llvm.linkage = #llvm.linkage<external>} {
// CHECK-NEXT:      %[[VAL_0:.*]] = arith.constant 1 : i64
// CHECK-NEXT:      %[[VAL_1:.*]] = llvm.alloca %[[VAL_0]] x !llvm.struct<(struct<(i8)>)> : (i64) -> !llvm.ptr
// CHECK-NEXT:      call @_ZN19basic_ostringstreamC1Ev(%[[VAL_1]]) : (!llvm.ptr) -> ()
// CHECK-NEXT:      return
// CHECK-NEXT:    }

// CHECK-LABEL:   func.func @_ZN19basic_ostringstreamC1Ev(
// CHECK-SAME:                                            %[[VAL_0:.*]]: !llvm.ptr) attributes {llvm.linkage = #llvm.linkage<linkonce_odr>} {
// CHECK-NEXT:      %[[VAL_1:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(struct<(i8)>)>
// CHECK-NEXT:      call @_ZN12_Alloc_hiderC1Ev(%[[VAL_1]]) : (!llvm.ptr) -> ()
// CHECK-NEXT:      call @_Z4run2Pv(%[[VAL_0]]) : (!llvm.ptr) -> ()
// CHECK-NEXT:      return
// CHECK-NEXT:    }

// CHECK-LABEL:   func.func @_ZN12_Alloc_hiderC1Ev(
// CHECK-SAME:                                     %[[VAL_0:.*]]: !llvm.ptr) attributes {llvm.linkage = #llvm.linkage<linkonce_odr>} {
// CHECK-NEXT:      call @_ZN1MC1Ev(%[[VAL_0]]) : (!llvm.ptr) -> ()
// CHECK-NEXT:      call @_Z4run1Pv(%[[VAL_0]]) : (!llvm.ptr) -> ()
// CHECK-NEXT:      return
// CHECK-NEXT:    }
// CHECK-NEXT:    func.func private @_Z4run2Pv(!llvm.ptr) attributes {llvm.linkage = #llvm.linkage<external>}

// CHECK-LABEL:   func.func @_ZN1MC1Ev(
// CHECK-SAME:                         %[[VAL_0:.*]]: !llvm.ptr) attributes {llvm.linkage = #llvm.linkage<linkonce_odr>} {
// CHECK-NEXT:      call @_Z4run0Pv(%[[VAL_0]]) : (!llvm.ptr) -> ()
// CHECK-NEXT:      return
// CHECK-NEXT:    }
// CHECK-NEXT:    func.func private @_Z4run1Pv(!llvm.ptr) attributes {llvm.linkage = #llvm.linkage<external>}
// CHECK-NEXT:    func.func private @_Z4run0Pv(!llvm.ptr) attributes {llvm.linkage = #llvm.linkage<external>}
