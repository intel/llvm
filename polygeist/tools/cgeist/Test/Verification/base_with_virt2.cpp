// RUN: cgeist %s -O0 --function=* -S | FileCheck %s

class M {
};

struct _Alloc_hider : M
      {
	_Alloc_hider() { }

	char* _M_p; // The actual data.
      };


    class basic_streambuf
    {
  public:
      /// Destructor deallocates no buffer space.
      virtual
      ~basic_streambuf()
      { }
    };
    class mbasic_stringbuf : public basic_streambuf

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
// CHECK:          call @_ZN16mbasic_stringbufC1Ev
// CHECK-NEXT:     return
// CHECK-NEXT:   }
// CHECK:   func @_ZN16mbasic_stringbufC1Ev(%arg0: !llvm.ptr<struct<(struct<(ptr<ptr<func<i32 (...)>>>)>, struct<(ptr<i8>)>)>>) attributes {llvm.linkage = #llvm.linkage<linkonce_odr>} {
// CHECK:          call @_ZN15basic_streambufC1Ev
// CHECK:          call @_ZN12_Alloc_hiderC1Ev
// CHECK-NEXT:     return
// CHECK-NEXT:   }
// CHECK:   func @_ZN15basic_streambufC1Ev(%arg0: !llvm.ptr<struct<(ptr<ptr<func<i32 (...)>>>)>>) attributes {llvm.linkage = #llvm.linkage<linkonce_odr>} {
// CHECK-NEXT:     return
// CHECK-NEXT:   }
// CHECK:   func @_ZN12_Alloc_hiderC1Ev(%arg0: !llvm.ptr<struct<(ptr<i8>)>>) attributes {llvm.linkage = #llvm.linkage<linkonce_odr>} {
// CHECK-NEXT:     return
// CHECK-NEXT:   }
