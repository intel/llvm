// RUN: %clang_cc1 -triple spir64-unknown-unknown -fsycl-is-device \
// RUN: -emit-llvm-bc %s -o %t-host.bc -opt-record-file %t-host.yaml
// RUN: FileCheck -check-prefix=SPIR --input-file %t-host.yaml %s

// RUN: %clang_cc1 -triple nvptx64-unknown-unknown -fsycl-is-device \
// RUN: -emit-llvm-bc %s -o %t-host.bc -opt-record-file %t-host.yaml
// RUN: FileCheck -check-prefix=NVPTX --input-file %t-host.yaml %s
// The test generates remarks about the kernel argument, their location and type
// in the resulting yaml file.

#include "Inputs/sycl.hpp"

sycl::handler H;

class DecomposedBase {
public:
  float DecompVar;
  int *DecompPtr;
  sycl::accessor<char, 1, sycl::access::mode::read> decompAcc;
  sycl::stream DecompStream{0, 0, H};
};

struct NotDecomposedBase {
public:
  int B;
};

struct StructWithPointer {
public:
  int *Ptr;
};

struct KernelFunctor : NotDecomposedBase, DecomposedBase, StructWithPointer {
  int A;
  int *Ptr;
  int Array[3];
  sycl::sampler Sampl;
  StructWithPointer Obj;
  void operator()() const {
  }
};

struct AccessorDerived : sycl::accessor<char, 1, sycl::access::mode::read> {
  int B;
};

int main() {
  sycl::queue q;
  q.submit([&](sycl::handler &cgh) {
    KernelFunctor f{};
    cgh.single_task(f);
  });

  AccessorDerived DerivedObject;
  q.submit([&](sycl::handler &cgh) {
    sycl::kernel_handler kh;

    cgh.single_task<class XYZ>(
        [=](auto) {
          DerivedObject.use();
        },
        kh);
  });

  return 0;
}

// SPIR: --- !Passed
// SPIR: Pass:{{.*}}sycl
// SPIR: Name:{{.*}}Region
// SPIR: DebugLoc:{{.*}} { File: '{{.*}}kernel-arg-opt-report.cpp',
// SPIR-NEXT: Line: 33, Column: 8 }
// SPIR-NEXT: Function:        _ZTS13KernelFunctor
// SPIR-NEXT: Args:
// SPIR-NEXT: String:          'Arg '
// SPIR-NEXT: Argument:        '0'
// SPIR-NEXT: String:          ':'
// SPIR-NEXT: String:          Compiler generated argument for base class,
// SPIR-NEXT: String:          NotDecomposedBase
// SPIR-NEXT: String:          '  ('
// SPIR-NEXT: String:          ''
// SPIR-NEXT: String:          'Type:'
// SPIR-NEXT: String:          NotDecomposedBase
// SPIR-NEXT: String:          ', '
// SPIR-NEXT: String:          'Size: '
// SPIR-NEXT: Argument:        '4'
// SPIR-NEXT: String:          ')'

// SPIR: --- !Passed
// SPIR: Pass:{{.*}}sycl
// SPIR: Name:{{.*}}Region
// SPIR: DebugLoc:{{.*}} { File: '{{.*}}kernel-arg-opt-report.cpp',
// SPIR-NEXT: Line: 33, Column: 8 }
// SPIR-NEXT: Function:        _ZTS13KernelFunctor
// SPIR-NEXT: Args:
// SPIR-NEXT: String:          'Arg '
// SPIR-NEXT: Argument:        '1'
// SPIR-NEXT: String:          ':'
// SPIR-NEXT: String:          'Compiler generated argument for decomposed struct/class,'
// SPIR-NEXT: String:          DecomposedBase
// SPIR-NEXT: String:          '  ('
// SPIR-NEXT: String:          'Field:DecompVar, '
// SPIR-NEXT: String:          'Type:'
// SPIR-NEXT: String:          float
// SPIR-NEXT: String:          ', '
// SPIR-NEXT: String:          'Size: '
// SPIR-NEXT: Argument:        '4'
// SPIR-NEXT: String:          ')'

// SPIR: --- !Passed
// SPIR: Pass:{{.*}}sycl
// SPIR: Name:{{.*}}Region
// SPIR: DebugLoc:{{.*}} { File: '{{.*}}kernel-arg-opt-report.cpp',
// SPIR-NEXT: Line: 33, Column: 8 }
// SPIR-NEXT: Function:        _ZTS13KernelFunctor
// SPIR-NEXT: Args:
// SPIR-NEXT: String:          'Arg '
// SPIR-NEXT: Argument:        '2'
// SPIR-NEXT: String:          ':'
// SPIR-NEXT: String:          Compiler generated argument for nested pointer,
// SPIR-NEXT: String:          DecompPtr
// SPIR-NEXT: String:          '  ('
// SPIR-NEXT: String:          ''
// SPIR-NEXT: String:          'Type:'
// SPIR-NEXT: String:          Compiler generated
// SPIR-NEXT: String:          ', '
// SPIR-NEXT: String:          'Size: '
// SPIR-NEXT: Argument:        '8'
// SPIR-NEXT: String:          ')'

// SPIR: --- !Passed
// SPIR: Pass:{{.*}}sycl
// SPIR: Name:{{.*}}Region
// SPIR: DebugLoc:{{.*}} { File: '{{.*}}kernel-arg-opt-report.cpp',
// SPIR-NEXT: Line: 33, Column: 8 }
// SPIR-NEXT: Function:        _ZTS13KernelFunctor
// SPIR-NEXT: Args:
// SPIR-NEXT: String:          'Arg '
// SPIR-NEXT: Argument:        '3'
// SPIR-NEXT: String:          ':'
// SPIR-NEXT: String:          'Compiler generated argument for sycl::accessor<char, 1, sycl::access::mode::read>,'
// SPIR-NEXT: String:          decompAcc
// SPIR-NEXT: String:          '  ('
// SPIR-NEXT: String:          ''
// SPIR-NEXT: String:          'Type:'
// SPIR-NEXT: String:          '__global char *'
// SPIR-NEXT: String:          ', '
// SPIR-NEXT: String:          'Size: '
// SPIR-NEXT: Argument:        '8'
// SPIR-NEXT: String:          ')'

// SPIR: --- !Passed
// SPIR: Pass:{{.*}}sycl
// SPIR: Name:{{.*}}Region
// SPIR: DebugLoc:{{.*}} { File: '{{.*}}kernel-arg-opt-report.cpp',
// SPIR-NEXT: Line: 33, Column: 8 }
// SPIR-NEXT: Function:        _ZTS13KernelFunctor
// SPIR-NEXT: Args:
// SPIR-NEXT: String:          'Arg '
// SPIR-NEXT: Argument:        '4'
// SPIR-NEXT: String:          ':'
// SPIR-NEXT: String:          'Compiler generated argument for sycl::accessor<char, 1, sycl::access::mode::read>,'
// SPIR-NEXT: String:          decompAcc
// SPIR-NEXT: String:          '  ('
// SPIR-NEXT: String:          ''
// SPIR-NEXT: String:          'Type:'
// SPIR-NEXT: String:          'struct sycl::range<1>'
// SPIR-NEXT: String:          ', '
// SPIR-NEXT: String:          'Size: '
// SPIR-NEXT: Argument:        '1'
// SPIR-NEXT: String:          ')'

// SPIR: --- !Passed
// SPIR: Pass:{{.*}}sycl
// SPIR: Name:{{.*}}Region
// SPIR: DebugLoc:{{.*}} { File: '{{.*}}kernel-arg-opt-report.cpp',
// SPIR-NEXT: Line: 33, Column: 8 }
// SPIR-NEXT: Function:        _ZTS13KernelFunctor
// SPIR-NEXT: Args:
// SPIR-NEXT: String:          'Arg '
// SPIR-NEXT: Argument:        '5'
// SPIR-NEXT: String:          ':'
// SPIR-NEXT: String:          'Compiler generated argument for sycl::accessor<char, 1, sycl::access::mode::read>,'
// SPIR-NEXT: String:          decompAcc
// SPIR-NEXT: String:          '  ('
// SPIR-NEXT: String:          ''
// SPIR-NEXT: String:          'Type:'
// SPIR-NEXT: String:          'struct sycl::range<1>'
// SPIR-NEXT: String:          ', '
// SPIR-NEXT: String:          'Size: '
// SPIR-NEXT: Argument:        '1'
// SPIR-NEXT: String:          ')'

// SPIR: --- !Passed
// SPIR: Pass:{{.*}}sycl
// SPIR: Name:{{.*}}Region
// SPIR: DebugLoc:{{.*}} { File: '{{.*}}kernel-arg-opt-report.cpp',
// SPIR-NEXT: Line: 33, Column: 8 }
// SPIR-NEXT: Function:        _ZTS13KernelFunctor
// SPIR-NEXT: Args:
// SPIR-NEXT: String:          'Arg '
// SPIR-NEXT: Argument:        '6'
// SPIR-NEXT: String:          ':'
// SPIR-NEXT: String:          'Compiler generated argument for sycl::accessor<char, 1, sycl::access::mode::read>,'
// SPIR-NEXT: String:          decompAcc
// SPIR-NEXT: String:          '  ('
// SPIR-NEXT: String:          ''
// SPIR-NEXT: String:          'Type:'
// SPIR-NEXT: String:          'struct sycl::id<1>'
// SPIR-NEXT: String:          ', '
// SPIR-NEXT: String:          'Size: '
// SPIR-NEXT: Argument:        '1'
// SPIR-NEXT: String:          ')'

// SPIR: --- !Passed
// SPIR: Pass:{{.*}}sycl
// SPIR: Name:{{.*}}Region
// SPIR: DebugLoc:{{.*}} { File: '{{.*}}kernel-arg-opt-report.cpp',
// SPIR-NEXT: Line: 33, Column: 8 }
// SPIR-NEXT: Function:        _ZTS13KernelFunctor
// SPIR-NEXT: Args:
// SPIR-NEXT: String:          'Arg '
// SPIR-NEXT: Argument:        '7'
// SPIR-NEXT: String:          ':'
// SPIR-NEXT: String:          'Compiler generated argument for sycl::stream,'
// SPIR-NEXT: String:          DecompStream
// SPIR-NEXT: String:          '  ('
// SPIR-NEXT: String:          ''
// SPIR-NEXT: String:          'Type:'
// SPIR-NEXT: String:          '__global char *'
// SPIR-NEXT: String:          ', '
// SPIR-NEXT: String:          'Size: '
// SPIR-NEXT: Argument:        '8'
// SPIR-NEXT: String:          ')'

// SPIR: --- !Passed
// SPIR: Pass:{{.*}}sycl
// SPIR: Name:{{.*}}Region
// SPIR: DebugLoc:{{.*}} { File: '{{.*}}kernel-arg-opt-report.cpp',
// SPIR-NEXT: Line: 33, Column: 8 }
// SPIR-NEXT: Function:        _ZTS13KernelFunctor
// SPIR-NEXT: Args:
// SPIR-NEXT: String:          'Arg '
// SPIR-NEXT: Argument:        '8'
// SPIR-NEXT: String:          ':'
// SPIR-NEXT: String:          'Compiler generated argument for sycl::stream,'
// SPIR-NEXT: String:          DecompStream
// SPIR-NEXT: String:          '  ('
// SPIR-NEXT: String:          ''
// SPIR-NEXT: String:          'Type:'
// SPIR-NEXT: String:          'struct sycl::range<1>'
// SPIR-NEXT: String:          ', '
// SPIR-NEXT: String:          'Size: '
// SPIR-NEXT: Argument:        '1'
// SPIR-NEXT: String:          ')'

// SPIR: --- !Passed
// SPIR: Pass:{{.*}}sycl
// SPIR: Name:{{.*}}Region
// SPIR: DebugLoc:{{.*}} { File: '{{.*}}kernel-arg-opt-report.cpp',
// SPIR-NEXT: Line: 33, Column: 8 }
// SPIR-NEXT: Function:        _ZTS13KernelFunctor
// SPIR-NEXT: Args:
// SPIR-NEXT: String:          'Arg '
// SPIR-NEXT: Argument:        '9'
// SPIR-NEXT: String:          ':'
// SPIR-NEXT: String:          'Compiler generated argument for sycl::stream,'
// SPIR-NEXT: String:          DecompStream
// SPIR-NEXT: String:          '  ('
// SPIR-NEXT: String:          ''
// SPIR-NEXT: String:          'Type:'
// SPIR-NEXT: String:          'struct sycl::range<1>'
// SPIR-NEXT: String:          ', '
// SPIR-NEXT: String:          'Size: '
// SPIR-NEXT: Argument:        '1'
// SPIR-NEXT: String:          ')'

// SPIR: --- !Passed
// SPIR: Pass:{{.*}}sycl
// SPIR: Name:{{.*}}Region
// SPIR: DebugLoc:{{.*}} { File: '{{.*}}kernel-arg-opt-report.cpp',
// SPIR-NEXT: Line: 33, Column: 8 }
// SPIR-NEXT: Function:        _ZTS13KernelFunctor
// SPIR-NEXT: Args:
// SPIR-NEXT: String:          'Arg '
// SPIR-NEXT: Argument:        '10'
// SPIR-NEXT: String:          ':'
// SPIR-NEXT: String:          'Compiler generated argument for sycl::stream,'
// SPIR-NEXT: String:          DecompStream
// SPIR-NEXT: String:          '  ('
// SPIR-NEXT: String:          ''
// SPIR-NEXT: String:          'Type:'
// SPIR-NEXT: String:          'struct sycl::id<1>'
// SPIR-NEXT: String:          ', '
// SPIR-NEXT: String:          'Size: '
// SPIR-NEXT: Argument:        '1'
// SPIR-NEXT: String:          ')'

// SPIR: --- !Passed
// SPIR: Pass:{{.*}}sycl
// SPIR: Name:{{.*}}Region
// SPIR: DebugLoc:{{.*}} { File: '{{.*}}kernel-arg-opt-report.cpp',
// SPIR-NEXT: Line: 33, Column: 8 }
// SPIR-NEXT: Function:        _ZTS13KernelFunctor
// SPIR-NEXT: Args:
// SPIR-NEXT: String:          'Arg '
// SPIR-NEXT: Argument:        '11'
// SPIR-NEXT: String:          ':'
// SPIR-NEXT: String:          'Compiler generated argument for sycl::stream,'
// SPIR-NEXT: String:          DecompStream
// SPIR-NEXT: String:          '  ('
// SPIR-NEXT: String:          ''
// SPIR-NEXT: String:          'Type:'
// SPIR-NEXT: String:          int
// SPIR-NEXT: String:          ', '
// SPIR-NEXT: String:          'Size: '
// SPIR-NEXT: Argument:        '4'
// SPIR-NEXT: String:          ')'

// SPIR: --- !Passed
// SPIR: Pass:{{.*}}sycl
// SPIR: Name:{{.*}}Region
// SPIR: DebugLoc:{{.*}} { File: '{{.*}}kernel-arg-opt-report.cpp',
// SPIR-NEXT: Line: 33, Column: 8 }
// SPIR-NEXT: Function:        _ZTS13KernelFunctor
// SPIR-NEXT: Args:
// SPIR-NEXT: String:          'Arg '
// SPIR-NEXT: Argument:        '12'
// SPIR-NEXT: String:          ':'
// SPIR-NEXT: String:          Compiler generated argument for base class with pointer,
// SPIR-NEXT: String:          StructWithPointer
// SPIR-NEXT: String:          '  ('
// SPIR-NEXT: String:          ''
// SPIR-NEXT: String:          'Type:'
// SPIR-NEXT: String:          Compiler generated
// SPIR-NEXT: String:          ', '
// SPIR-NEXT: String:          'Size: '
// SPIR-NEXT: Argument:        '8'
// SPIR-NEXT: String:          ')'

// SPIR: --- !Passed
// SPIR: Pass:{{.*}}sycl
// SPIR: Name:{{.*}}Region
// SPIR: DebugLoc:{{.*}} { File: '{{.*}}kernel-arg-opt-report.cpp',
// SPIR-NEXT: Line: 33, Column: 8 }
// SPIR-NEXT: Function:        _ZTS13KernelFunctor
// SPIR-NEXT: Args:
// SPIR-NEXT: String:          'Arg '
// SPIR-NEXT: Argument:        '13'
// SPIR-NEXT: String:          ':'
// SPIR-NEXT: String:          'Compiler generated argument for decomposed struct/class,'
// SPIR-NEXT: String:          KernelFunctor
// SPIR-NEXT: String:          '  ('
// SPIR-NEXT: String:          'Field:A, '
// SPIR-NEXT: String:          'Type:'
// SPIR-NEXT: String:          int
// SPIR-NEXT: String:          ', '
// SPIR-NEXT: String:          'Size: '
// SPIR-NEXT: Argument:        '4'
// SPIR-NEXT: String:          ')'

// SPIR: --- !Passed
// SPIR: Pass:{{.*}}sycl
// SPIR: Name:{{.*}}Region
// SPIR: DebugLoc:{{.*}} { File: '{{.*}}kernel-arg-opt-report.cpp',
// SPIR-NEXT: Line: 33, Column: 8 }
// SPIR-NEXT: Function:        _ZTS13KernelFunctor
// SPIR-NEXT: Args:
// SPIR-NEXT: String:          'Arg '
// SPIR-NEXT: Argument:        '14'
// SPIR-NEXT: String:          ':'
// SPIR-NEXT: String:          ''
// SPIR-NEXT: String:          Ptr
// SPIR-NEXT: String:          '  ('
// SPIR-NEXT: String:          ''
// SPIR-NEXT: String:          'Type:'
// SPIR-NEXT: String:          '__global int *'
// SPIR-NEXT: String:          ', '
// SPIR-NEXT: String:          'Size: '
// SPIR-NEXT: Argument:        '8'
// SPIR-NEXT: String:          ')'

// SPIR: --- !Passed
// SPIR: Pass:{{.*}}sycl
// SPIR: Name:{{.*}}Region
// SPIR: DebugLoc:{{.*}} { File: '{{.*}}kernel-arg-opt-report.cpp',
// SPIR-NEXT: Line: 33, Column: 8 }
// SPIR-NEXT: Function:        _ZTS13KernelFunctor
// SPIR-NEXT: Args:
// SPIR-NEXT: String:          'Arg '
// SPIR-NEXT: Argument:        '15'
// SPIR-NEXT: String:          ':'
// SPIR-NEXT: String:          Compiler generated argument for array,
// SPIR-NEXT: String:          Array
// SPIR-NEXT: String:          '  ('
// SPIR-NEXT: String:          ''
// SPIR-NEXT: String:          'Type:'
// SPIR-NEXT: String:          Compiler generated
// SPIR-NEXT: String:          ', '
// SPIR-NEXT: String:          'Size: '
// SPIR-NEXT: Argument:        '12'
// SPIR-NEXT: String:          ')'

// SPIR: --- !Passed
// SPIR: Pass:{{.*}}sycl
// SPIR: Name:{{.*}}Region
// SPIR: DebugLoc:{{.*}} { File: '{{.*}}kernel-arg-opt-report.cpp',
// SPIR-NEXT: Line: 33, Column: 8 }
// SPIR-NEXT: Function:        _ZTS13KernelFunctor
// SPIR-NEXT: Args:
// SPIR-NEXT: String:          'Arg '
// SPIR-NEXT: Argument:        '16'
// SPIR-NEXT: String:          ':'
// SPIR-NEXT: String:          'Compiler generated argument for sycl::sampler,'
// SPIR-NEXT: String:          Sampl
// SPIR-NEXT: String:          '  ('
// SPIR-NEXT: String:          ''
// SPIR-NEXT: String:          'Type:'
// SPIR-NEXT: String:          sampler_t
// SPIR-NEXT: String:          ', '
// SPIR-NEXT: String:          'Size: '
// SPIR-NEXT: Argument:        '8'
// SPIR-NEXT: String:          ')'

// SPIR: --- !Passed
// SPIR: Pass:{{.*}}sycl
// SPIR: Name:{{.*}}Region
// SPIR: DebugLoc:{{.*}} { File: '{{.*}}kernel-arg-opt-report.cpp',
// SPIR-NEXT: Line: 33, Column: 8 }
// SPIR-NEXT: Function:        _ZTS13KernelFunctor
// SPIR-NEXT: Args:
// SPIR-NEXT: String:          'Arg '
// SPIR-NEXT: Argument:        '17'
// SPIR-NEXT: String:          ':'
// SPIR-NEXT: String:          Compiler generated argument for object with pointer,
// SPIR-NEXT: String:          Obj
// SPIR-NEXT: String:          '  ('
// SPIR-NEXT: String:          ''
// SPIR-NEXT: String:          'Type:'
// SPIR-NEXT: String:          Compiler generated 
// SPIR-NEXT: String:          ', '
// SPIR-NEXT: String:          'Size: '
// SPIR-NEXT: Argument:        '8'
// SPIR-NEXT: String:          ')'
// Output for kernel XYZ

// SPIR: --- !Passed
// SPIR: Pass:{{.*}}sycl
// SPIR: Name:{{.*}}Region
// SPIR: DebugLoc:{{.*}} { File: '{{.*}}kernel-arg-opt-report.cpp',
// SPIR-NEXT: Line: 59, Column: 9 }
// SPIR-NEXT: Function:        _ZTSZZ4mainENKUlRN4sycl3_V17handlerEE0_clES2_E3XYZ
// SPIR-NEXT: Args:
// SPIR-NEXT: String:          'Arg '
// SPIR-NEXT: Argument:        '0'
// SPIR-NEXT: String:          ':'
// SPIR-NEXT: String:          'Compiler generated argument for base class sycl::accessor<char, 1, sycl::access::mode::read>,'
// SPIR-NEXT: String:          'sycl::accessor<char, 1, sycl::access::mode::read>'
// SPIR-NEXT: String:          '  ('
// SPIR-NEXT: String:          ''
// SPIR-NEXT: String:          'Type:'
// SPIR-NEXT: String:          '__global char *'
// SPIR-NEXT: String:          ', '
// SPIR-NEXT: String:          'Size: '
// SPIR-NEXT: Argument:        '8'
// SPIR-NEXT: String:          ')'

// SPIR: --- !Passed
// SPIR: Pass:{{.*}}sycl
// SPIR: Name:{{.*}}Region
// SPIR: DebugLoc:{{.*}} { File: '{{.*}}kernel-arg-opt-report.cpp',
// SPIR-NEXT: Line: 59, Column: 9 }
// SPIR-NEXT: Function:        _ZTSZZ4mainENKUlRN4sycl3_V17handlerEE0_clES2_E3XYZ
// SPIR-NEXT: Args:
// SPIR-NEXT: String:          'Arg '
// SPIR-NEXT: Argument:        '1'
// SPIR-NEXT: String:          ':'
// SPIR-NEXT: String:          'Compiler generated argument for base class sycl::accessor<char, 1, sycl::access::mode::read>,'
// SPIR-NEXT: String:          'sycl::accessor<char, 1, sycl::access::mode::read>'
// SPIR-NEXT: String:          '  ('
// SPIR-NEXT: String:          ''
// SPIR-NEXT: String:          'Type:'
// SPIR-NEXT: String:          'struct sycl::range<1>'
// SPIR-NEXT: String:          ', '
// SPIR-NEXT: String:          'Size: '
// SPIR-NEXT: Argument:        '1'
// SPIR-NEXT: String:          ')'

// SPIR: --- !Passed
// SPIR: Pass:{{.*}}sycl
// SPIR: Name:{{.*}}Region
// SPIR: DebugLoc:{{.*}} { File: '{{.*}}kernel-arg-opt-report.cpp',
// SPIR-NEXT: Line: 59, Column: 9 }
// SPIR-NEXT: Function:        _ZTSZZ4mainENKUlRN4sycl3_V17handlerEE0_clES2_E3XYZ
// SPIR-NEXT: Args:
// SPIR-NEXT: String:          'Arg '
// SPIR-NEXT: Argument:        '2'
// SPIR-NEXT: String:          ':'
// SPIR-NEXT: String:          'Compiler generated argument for base class sycl::accessor<char, 1, sycl::access::mode::read>,'
// SPIR-NEXT: String:          'sycl::accessor<char, 1, sycl::access::mode::read>'
// SPIR-NEXT: String:          '  ('
// SPIR-NEXT: String:          ''
// SPIR-NEXT: String:          'Type:'
// SPIR-NEXT: String:          'struct sycl::range<1>'
// SPIR-NEXT: String:          ', '
// SPIR-NEXT: String:          'Size: '
// SPIR-NEXT: Argument:        '1'
// SPIR-NEXT: String:          ')'

// SPIR: --- !Passed
// SPIR: Pass:{{.*}}sycl
// SPIR: Name:{{.*}}Region
// SPIR: DebugLoc:{{.*}} { File: '{{.*}}kernel-arg-opt-report.cpp',
// SPIR-NEXT: Line: 59, Column: 9 }
// SPIR-NEXT: Function:        _ZTSZZ4mainENKUlRN4sycl3_V17handlerEE0_clES2_E3XYZ
// SPIR-NEXT: Args:
// SPIR-NEXT: String:          'Arg '
// SPIR-NEXT: Argument:        '3'
// SPIR-NEXT: String:          ':'
// SPIR-NEXT: String:          'Compiler generated argument for base class sycl::accessor<char, 1, sycl::access::mode::read>,'
// SPIR-NEXT: String:          'sycl::accessor<char, 1, sycl::access::mode::read>'
// SPIR-NEXT: String:          '  ('
// SPIR-NEXT: String:          ''
// SPIR-NEXT: String:          'Type:'
// SPIR-NEXT: String:          'struct sycl::id<1>'
// SPIR-NEXT: String:          ', '
// SPIR-NEXT: String:          'Size: '
// SPIR-NEXT: Argument:        '1'
// SPIR-NEXT: String:          ')'

// SPIR: --- !Passed
// SPIR: Pass:{{.*}}sycl
// SPIR: Name:{{.*}}Region
// SPIR: DebugLoc:{{.*}} { File: '{{.*}}kernel-arg-opt-report.cpp',
// SPIR-NEXT: Line: 59, Column: 9 }
// SPIR-NEXT: Function:        _ZTSZZ4mainENKUlRN4sycl3_V17handlerEE0_clES2_E3XYZ
// SPIR-NEXT: Args:
// SPIR-NEXT: String:          'Arg '
// SPIR-NEXT: Argument:        '4'
// SPIR-NEXT: String:          ':'
// SPIR-NEXT: String:          'Compiler generated argument for decomposed struct/class,'
// SPIR-NEXT: String:          AccessorDerived
// SPIR-NEXT: String:          '  ('
// SPIR-NEXT: String:          'Field:B, '
// SPIR-NEXT: String:          'Type:'
// SPIR-NEXT: String:          int
// SPIR-NEXT: String:          ', '
// SPIR-NEXT: String:          'Size: '
// SPIR-NEXT: Argument:        '4'
// SPIR-NEXT: String:          ')'

// NVPTX: --- !Passed
// NVPTX: Pass:{{.*}}sycl
// NVPTX: Name:{{.*}}Region
// NVPTX: DebugLoc:{{.*}} { File: '{{.*}}kernel-arg-opt-report.cpp',
// NVPTX: Line: 59, Column: 9 }
// NVPTX-NEXT: Function:        _ZTSZZ4mainENKUlRN4sycl3_V17handlerEE0_clES2_E3XYZ
// NVPTX-NEXT: Args:
// NVPTX-NEXT: String:          'Arg '
// NVPTX: Argument:        '5'
// NVPTX-NEXT: String:          ':'
// NVPTX-NEXT: String:          Compiler generated argument for SYCL2020 specialization constant
// NVPTX-NEXT: String:          ''
// NVPTX-NEXT: String:          '  ('
// NVPTX-NEXT: String:          ''
// NVPTX-NEXT: String:          'Type:'
// NVPTX-NEXT: String:          '__global char *'
// NVPTX-NEXT: String:          ', '
// NVPTX-NEXT: String:          'Size: '
// NVPTX-NEXT: Argument:        '8'
// NVPTX-NEXT: String:          ')'
