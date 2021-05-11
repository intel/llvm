// RUN: %clang_cc1 -triple spir64-unknown-unknown-sycldevice -fsycl-is-device \
// RUN: -Wno-sycl-2017-compat -emit-llvm-bc %s -o %t-host.bc -opt-record-file %t-host.yaml
// RUN: FileCheck -check-prefix=CHECK --input-file %t-host.yaml %s
// The test generates remarks about the kernel argument, their location and type
// in the resulting yaml file.

#include "Inputs/sycl.hpp"

sycl::handler H;

class decomposedbase {
public:
  float decompvar;
  int *decompptr;
  sycl::accessor<char, 1, sycl::access::mode::read> decompAcc;
  sycl::stream decompStream{0, 0, H};
};

struct notdecomposedbase {
public:
  int b;
};

struct kernelfunctor : notdecomposedbase, decomposedbase {
  int a;
  int *ptr;
  int array[3];
  sycl::sampler sampl;
  void operator()() const {
  }
};

int main() {
  sycl::queue q;

  q.submit([&](sycl::handler &cgh) {
    kernelfunctor f{};
    cgh.single_task(f);
  });

  return 0;
}

// CHECK: --- !Passed
// CHECK: Pass:{{.*}}sycl
// CHECK: Name:{{.*}}Region
// CHECK: DebugLoc:{{.*}} { File: '{{.*}}kernel-arg-opt-report.cpp',
// CHECK-NEXT: Line: 24, Column: 8 }
// CHECK-NEXT: Function:        _ZTS13kernelfunctor
// CHECK-NEXT: Args:
// CHECK-NEXT: String:          'Arg '
// CHECK-NEXT: Argument:        '0'
// CHECK-NEXT: String:          ':'
// CHECK-NEXT: String:          Compiler generated argument for base class,
// CHECK-NEXT: String:          struct notdecomposedbase
// CHECK-NEXT: String:          '  ('
// CHECK-NEXT: String:          ''
// CHECK-NEXT: String:          'Type:'
// CHECK-NEXT: String:          struct notdecomposedbase
// CHECK-NEXT: String:          ', '
// CHECK-NEXT: String:          'Size: '
// CHECK-NEXT: Argument:        '4'
// CHECK-NEXT: String:          ')'

// CHECK: --- !Passed
// CHECK: Pass:{{.*}}sycl
// CHECK: Name:{{.*}}Region
// CHECK: DebugLoc:{{.*}} { File: '{{.*}}kernel-arg-opt-report.cpp',
// CHECK-NEXT: Line: 24, Column: 8 }
// CHECK-NEXT: Function:        _ZTS13kernelfunctor
// CHECK-NEXT: Args:
// CHECK-NEXT: String:          'Arg '
// CHECK-NEXT: Argument:        '1'
// CHECK-NEXT: String:          ':'
// CHECK-NEXT: String:          'Compiler generated argument for decomposed struct/class,'
// CHECK-NEXT: String:          decomposedbase
// CHECK-NEXT: String:          '  ('
// CHECK-NEXT: String:          'Field:decompvar, '
// CHECK-NEXT: String:          'Type:'
// CHECK-NEXT: String:          float
// CHECK-NEXT: String:          ', '
// CHECK-NEXT: String:          'Size: '
// CHECK-NEXT: Argument:        '4'
// CHECK-NEXT: String:          ')'

// CHECK: --- !Passed
// CHECK: Pass:{{.*}}sycl
// CHECK: Name:{{.*}}Region
// CHECK: DebugLoc:{{.*}} { File: '{{.*}}kernel-arg-opt-report.cpp',
// CHECK-NEXT: Line: 24, Column: 8 }
// CHECK-NEXT: Function:        _ZTS13kernelfunctor
// CHECK-NEXT: Args:
// CHECK-NEXT: String:          'Arg '
// CHECK-NEXT: Argument:        '2'
// CHECK-NEXT: String:          ':'
// CHECK-NEXT: String:          Compiler generated argument for nested pointer,
// CHECK-NEXT: String:          decompptr
// CHECK-NEXT: String:          '  ('
// CHECK-NEXT: String:          ''
// CHECK-NEXT: String:          'Type:'
// CHECK-NEXT: String:          Compiler generated
// CHECK-NEXT: String:          ', '
// CHECK-NEXT: String:          'Size: '
// CHECK-NEXT: Argument:        '8'
// CHECK-NEXT: String:          ')'

// CHECK: --- !Passed
// CHECK: Pass:{{.*}}sycl
// CHECK: Name:{{.*}}Region
// CHECK: DebugLoc:{{.*}} { File: '{{.*}}kernel-arg-opt-report.cpp',
// CHECK-NEXT: Line: 24, Column: 8 }
// CHECK-NEXT: Function:        _ZTS13kernelfunctor
// CHECK-NEXT: Args:
// CHECK-NEXT: String:          'Arg '
// CHECK-NEXT: Argument:        '3'
// CHECK-NEXT: String:          ':'
// CHECK-NEXT: String:          Compiler generated argument for accessor,
// CHECK-NEXT: String:          decompAcc
// CHECK-NEXT: String:          '  ('
// CHECK-NEXT: String:          ''
// CHECK-NEXT: String:          'Type:'
// CHECK-NEXT: String:          '__global char *'
// CHECK-NEXT: String:          ', '
// CHECK-NEXT: String:          'Size: '
// CHECK-NEXT: Argument:        '8'
// CHECK-NEXT: String:          ')'

// CHECK: --- !Passed
// CHECK: Pass:{{.*}}sycl
// CHECK: Name:{{.*}}Region
// CHECK: DebugLoc:{{.*}} { File: '{{.*}}kernel-arg-opt-report.cpp',
// CHECK-NEXT: Line: 24, Column: 8 }
// CHECK-NEXT: Function:        _ZTS13kernelfunctor
// CHECK-NEXT: Args:
// CHECK-NEXT: String:          'Arg '
// CHECK-NEXT: Argument:        '4'
// CHECK-NEXT: String:          ':'
// CHECK-NEXT: String:          Compiler generated argument for accessor,
// CHECK-NEXT: String:          decompAcc
// CHECK-NEXT: String:          '  ('
// CHECK-NEXT: String:          ''
// CHECK-NEXT: String:          'Type:'
// CHECK-NEXT: String:          'struct sycl::range<1>'
// CHECK-NEXT: String:          ', '
// CHECK-NEXT: String:          'Size: '
// CHECK-NEXT: Argument:        '1'
// CHECK-NEXT: String:          ')'

// CHECK: --- !Passed
// CHECK: Pass:{{.*}}sycl
// CHECK: Name:{{.*}}Region
// CHECK: DebugLoc:{{.*}} { File: '{{.*}}kernel-arg-opt-report.cpp',
// CHECK-NEXT: Line: 24, Column: 8 }
// CHECK-NEXT: Function:        _ZTS13kernelfunctor
// CHECK-NEXT: Args:
// CHECK-NEXT: String:          'Arg '
// CHECK-NEXT: Argument:        '5'
// CHECK-NEXT: String:          ':'
// CHECK-NEXT: String:          Compiler generated argument for accessor,
// CHECK-NEXT: String:          decompAcc
// CHECK-NEXT: String:          '  ('
// CHECK-NEXT: String:          ''
// CHECK-NEXT: String:          'Type:'
// CHECK-NEXT: String:          'struct sycl::range<1>'
// CHECK-NEXT: String:          ', '
// CHECK-NEXT: String:          'Size: '
// CHECK-NEXT: Argument:        '1'
// CHECK-NEXT: String:          ')'

// CHECK: --- !Passed
// CHECK: Pass:{{.*}}sycl
// CHECK: Name:{{.*}}Region
// CHECK: DebugLoc:{{.*}} { File: '{{.*}}kernel-arg-opt-report.cpp',
// CHECK-NEXT: Line: 24, Column: 8 }
// CHECK-NEXT: Function:        _ZTS13kernelfunctor
// CHECK-NEXT: Args:
// CHECK-NEXT: String:          'Arg '
// CHECK-NEXT: Argument:        '6'
// CHECK-NEXT: String:          ':'
// CHECK-NEXT: String:          Compiler generated argument for accessor,
// CHECK-NEXT: String:          decompAcc
// CHECK-NEXT: String:          '  ('
// CHECK-NEXT: String:          ''
// CHECK-NEXT: String:          'Type:'
// CHECK-NEXT: String:          'struct sycl::id<1>'
// CHECK-NEXT: String:          ', '
// CHECK-NEXT: String:          'Size: '
// CHECK-NEXT: Argument:        '1'
// CHECK-NEXT: String:          ')'

// CHECK: --- !Passed
// CHECK: Pass:{{.*}}sycl
// CHECK: Name:{{.*}}Region
// CHECK: DebugLoc:{{.*}} { File: '{{.*}}kernel-arg-opt-report.cpp',
// CHECK-NEXT: Line: 24, Column: 8 }
// CHECK-NEXT: Function:        _ZTS13kernelfunctor
// CHECK-NEXT: Args:
// CHECK-NEXT: String:          'Arg '
// CHECK-NEXT: Argument:        '7'
// CHECK-NEXT: String:          ':'
// CHECK-NEXT: String:          Compiler generated argument for stream,
// CHECK-NEXT: String:          decompStream
// CHECK-NEXT: String:          '  ('
// CHECK-NEXT: String:          ''
// CHECK-NEXT: String:          'Type:'
// CHECK-NEXT: String:          'sycl::stream'
// CHECK-NEXT: String:          ', '
// CHECK-NEXT: String:          'Size: '
// CHECK-NEXT: Argument:        '3'
// CHECK-NEXT: String:          ')'

// CHECK: --- !Passed
// CHECK: Pass:{{.*}}sycl
// CHECK: Name:{{.*}}Region
// CHECK: DebugLoc:{{.*}} { File: '{{.*}}kernel-arg-opt-report.cpp',
// CHECK-NEXT: Line: 24, Column: 8 }
// CHECK-NEXT: Function:        _ZTS13kernelfunctor
// CHECK-NEXT: Args:
// CHECK-NEXT: String:          'Arg '
// CHECK-NEXT: Argument:        '8'
// CHECK-NEXT: String:          ':'
// CHECK-NEXT: String:          Compiler generated argument for accessor,
// CHECK-NEXT: String:          acc
// CHECK-NEXT: String:          '  ('
// CHECK-NEXT: String:          ''
// CHECK-NEXT: String:          'Type:'
// CHECK-NEXT: String:          '__global int *'
// CHECK-NEXT: String:          ', '
// CHECK-NEXT: String:          'Size: '
// CHECK-NEXT: Argument:        '8'
// CHECK-NEXT: String:          ')'

// CHECK: --- !Passed
// CHECK: Pass:{{.*}}sycl
// CHECK: Name:{{.*}}Region
// CHECK: DebugLoc:{{.*}} { File: '{{.*}}kernel-arg-opt-report.cpp',
// CHECK-NEXT: Line: 24, Column: 8 }
// CHECK-NEXT: Function:        _ZTS13kernelfunctor
// CHECK-NEXT: Args:
// CHECK-NEXT: String:          'Arg '
// CHECK-NEXT: Argument:        '9'
// CHECK-NEXT: String:          ':'
// CHECK-NEXT: String:          Compiler generated argument for accessor,
// CHECK-NEXT: String:          acc
// CHECK-NEXT: String:          '  ('
// CHECK-NEXT: String:          ''
// CHECK-NEXT: String:          'Type:'
// CHECK-NEXT: String:          'struct sycl::range<1>'
// CHECK-NEXT: String:          ', '
// CHECK-NEXT: String:          'Size: '
// CHECK-NEXT: Argument:        '1'
// CHECK-NEXT: String:          ')'

// CHECK: --- !Passed
// CHECK: Pass:{{.*}}sycl
// CHECK: Name:{{.*}}Region
// CHECK: DebugLoc:{{.*}} { File: '{{.*}}kernel-arg-opt-report.cpp',
// CHECK-NEXT: Line: 24, Column: 8 }
// CHECK-NEXT: Function:        _ZTS13kernelfunctor
// CHECK-NEXT: Args:
// CHECK-NEXT: String:          'Arg '
// CHECK-NEXT: Argument:        '10'
// CHECK-NEXT: String:          ':'
// CHECK-NEXT: String:          Compiler generated argument for accessor,
// CHECK-NEXT: String:          acc
// CHECK-NEXT: String:          '  ('
// CHECK-NEXT: String:          ''
// CHECK-NEXT: String:          'Type:'
// CHECK-NEXT: String:          'struct sycl::range<1>'
// CHECK-NEXT: String:          ', '
// CHECK-NEXT: String:          'Size: '
// CHECK-NEXT: Argument:        '1'
// CHECK-NEXT: String:          ')'

// CHECK: --- !Passed
// CHECK: Pass:{{.*}}sycl
// CHECK: Name:{{.*}}Region
// CHECK: DebugLoc:{{.*}} { File: '{{.*}}kernel-arg-opt-report.cpp',
// CHECK-NEXT: Line: 24, Column: 8 }
// CHECK-NEXT: Function:        _ZTS13kernelfunctor
// CHECK-NEXT: Args:
// CHECK-NEXT: String:          'Arg '
// CHECK-NEXT: Argument:        '11'
// CHECK-NEXT: String:          ':'
// CHECK-NEXT: String:          Compiler generated argument for accessor,
// CHECK-NEXT: String:          acc
// CHECK-NEXT: String:          '  ('
// CHECK-NEXT: String:          ''
// CHECK-NEXT: String:          'Type:'
// CHECK-NEXT: String:          'struct sycl::id<1>'
// CHECK-NEXT: String:          ', '
// CHECK-NEXT: String:          'Size: '
// CHECK-NEXT: Argument:        '1'
// CHECK-NEXT: String:          ')'

// CHECK: --- !Passed
// CHECK: Pass:{{.*}}sycl
// CHECK: Name:{{.*}}Region
// CHECK: DebugLoc:{{.*}} { File: '{{.*}}kernel-arg-opt-report.cpp',
// CHECK-NEXT: Line: 24, Column: 8 }
// CHECK-NEXT: Function:        _ZTS13kernelfunctor
// CHECK-NEXT: Args:
// CHECK-NEXT: String:          'Arg '
// CHECK-NEXT: Argument:        '12'
// CHECK-NEXT: String:          ':'
// CHECK-NEXT: String:          ''
// CHECK-NEXT: String:          a
// CHECK-NEXT: String:          '  ('
// CHECK-NEXT: String:          ''
// CHECK-NEXT: String:          'Type:'
// CHECK-NEXT: String:          int
// CHECK-NEXT: String:          ', '
// CHECK-NEXT: String:          'Size: '
// CHECK-NEXT: Argument:        '4'
// CHECK-NEXT: String:          ')'

// CHECK: --- !Passed
// CHECK: Pass:{{.*}}sycl
// CHECK: Name:{{.*}}Region
// CHECK: DebugLoc:{{.*}} { File: '{{.*}}kernel-arg-opt-report.cpp',
// CHECK-NEXT: Line: 24, Column: 8 }
// CHECK-NEXT: Function:        _ZTS13kernelfunctor
// CHECK-NEXT: Args:
// CHECK-NEXT: String:          'Arg '
// CHECK-NEXT: Argument:        '13'
// CHECK-NEXT: String:          ':'
// CHECK-NEXT: String:          ''
// CHECK-NEXT: String:          ptr
// CHECK-NEXT: String:          '  ('
// CHECK-NEXT: String:          ''
// CHECK-NEXT: String:          'Type:'
// CHECK-NEXT: String:          '__global int *'
// CHECK-NEXT: String:          ', '
// CHECK-NEXT: String:          'Size: '
// CHECK-NEXT: Argument:        '8'
// CHECK-NEXT: String:          ')'

// CHECK: --- !Passed
// CHECK: Pass:{{.*}}sycl
// CHECK: Name:{{.*}}Region
// CHECK: DebugLoc:{{.*}} { File: '{{.*}}kernel-arg-opt-report.cpp',
// CHECK-NEXT: Line: 24, Column: 8 }
// CHECK-NEXT: Function:        _ZTS13kernelfunctor
// CHECK-NEXT: Args:
// CHECK-NEXT: String:          'Arg '
// CHECK-NEXT: Argument:        '14'
// CHECK-NEXT: String:          ':'
// CHECK-NEXT: String:          Compiler generated argument for array,
// CHECK-NEXT: String:          array
// CHECK-NEXT: String:          '  ('
// CHECK-NEXT: String:          ''
// CHECK-NEXT: String:          'Type:'
// CHECK-NEXT: String:          Compiler generated
// CHECK-NEXT: String:          ', '
// CHECK-NEXT: String:          'Size: '
// CHECK-NEXT: Argument:        '12'
// CHECK-NEXT: String:          ')'

// CHECK: --- !Passed
// CHECK: Pass:{{.*}}sycl
// CHECK: Name:{{.*}}Region
// CHECK: DebugLoc:{{.*}} { File: '{{.*}}kernel-arg-opt-report.cpp',
// CHECK-NEXT: Line: 24, Column: 8 }
// CHECK-NEXT: Function:        _ZTS13kernelfunctor
// CHECK-NEXT: Args:
// CHECK-NEXT: String:          'Arg '
// CHECK-NEXT: Argument:        '15'
// CHECK-NEXT: String:          ':'
// CHECK-NEXT: String:          Compiler generated argument for sampler,
// CHECK-NEXT: String:          sampl
// CHECK-NEXT: String:          '  ('
// CHECK-NEXT: String:          ''
// CHECK-NEXT: String:          'Type:'
// CHECK-NEXT: String:          sampler_t
// CHECK-NEXT: String:          ', '
// CHECK-NEXT: String:          'Size: '
// CHECK-NEXT: Argument:        '8'
// CHECK-NEXT: String:          ')'
