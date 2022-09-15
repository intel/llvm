// RUN: %clang_cc1 -fsycl-is-device -triple spir64-unknown-unknown -disable-llvm-passes -emit-llvm %s -o - | FileCheck %s

// Tests for IR of [[__sycl_detail__::sycl_type(aspect)]] enum.
enum class [[__sycl_detail__::sycl_type(aspect)]] aspect {
  host = 0,
  cpu = 1,
  gpu = 2,
  accelerator = 3,
  custom = 4,
  fp16 = 5,
  fp64 = 6,
  future_aspect = 12
};

// CHECK: !intel_sycl_aspects = !{![[HOST:[0-9]+]], ![[CPU:[0-9]+]], ![[GPU:[0-9]+]], ![[ACC:[0-9]+]], ![[CUSTOM:[0-9]+]], ![[FP16:[0-9]+]], ![[FP64:[0-9]+]], ![[FUTURE_ASPECT:[0-9]+]]}
// CHECK: [[HOST]] = !{!"host", i32 0}
// CHECK: [[CPU]] = !{!"cpu", i32 1}
// CHECK: [[GPU]] = !{!"gpu", i32 2}
// CHECK: [[ACC]] = !{!"accelerator", i32 3}
// CHECK: [[CUSTOM]] = !{!"custom", i32 4}
// CHECK: [[FP16]] = !{!"fp16", i32 5}
// CHECK: [[FP64]] = !{!"fp64", i32 6}
// CHECK: [[FUTURE_ASPECT]] = !{!"future_aspect", i32 12}
