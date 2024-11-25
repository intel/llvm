// UNSUPPORTED: windows
// REQUIRES: cuda || hip
// REQUIRES: build-and-run-mode

// This test relies on debug output from a pass, make sure that the compiler
// can generate it.
// REQUIRES: has_ndebug

// RUN: %{build} -fsycl-embed-ir -o %t.out
// RUN: env SYCL_JIT_AMDGCN_PTX_KERNELS=1 env SYCL_JIT_COMPILER_DEBUG="sycl-spec-const-materializer" %{run} %t.out &> %t.txt ; FileCheck %s --input-file %t.txt

// Test the JIT compilation in an e2e fashion, the only way to make sure that
// the JIT pipeline has been executed and that the original binary has been
// replaced with the JIT-ed one is to inspect the output of one of its passes,
// that otherwise does not get run.

#include <sycl/detail/core.hpp>
#include <sycl/specialization_id.hpp>

constexpr size_t Size = 16;
constexpr int SeedKernel = 3;
constexpr int SeedKernelBundle = 5;

constexpr int ValInt = 11;
constexpr std::array<int, 2> ValArr{13, 17};
const static sycl::specialization_id<int> SpecConstInt;
const static sycl::specialization_id<std::array<int, 2>> SpecConstArr;

int validate(int Seed, std::vector<int> &Input, std::vector<int> &Output) {
  for (int i = 0; i < Size; ++i) {
    int Expected = ValInt + ValArr[0] + ValArr[1] + Input[i] + Seed;
    if (Expected != Output[i]) {
      return -1;
    }
  }
  return 0;
}

// CHECK: Working on function:
// CHECK: ==================
// CHECK: _ZTSZ15runKernelBundleN4sycl3_V15queueERSt6vectorIiSaIiEES5_E10WoofBundle
int runKernelBundle(sycl::queue Queue, std::vector<int> &Input,
                    std::vector<int> &Output) {
  for (int i = 0; i < Size; ++i) {
    Output[i] = 42;
    Input[i] = i * i;
  }

  sycl::device Device;
  sycl::context Context = Queue.get_context();

  auto InputBundle =
      sycl::get_kernel_bundle<class WoofBundle, sycl::bundle_state::input>(
          Context, {Device});
  InputBundle.set_specialization_constant<SpecConstInt>(ValInt);
  InputBundle.set_specialization_constant<SpecConstArr>(ValArr);

  auto ExecBundle = sycl::build(InputBundle);

  {
    sycl::buffer<int> OutBuff(Output.data(), Output.size());
    sycl::buffer<int> InBuff(Input.data(), Input.size());
    Queue.submit([&](sycl::handler &cgh) {
      sycl::accessor OutAcc(OutBuff, cgh, sycl::write_only);
      sycl::accessor InAcc(InBuff, cgh, sycl::read_only);
      cgh.use_kernel_bundle(ExecBundle);
      cgh.template parallel_for<class WoofBundle>(
          sycl::range<1>{Size}, [=](sycl::id<1> i, sycl::kernel_handler kh) {
            const auto KernelSpecConst =
                kh.get_specialization_constant<SpecConstInt>();
            const auto KernelSpecConstArr =
                kh.get_specialization_constant<SpecConstArr>();
            OutAcc[i] = KernelSpecConst + KernelSpecConstArr[0] +
                        KernelSpecConstArr[1] + InAcc[i] + SeedKernelBundle;
          });
    });
    Queue.wait_and_throw();
  }

  return validate(SeedKernelBundle, Input, Output);
}

// CHECK: Working on function:
// CHECK: ==================
// CHECK: _ZTSZZ9runKernelN4sycl3_V15queueERSt6vectorIiSaIiEES5_ENKUlRT_E_clINS0_7handlerEEEDaS7_E10WoofKernel
int runKernel(sycl::queue Queue, std::vector<int> &Input,
              std::vector<int> &Output) {
  for (int i = 0; i < Size; ++i) {
    Output[i] = 42;
    Input[i] = i * i;
  }
  {
    sycl::buffer<int> OutBuff(Output.data(), Output.size());
    sycl::buffer<int> InBuff(Input.data(), Input.size());
    Queue.submit([&](auto &CGH) {
      sycl::accessor OutAcc(OutBuff, CGH, sycl::write_only);
      sycl::accessor InAcc(InBuff, CGH, sycl::read_only);
      CGH.template set_specialization_constant<SpecConstInt>(ValInt);
      CGH.template set_specialization_constant<SpecConstArr>(ValArr);
      CGH.template parallel_for<class WoofKernel>(
          sycl::range<1>{Size}, [=](sycl::id<1> i, sycl::kernel_handler KH) {
            const auto KernelSpecConst =
                KH.get_specialization_constant<SpecConstInt>();
            const auto KernelSpecConstArr =
                KH.get_specialization_constant<SpecConstArr>();
            OutAcc[i] = KernelSpecConst + KernelSpecConstArr[0] +
                        KernelSpecConstArr[1] + InAcc[i] + SeedKernel;
          });
    });
    Queue.wait_and_throw();
  }

  return validate(SeedKernel, Input, Output);
}

int main() {
  std::vector<int> Input(Size);
  std::vector<int> Output(Size);
  sycl::queue Queue;
  return runKernel(Queue, Input, Output) |
         runKernelBundle(Queue, Input, Output);
}
