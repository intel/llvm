// RUN: %{build} -fsycl-device-code-split=per_source -I %S/Inputs -o %t.out %S/Inputs/split-per-source-second-file.cpp \
// RUN: --offload-new-driver -fsycl-dead-args-optimization
// RUN: %{run} %t.out
//
// XFAIL: hip_nvidia

#include "Inputs/split-per-source.h"

int main() {
  sycl::queue Q;
  int Data = 0;

  auto KernelID = sycl::get_kernel_id<File1Kern1>();
  auto KB = sycl::get_kernel_bundle<sycl::bundle_state::executable>(
      Q.get_context(), {KernelID});
  assert(KB.has_kernel(KernelID));
  auto Krn1 = KB.get_kernel(KernelID);

  auto KernelID2 = sycl::get_kernel_id<File1Kern2>();
  assert(KB.has_kernel(KernelID2));
  auto Krn2 = KB.get_kernel(KernelID2);

  std::vector<sycl::kernel_id> KernelIDStorage = KB.get_kernel_ids();
  assert(KernelIDStorage.size() == 2);
  assert(std::any_of(
      KernelIDStorage.begin(), KernelIDStorage.end(),
      [&KernelID](const sycl::kernel_id &id) { return id == KernelID; }));
  assert(std::any_of(
      KernelIDStorage.begin(), KernelIDStorage.end(),
      [&KernelID2](const sycl::kernel_id &id) { return id == KernelID2; }));

  {
    sycl::buffer<int, 1> Buf(&Data, sycl::range<1>(1));
    Q.submit([&](sycl::handler &Cgh) {
      auto Acc = Buf.get_access<sycl::access::mode::read_write>(Cgh);
      Cgh.single_task<File1Kern1>(Krn1, [=]() { Acc[0] = 1; });
    });
  }
  assert(Data == 1);

  {
    sycl::buffer<int, 1> Buf(&Data, sycl::range<1>(1));
    Q.submit([&](sycl::handler &Cgh) {
      auto Acc = Buf.get_access<sycl::access::mode::read_write>(Cgh);
      Cgh.single_task<File1Kern2>(Krn2, [=]() { Acc[0] = 2; });
    });
  }
  assert(Data == 2);

  runKernelsFromFile2();

  return 0;
}
