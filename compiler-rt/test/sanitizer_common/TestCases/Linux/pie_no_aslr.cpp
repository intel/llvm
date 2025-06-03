// There is a bug with Linux kernel with la57
// https://lore.kernel.org/all/20230803151609.22141-1-kirill.shutemov@linux.intel.com/ ,
// some of our machines are still affected, disable this test for now.
// UNSUPPORTED: tsan && la57

// RUN: %clangxx %s -pie -fPIE -o %t && %run setarch x86_64 -R %t
// REQUIRES: x86_64-target-arch

int main() { return 0; }
