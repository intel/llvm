// UNSUPPORTED: system-windows
// 
// Test to determine that temp file cleanup is working for fat static archives
// creates a single file fat archive then uses that
// RUN: rm -rf %t
// RUN: mkdir %t
// Build static library
// RUN: %clang -DBUILD_B -fsycl -c -o %t_obj.o %s
// RUN: ar cr %t_lib.a %t_obj.o
// Build main object
// RUN: %clang -DBUILD_A -fsycl -c -o %t_main.o %s
// Build final binary, overriding output temp location
// RUN: env TMPDIR=%t TEMP=%t TMP=%t                                     \
// RUN: %clang -fsycl %t_main.o -foffload-static-lib=%t_lib.a
// RUN: not ls %t/*
#ifdef BUILD_A
const int VAL = 10;
extern int run_test_b(int);

int run_test_a(int v) {
  return v*4;
}

int main(int argc, char **argv) {
  run_test_a(VAL);
  run_test_b(VAL);
  return 0;
}
#endif // BUILD_A
#if BUILD_B
int run_test_b(int v) {
  return v*3;
}
#endif // BUILD_B

