static int __attribute__((always_inline)) foo() { return 10; }
int func() { return foo(); }