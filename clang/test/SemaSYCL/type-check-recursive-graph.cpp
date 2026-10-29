// RUN: %clang_cc1 -triple spir64 -fsycl-is-device -sycl-std=2020 -fsyntax-only -verify %s
// expected-no-diagnostics

// Regression test for https://github.com/intel/llvm/issues/22972.
//
// SYCL device type checking walks the field/pointer graph of every type used
// in device code. Each level below has two pointers to the next, so the graph
// has only N+1 nodes but 2^N distinct root-to-leaf paths. When the traversal
// deduplicated visited types per path instead of globally, the walk became
// exponential and the frontend hung on densely cross-linked graphs such as
// Dear ImGui. With global deduplication every type is visited once and this
// compiles instantly; before the fix it would run for hours and be killed by
// the test timeout.

struct Leaf { int x; };
struct L39 { Leaf *a; Leaf *b; };
struct L38 { L39 *a; L39 *b; };
struct L37 { L38 *a; L38 *b; };
struct L36 { L37 *a; L37 *b; };
struct L35 { L36 *a; L36 *b; };
struct L34 { L35 *a; L35 *b; };
struct L33 { L34 *a; L34 *b; };
struct L32 { L33 *a; L33 *b; };
struct L31 { L32 *a; L32 *b; };
struct L30 { L31 *a; L31 *b; };
struct L29 { L30 *a; L30 *b; };
struct L28 { L29 *a; L29 *b; };
struct L27 { L28 *a; L28 *b; };
struct L26 { L27 *a; L27 *b; };
struct L25 { L26 *a; L26 *b; };
struct L24 { L25 *a; L25 *b; };
struct L23 { L24 *a; L24 *b; };
struct L22 { L23 *a; L23 *b; };
struct L21 { L22 *a; L22 *b; };
struct L20 { L21 *a; L21 *b; };
struct L19 { L20 *a; L20 *b; };
struct L18 { L19 *a; L19 *b; };
struct L17 { L18 *a; L18 *b; };
struct L16 { L17 *a; L17 *b; };
struct L15 { L16 *a; L16 *b; };
struct L14 { L15 *a; L15 *b; };
struct L13 { L14 *a; L14 *b; };
struct L12 { L13 *a; L13 *b; };
struct L11 { L12 *a; L12 *b; };
struct L10 { L11 *a; L11 *b; };
struct L9  { L10 *a; L10 *b; };
struct L8  { L9  *a; L9  *b; };
struct L7  { L8  *a; L8  *b; };
struct L6  { L7  *a; L7  *b; };
struct L5  { L6  *a; L6  *b; };
struct L4  { L5  *a; L5  *b; };
struct L3  { L4  *a; L4  *b; };
struct L2  { L3  *a; L3  *b; };
struct L1  { L2  *a; L2  *b; };
struct Root { L1 *a; L1 *b; };

// A namespace-scope variable is enough to trigger the device type check.
Root g;
