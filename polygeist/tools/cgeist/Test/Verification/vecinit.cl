// RUN: cgeist -O0 %s --function=* -S | FileCheck %s
// XFAIL: *

float8 test0(float A, float B, float C, float D,
             float E, float F, float G, float H) {
  return (float8)(A, B, C, D, E, F, G, H);
}

float8 test1(float8 Arg0) {
  return (float8)(Arg0);
}

float8 test4(float4 A, float4 B) {
  return (float8)(A, B);
}

float8 test5(float4 A, float B, float C, float D, float E) {
  return (float8)(A, B, C, D, E);
}

float8 test6(float2 A, float B, float C, float D, float E, float F, float G) {
  return (float8)(A, B, C, D, E, F, G);
}

float8 test7(float A, float2 B, float C, float D, float E, float F, float G) {
  return (float8)(A, B, C, D, E, F, G);
}
