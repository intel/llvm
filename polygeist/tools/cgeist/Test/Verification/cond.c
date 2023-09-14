// RUN: cgeist %s %stdinclude --function=set -S | FileCheck %s

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>


/* Array initialization. */

int set (int b)
{
    int res;
    if (b)
        res = 1;
    else
        res = 2;
    return res;
  //path[0][1] = 2;
}

// CHECK:  func @set(%arg0: i32) -> i32
// CHCEK-NEXT:     %c1_i32 = constant 1 : i32
// CHCEK-NEXT:     %c2_i32 = constant 2 : i32
// CHCEK-NEXT:     %0 = arith.trunci %arg0 : i32 to i1
// CHCEK-NEXT:     %1 = arith.select %0, %c1_i32, %c2_i32 : i32
// CHCEK-NEXT:     return %1 : i32
// CHCEK-NEXT:   }
