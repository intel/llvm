//==---------------- kmeans.h  - DPC++ ESIMD on-device test ----------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef KMEANS_H_INCLUDED
#define KMEANS_H_INCLUDED

#define NUM_POINTS 98304 * 8

#define POINTS_PER_THREAD 256

// can handle up to 160 centroids without spilling
#define NUM_CENTROIDS 20

// round to next 16 so as to handle reading centroids easily
// read 16 centroids at one time
#define ROUND_TO_16_NUM_CENTROIDS (((NUM_CENTROIDS - 1) / 16 + 1) * 16)

#define ACCUM_REDUCTION_RATIO 1

#endif // KMEANS_H_INCLUDED
