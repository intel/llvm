//==-- memcpy2d_host_to_device.cpp - 2D memcpy from host USM to device USM -==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// REQUIRES: aspect-usm_host_allocations, aspect-usm_device_allocations
// RUN: %{build} -o %t.out
// TODO: revert this once #23392 is root-caused and fixed.
// Run the batched (non-immediate command list) experiment first and
// unconditionally, capturing its output, so a real failure in the default
// path below can't prevent us from seeing whether batched mode avoids it.
// RUN: env UR_L0_V2_FORCE_BATCHED=1 %{run} %t.out > %t.batched1.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_BATCHED=1 %{run} %t.out > %t.batched2.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_BATCHED=1 %{run} %t.out > %t.batched3.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_BATCHED=1 %{run} %t.out > %t.batched4.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_BATCHED=1 %{run} %t.out > %t.batched5.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_BATCHED=1 %{run} %t.out > %t.batched6.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_BATCHED=1 %{run} %t.out > %t.batched7.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_BATCHED=1 %{run} %t.out > %t.batched8.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_BATCHED=1 %{run} %t.out > %t.batched9.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_BATCHED=1 %{run} %t.out > %t.batched10.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_BATCHED=1 %{run} %t.out > %t.batched11.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_BATCHED=1 %{run} %t.out > %t.batched12.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_BATCHED=1 %{run} %t.out > %t.batched13.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_BATCHED=1 %{run} %t.out > %t.batched14.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_BATCHED=1 %{run} %t.out > %t.batched15.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_BATCHED=1 %{run} %t.out > %t.batched16.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_BATCHED=1 %{run} %t.out > %t.batched17.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_BATCHED=1 %{run} %t.out > %t.batched18.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_BATCHED=1 %{run} %t.out > %t.batched19.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_BATCHED=1 %{run} %t.out > %t.batched20.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_BATCHED=1 %{run} %t.out > %t.batched21.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_BATCHED=1 %{run} %t.out > %t.batched22.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_BATCHED=1 %{run} %t.out > %t.batched23.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_BATCHED=1 %{run} %t.out > %t.batched24.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_BATCHED=1 %{run} %t.out > %t.batched25.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_BATCHED=1 %{run} %t.out > %t.batched26.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_BATCHED=1 %{run} %t.out > %t.batched27.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_BATCHED=1 %{run} %t.out > %t.batched28.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_BATCHED=1 %{run} %t.out > %t.batched29.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_BATCHED=1 %{run} %t.out > %t.batched30.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_BATCHED=1 %{run} %t.out > %t.batched31.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_BATCHED=1 %{run} %t.out > %t.batched32.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_BATCHED=1 %{run} %t.out > %t.batched33.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_BATCHED=1 %{run} %t.out > %t.batched34.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_BATCHED=1 %{run} %t.out > %t.batched35.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_BATCHED=1 %{run} %t.out > %t.batched36.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_BATCHED=1 %{run} %t.out > %t.batched37.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_BATCHED=1 %{run} %t.out > %t.batched38.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_BATCHED=1 %{run} %t.out > %t.batched39.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_BATCHED=1 %{run} %t.out > %t.batched40.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_BATCHED=1 %{run} %t.out > %t.batched41.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_BATCHED=1 %{run} %t.out > %t.batched42.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_BATCHED=1 %{run} %t.out > %t.batched43.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_BATCHED=1 %{run} %t.out > %t.batched44.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_BATCHED=1 %{run} %t.out > %t.batched45.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_BATCHED=1 %{run} %t.out > %t.batched46.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_BATCHED=1 %{run} %t.out > %t.batched47.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_BATCHED=1 %{run} %t.out > %t.batched48.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_BATCHED=1 %{run} %t.out > %t.batched49.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_BATCHED=1 %{run} %t.out > %t.batched50.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_BATCHED=1 %{run} %t.out > %t.batched51.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_BATCHED=1 %{run} %t.out > %t.batched52.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_BATCHED=1 %{run} %t.out > %t.batched53.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_BATCHED=1 %{run} %t.out > %t.batched54.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_BATCHED=1 %{run} %t.out > %t.batched55.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_BATCHED=1 %{run} %t.out > %t.batched56.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_BATCHED=1 %{run} %t.out > %t.batched57.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_BATCHED=1 %{run} %t.out > %t.batched58.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_BATCHED=1 %{run} %t.out > %t.batched59.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_BATCHED=1 %{run} %t.out > %t.batched60.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_BATCHED=1 %{run} %t.out > %t.batched61.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_BATCHED=1 %{run} %t.out > %t.batched62.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_BATCHED=1 %{run} %t.out > %t.batched63.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_BATCHED=1 %{run} %t.out > %t.batched64.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_BATCHED=1 %{run} %t.out > %t.batched65.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_BATCHED=1 %{run} %t.out > %t.batched66.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_BATCHED=1 %{run} %t.out > %t.batched67.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_BATCHED=1 %{run} %t.out > %t.batched68.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_BATCHED=1 %{run} %t.out > %t.batched69.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_BATCHED=1 %{run} %t.out > %t.batched70.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_BATCHED=1 %{run} %t.out > %t.batched71.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_BATCHED=1 %{run} %t.out > %t.batched72.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_BATCHED=1 %{run} %t.out > %t.batched73.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_BATCHED=1 %{run} %t.out > %t.batched74.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_BATCHED=1 %{run} %t.out > %t.batched75.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_BATCHED=1 %{run} %t.out > %t.batched76.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_BATCHED=1 %{run} %t.out > %t.batched77.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_BATCHED=1 %{run} %t.out > %t.batched78.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_BATCHED=1 %{run} %t.out > %t.batched79.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_BATCHED=1 %{run} %t.out > %t.batched80.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_BATCHED=1 %{run} %t.out > %t.batched81.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_BATCHED=1 %{run} %t.out > %t.batched82.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_BATCHED=1 %{run} %t.out > %t.batched83.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_BATCHED=1 %{run} %t.out > %t.batched84.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_BATCHED=1 %{run} %t.out > %t.batched85.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_BATCHED=1 %{run} %t.out > %t.batched86.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_BATCHED=1 %{run} %t.out > %t.batched87.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_BATCHED=1 %{run} %t.out > %t.batched88.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_BATCHED=1 %{run} %t.out > %t.batched89.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_BATCHED=1 %{run} %t.out > %t.batched90.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_BATCHED=1 %{run} %t.out > %t.batched91.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_BATCHED=1 %{run} %t.out > %t.batched92.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_BATCHED=1 %{run} %t.out > %t.batched93.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_BATCHED=1 %{run} %t.out > %t.batched94.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_BATCHED=1 %{run} %t.out > %t.batched95.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_BATCHED=1 %{run} %t.out > %t.batched96.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_BATCHED=1 %{run} %t.out > %t.batched97.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_BATCHED=1 %{run} %t.out > %t.batched98.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_BATCHED=1 %{run} %t.out > %t.batched99.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_BATCHED=1 %{run} %t.out > %t.batched100.log 2>&1 || true
// Check the batched logs now, before the default path below (which can
// abort the script on a real failure) gets a chance to run.
// RUN: not grep -l "did not match" %t.batched1.log %t.batched2.log %t.batched3.log %t.batched4.log %t.batched5.log %t.batched6.log %t.batched7.log %t.batched8.log %t.batched9.log %t.batched10.log %t.batched11.log %t.batched12.log %t.batched13.log %t.batched14.log %t.batched15.log %t.batched16.log %t.batched17.log %t.batched18.log %t.batched19.log %t.batched20.log
// RUN: not grep -l "did not match" %t.batched21.log %t.batched22.log %t.batched23.log %t.batched24.log %t.batched25.log %t.batched26.log %t.batched27.log %t.batched28.log %t.batched29.log %t.batched30.log %t.batched31.log %t.batched32.log %t.batched33.log %t.batched34.log %t.batched35.log %t.batched36.log %t.batched37.log %t.batched38.log %t.batched39.log %t.batched40.log
// RUN: not grep -l "did not match" %t.batched41.log %t.batched42.log %t.batched43.log %t.batched44.log %t.batched45.log %t.batched46.log %t.batched47.log %t.batched48.log %t.batched49.log %t.batched50.log %t.batched51.log %t.batched52.log %t.batched53.log %t.batched54.log %t.batched55.log %t.batched56.log %t.batched57.log %t.batched58.log %t.batched59.log %t.batched60.log
// RUN: not grep -l "did not match" %t.batched61.log %t.batched62.log %t.batched63.log %t.batched64.log %t.batched65.log %t.batched66.log %t.batched67.log %t.batched68.log %t.batched69.log %t.batched70.log %t.batched71.log %t.batched72.log %t.batched73.log %t.batched74.log %t.batched75.log %t.batched76.log %t.batched77.log %t.batched78.log %t.batched79.log %t.batched80.log
// RUN: not grep -l "did not match" %t.batched81.log %t.batched82.log %t.batched83.log %t.batched84.log %t.batched85.log %t.batched86.log %t.batched87.log %t.batched88.log %t.batched89.log %t.batched90.log %t.batched91.log %t.batched92.log %t.batched93.log %t.batched94.log %t.batched95.log %t.batched96.log %t.batched97.log %t.batched98.log %t.batched99.log %t.batched100.log
// Isolate the copy-offload-hint variable from the submission-mode
// variable: keep immediate command lists (the default), but force the
// driver to disable copy engine offload globally, independent of any
// DG2-specific gating in this adapter's own code.
// RUN: env UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD=1 %{run} %t.out > %t.offload_disabled1.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD=1 %{run} %t.out > %t.offload_disabled2.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD=1 %{run} %t.out > %t.offload_disabled3.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD=1 %{run} %t.out > %t.offload_disabled4.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD=1 %{run} %t.out > %t.offload_disabled5.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD=1 %{run} %t.out > %t.offload_disabled6.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD=1 %{run} %t.out > %t.offload_disabled7.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD=1 %{run} %t.out > %t.offload_disabled8.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD=1 %{run} %t.out > %t.offload_disabled9.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD=1 %{run} %t.out > %t.offload_disabled10.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD=1 %{run} %t.out > %t.offload_disabled11.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD=1 %{run} %t.out > %t.offload_disabled12.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD=1 %{run} %t.out > %t.offload_disabled13.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD=1 %{run} %t.out > %t.offload_disabled14.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD=1 %{run} %t.out > %t.offload_disabled15.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD=1 %{run} %t.out > %t.offload_disabled16.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD=1 %{run} %t.out > %t.offload_disabled17.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD=1 %{run} %t.out > %t.offload_disabled18.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD=1 %{run} %t.out > %t.offload_disabled19.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD=1 %{run} %t.out > %t.offload_disabled20.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD=1 %{run} %t.out > %t.offload_disabled21.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD=1 %{run} %t.out > %t.offload_disabled22.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD=1 %{run} %t.out > %t.offload_disabled23.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD=1 %{run} %t.out > %t.offload_disabled24.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD=1 %{run} %t.out > %t.offload_disabled25.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD=1 %{run} %t.out > %t.offload_disabled26.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD=1 %{run} %t.out > %t.offload_disabled27.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD=1 %{run} %t.out > %t.offload_disabled28.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD=1 %{run} %t.out > %t.offload_disabled29.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD=1 %{run} %t.out > %t.offload_disabled30.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD=1 %{run} %t.out > %t.offload_disabled31.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD=1 %{run} %t.out > %t.offload_disabled32.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD=1 %{run} %t.out > %t.offload_disabled33.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD=1 %{run} %t.out > %t.offload_disabled34.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD=1 %{run} %t.out > %t.offload_disabled35.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD=1 %{run} %t.out > %t.offload_disabled36.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD=1 %{run} %t.out > %t.offload_disabled37.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD=1 %{run} %t.out > %t.offload_disabled38.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD=1 %{run} %t.out > %t.offload_disabled39.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD=1 %{run} %t.out > %t.offload_disabled40.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD=1 %{run} %t.out > %t.offload_disabled41.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD=1 %{run} %t.out > %t.offload_disabled42.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD=1 %{run} %t.out > %t.offload_disabled43.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD=1 %{run} %t.out > %t.offload_disabled44.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD=1 %{run} %t.out > %t.offload_disabled45.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD=1 %{run} %t.out > %t.offload_disabled46.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD=1 %{run} %t.out > %t.offload_disabled47.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD=1 %{run} %t.out > %t.offload_disabled48.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD=1 %{run} %t.out > %t.offload_disabled49.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD=1 %{run} %t.out > %t.offload_disabled50.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD=1 %{run} %t.out > %t.offload_disabled51.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD=1 %{run} %t.out > %t.offload_disabled52.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD=1 %{run} %t.out > %t.offload_disabled53.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD=1 %{run} %t.out > %t.offload_disabled54.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD=1 %{run} %t.out > %t.offload_disabled55.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD=1 %{run} %t.out > %t.offload_disabled56.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD=1 %{run} %t.out > %t.offload_disabled57.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD=1 %{run} %t.out > %t.offload_disabled58.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD=1 %{run} %t.out > %t.offload_disabled59.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD=1 %{run} %t.out > %t.offload_disabled60.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD=1 %{run} %t.out > %t.offload_disabled61.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD=1 %{run} %t.out > %t.offload_disabled62.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD=1 %{run} %t.out > %t.offload_disabled63.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD=1 %{run} %t.out > %t.offload_disabled64.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD=1 %{run} %t.out > %t.offload_disabled65.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD=1 %{run} %t.out > %t.offload_disabled66.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD=1 %{run} %t.out > %t.offload_disabled67.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD=1 %{run} %t.out > %t.offload_disabled68.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD=1 %{run} %t.out > %t.offload_disabled69.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD=1 %{run} %t.out > %t.offload_disabled70.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD=1 %{run} %t.out > %t.offload_disabled71.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD=1 %{run} %t.out > %t.offload_disabled72.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD=1 %{run} %t.out > %t.offload_disabled73.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD=1 %{run} %t.out > %t.offload_disabled74.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD=1 %{run} %t.out > %t.offload_disabled75.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD=1 %{run} %t.out > %t.offload_disabled76.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD=1 %{run} %t.out > %t.offload_disabled77.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD=1 %{run} %t.out > %t.offload_disabled78.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD=1 %{run} %t.out > %t.offload_disabled79.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD=1 %{run} %t.out > %t.offload_disabled80.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD=1 %{run} %t.out > %t.offload_disabled81.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD=1 %{run} %t.out > %t.offload_disabled82.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD=1 %{run} %t.out > %t.offload_disabled83.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD=1 %{run} %t.out > %t.offload_disabled84.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD=1 %{run} %t.out > %t.offload_disabled85.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD=1 %{run} %t.out > %t.offload_disabled86.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD=1 %{run} %t.out > %t.offload_disabled87.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD=1 %{run} %t.out > %t.offload_disabled88.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD=1 %{run} %t.out > %t.offload_disabled89.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD=1 %{run} %t.out > %t.offload_disabled90.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD=1 %{run} %t.out > %t.offload_disabled91.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD=1 %{run} %t.out > %t.offload_disabled92.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD=1 %{run} %t.out > %t.offload_disabled93.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD=1 %{run} %t.out > %t.offload_disabled94.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD=1 %{run} %t.out > %t.offload_disabled95.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD=1 %{run} %t.out > %t.offload_disabled96.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD=1 %{run} %t.out > %t.offload_disabled97.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD=1 %{run} %t.out > %t.offload_disabled98.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD=1 %{run} %t.out > %t.offload_disabled99.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD=1 %{run} %t.out > %t.offload_disabled100.log 2>&1 || true
// RUN: not grep -l "did not match" %t.offload_disabled1.log %t.offload_disabled2.log %t.offload_disabled3.log %t.offload_disabled4.log %t.offload_disabled5.log %t.offload_disabled6.log %t.offload_disabled7.log %t.offload_disabled8.log %t.offload_disabled9.log %t.offload_disabled10.log %t.offload_disabled11.log %t.offload_disabled12.log %t.offload_disabled13.log %t.offload_disabled14.log %t.offload_disabled15.log %t.offload_disabled16.log %t.offload_disabled17.log %t.offload_disabled18.log %t.offload_disabled19.log %t.offload_disabled20.log
// RUN: not grep -l "did not match" %t.offload_disabled21.log %t.offload_disabled22.log %t.offload_disabled23.log %t.offload_disabled24.log %t.offload_disabled25.log %t.offload_disabled26.log %t.offload_disabled27.log %t.offload_disabled28.log %t.offload_disabled29.log %t.offload_disabled30.log %t.offload_disabled31.log %t.offload_disabled32.log %t.offload_disabled33.log %t.offload_disabled34.log %t.offload_disabled35.log %t.offload_disabled36.log %t.offload_disabled37.log %t.offload_disabled38.log %t.offload_disabled39.log %t.offload_disabled40.log
// RUN: not grep -l "did not match" %t.offload_disabled41.log %t.offload_disabled42.log %t.offload_disabled43.log %t.offload_disabled44.log %t.offload_disabled45.log %t.offload_disabled46.log %t.offload_disabled47.log %t.offload_disabled48.log %t.offload_disabled49.log %t.offload_disabled50.log %t.offload_disabled51.log %t.offload_disabled52.log %t.offload_disabled53.log %t.offload_disabled54.log %t.offload_disabled55.log %t.offload_disabled56.log %t.offload_disabled57.log %t.offload_disabled58.log %t.offload_disabled59.log %t.offload_disabled60.log
// RUN: not grep -l "did not match" %t.offload_disabled61.log %t.offload_disabled62.log %t.offload_disabled63.log %t.offload_disabled64.log %t.offload_disabled65.log %t.offload_disabled66.log %t.offload_disabled67.log %t.offload_disabled68.log %t.offload_disabled69.log %t.offload_disabled70.log %t.offload_disabled71.log %t.offload_disabled72.log %t.offload_disabled73.log %t.offload_disabled74.log %t.offload_disabled75.log %t.offload_disabled76.log %t.offload_disabled77.log %t.offload_disabled78.log %t.offload_disabled79.log %t.offload_disabled80.log
// RUN: not grep -l "did not match" %t.offload_disabled81.log %t.offload_disabled82.log %t.offload_disabled83.log %t.offload_disabled84.log %t.offload_disabled85.log %t.offload_disabled86.log %t.offload_disabled87.log %t.offload_disabled88.log %t.offload_disabled89.log %t.offload_disabled90.log %t.offload_disabled91.log %t.offload_disabled92.log %t.offload_disabled93.log %t.offload_disabled94.log %t.offload_disabled95.log %t.offload_disabled96.log %t.offload_disabled97.log %t.offload_disabled98.log %t.offload_disabled99.log %t.offload_disabled100.log
// Default (immediate command list) path; repeated for intermittent
// failures. A real failure here aborts the test, as before.
// RUN: %{run} %t.out
// RUN: %{run} %t.out
// RUN: %{run} %t.out
// RUN: %{run} %t.out
// RUN: %{run} %t.out
// RUN: %{run} %t.out
// RUN: %{run} %t.out
// RUN: %{run} %t.out
// RUN: %{run} %t.out
// RUN: %{run} %t.out
// RUN: %{run} %t.out
// RUN: %{run} %t.out
// RUN: %{run} %t.out
// RUN: %{run} %t.out
// RUN: %{run} %t.out
// RUN: %{run} %t.out
// RUN: %{run} %t.out
// RUN: %{run} %t.out
// RUN: %{run} %t.out
// RUN: %{run} %t.out
// RUN: %{run} %t.out
// RUN: %{run} %t.out
// RUN: %{run} %t.out
// RUN: %{run} %t.out
// RUN: %{run} %t.out
// RUN: %{run} %t.out
// RUN: %{run} %t.out
// RUN: %{run} %t.out
// RUN: %{run} %t.out
// RUN: %{run} %t.out
// RUN: %{run} %t.out
// RUN: %{run} %t.out
// RUN: %{run} %t.out
// RUN: %{run} %t.out
// RUN: %{run} %t.out
// RUN: %{run} %t.out
// RUN: %{run} %t.out
// RUN: %{run} %t.out
// RUN: %{run} %t.out
// RUN: %{run} %t.out
// RUN: %{run} %t.out
// RUN: %{run} %t.out
// RUN: %{run} %t.out
// RUN: %{run} %t.out
// RUN: %{run} %t.out
// RUN: %{run} %t.out
// RUN: %{run} %t.out
// RUN: %{run} %t.out
// RUN: %{run} %t.out
// RUN: %{run} %t.out
// RUN: %{run} %t.out
// RUN: %{run} %t.out
// RUN: %{run} %t.out
// RUN: %{run} %t.out
// RUN: %{run} %t.out
// RUN: %{run} %t.out
// RUN: %{run} %t.out
// RUN: %{run} %t.out
// RUN: %{run} %t.out
// RUN: %{run} %t.out
// RUN: %{run} %t.out
// RUN: %{run} %t.out
// RUN: %{run} %t.out
// RUN: %{run} %t.out
// RUN: %{run} %t.out
// RUN: %{run} %t.out
// RUN: %{run} %t.out
// RUN: %{run} %t.out
// RUN: %{run} %t.out
// RUN: %{run} %t.out
// RUN: %{run} %t.out
// RUN: %{run} %t.out
// RUN: %{run} %t.out
// RUN: %{run} %t.out
// RUN: %{run} %t.out
// RUN: %{run} %t.out
// RUN: %{run} %t.out
// RUN: %{run} %t.out
// RUN: %{run} %t.out
// RUN: %{run} %t.out
// RUN: %{run} %t.out
// RUN: %{run} %t.out
// RUN: %{run} %t.out
// RUN: %{run} %t.out
// RUN: %{run} %t.out
// RUN: %{run} %t.out
// RUN: %{run} %t.out
// RUN: %{run} %t.out
// RUN: %{run} %t.out
// RUN: %{run} %t.out
// RUN: %{run} %t.out
// RUN: %{run} %t.out
// RUN: %{run} %t.out
// RUN: %{run} %t.out
// RUN: %{run} %t.out
// RUN: %{run} %t.out
// RUN: %{run} %t.out
// RUN: %{run} %t.out
// RUN: %{run} %t.out
// RUN: %{run} %t.out

// UNSUPPORTED: hip
// UNSUPPORTED-TRACKER: https://github.com/intel/llvm/issues/10157

#include "memcpy2d_common.hpp"

int main() { return test<Alloc::Host, Alloc::Device>(); }
