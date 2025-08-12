# Copyright (C) 2025 Intel Corporation
# Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE.TXT
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# entry format:
#  [bench_driver, bench_name, bench_args, rungraph]
#  bench_driver is the name of the benchdnn driver, e.g. "sum", "graph", etc.
#  bench_name is the display name of the benchmark,
#  bench_args is the benchdnn arguments for the benchmark, e.g. "--sdt=f16:f16:f16 --stag=abx:abx:abx --scales=1.25:3:0.5 16x2x6x4x3"
#  rungraph is an optional boolean value indicating whether to run the benchmark in graph mode or not (default is True)
#    if rungraph is True, both direct and graph execution modes will be run for the benchmark
#    if False, only direct execution mode will be run

# the final choice of benchmarks to run, used in CI and other environments
benches_final_set = [
    # [
    #     "sum",
    #     "f16-1",
    #     "--sdt=f16:f16:f16 --stag=abx:abx:abx --scales=1.25:3:0.5 16x2x6x4x3",
    #     False,  # Do not run graph for this benchmark
    # ],
    # [
    #     "sum",
    #     "f16-2",
    #     "--reset --ddt=f16 \
    #         --sdt=f16:f16:f16:f16:f16:f16:f16:f16:f16:f16 \
    #         --stag=abx:aBx16b:ABx16a16b:ABcd16b16a:BAcd16a16b:BAcd16b16a:aBCd16b16c:aBCd16c16b:aCBd16b16c:aCBd16c16b \
    #         --dtag=abx,aBx16b,ABx16a16b,ABcd16b16a,BAcd16a16b,BAcd16b16a,aBCd16b16c,aBCd16c16b,aCBd16b16c,aCBd16c16b \
    #         --scales=1.25:3:0.5:2:0.5:2:0.5:2:0.5:2 \
    #         16x32x48x5",
    #     False,  # Do not run graph for this benchmark
    # ],
    # [
    #     "sum",
    #     "f32-1",
    #     "--sdt=bf16:bf16:bf16 --stag=abx:abx:abx --scales=0.5:2:0.5    16x2x6x4x3",
    #     False,  # Do not run graph for this benchmark
    # ],
    [
        "sum",
        "f32-2",
        "--reset --inplace=true,false --ddt=bf16 \
            --sdt=bf16:bf16:bf16:bf16:bf16:bf16:bf16:bf16:bf16:bf16 \
            --stag=abx:aBx16b:ABx16a16b:ABcd16b16a:BAcd16a16b:BAcd16b16a:aBCd16b16c:aBCd16c16b:aCBd16b16c:aCBd16c16b \
            --dtag=abx,aBx16b,ABx16a16b,ABcd16b16a,BAcd16a16b,BAcd16b16a,aBCd16b16c,aBCd16c16b,aCBd16b16c,aCBd16c16b \
            --scales=0.25:0.15:0.25:0.25:0.25:0.25:0.15:0.25:0.25:0.15 \
            16x32x48x5",
        False,  # Do not run graph for this benchmark
    ],
    [
        "sum",
        "padding-1",
        "--ddt=f32 --sdt=f32:f32 --stag=aBcd16b --dtag=aBcd16b 1x8x64x64 1x8x640x1024 1x24x640x1024",
    ],
    [
        "sum",
        "padding-2",
        "--sdt=bf16:bf16 --ddt=bf16 --stag=AB48a16b:AB48a16b --dtag=AB48a16b 512x1024",
    ],
    [
        "graph",
        "sdpa-plain-f16",
        "--reset --dt=f16 --case=complex_fusion/mha/sdpa-plain-implicit-causal-mask-fp32-bs1.json",
        False,  # Do not run SYCL graph for this benchmark
    ],
    [
        "graph",
        "sdpa-plain-f32",
        "--reset --dt=f32 --case=complex_fusion/mha/sdpa-plain-implicit-causal-mask-fp32-bs1.json",
    ],
]

# the complete set of benchmarks aimed at gpu operations, normally too long to run in CI
benches_all_big = [
    # ["binary", "gpu", "--batch=test_binary_gpu"],
    ["bnorm", "gpu", "--batch=test_bnorm_gpu"],
    ["brgemm", "bf16", "--batch=test_brgemm_bf16"],
    ["concat", "gpu", "--batch=test_concat_gpu"],
    ["concat", "large", "--batch=test_concat_large_gpu", False],
    ["conv", "set-gpu", "--batch=set_gpu"],
    ["deconv", "smoke", "--batch=test_deconv_smoke"],
    ["eltwise", "gpu", "--batch=test_eltwise_gpu"],
    ["gnorm", "all", "--batch=test_gnorm_all"],
    ["graph", "fusions", "--batch=test_graph_fusions_gpu"],
    ["graph", "op", "--batch=test_graph_op_gpu"],
    ["graph", "pattern", "--batch=test_graph_pattern_gpu"],
    ["ip", "gpu", "--batch=test_ip_gpu"],
    ["ip", "large-gpu", "--batch=test_ip_large_gpu", False],
    ["matmul", "llm-gpu", "--batch=test_matmul_llm_gpu"],
    ["pool", "gpu", "--batch=test_pool_gpu"],
    ["prelu", "gpu", "--batch=test_prelu_gpu"],
    ["reduction", "gpu", "--batch=perf_reduction_gpu"],
    ["reorder", "gpu", "--batch=test_reorder_gpu"],
    ["resampling", "gpu", "--batch=test_resampling_gpu"],
    ["rnn", "gpu", "--batch=test_rnn_gpu", False],
    ["shuffle", "gpu", "--batch=test_shuffle_gpu"],
    ["softmax", "gpu", "--batch=test_softmax_gpu"],
    ["sum", "gpu", "--batch=test_sum_gpu"],
    ["zeropad", "gpu", "--batch=test_zeropad_gpu"],
]

# miscellaneous sets of benchmarks for future use or modifications
benches_binary_gpu = [
    [
        "binary",
        "gen9_binary",
        "--reset --inplace=false --attr-post-ops=,sum:0.25+relu:-0.01+add:f32 --alg=ADD \
            --ddt=f32 --sdt=f32:f32 --stag=abcd:abcd,abcd:aBcd16b,aBcd16b:abcd,aBcd16b:aBcd16b --dtag=abcd,aBcd16b \
            1x1024x7x7:1x1024x1x1 1x16x16x16:1x16x16x16 1x16x16x16:1x16x16x1 1x16x16x16:1x16x1x16 1x16x16x16:1x1x16x16",
        False,  # Do not run graph for this benchmark
    ],
    [
        "binary",
        "ci-nightly",
        "--reset --batch=test_binary_ci",
    ],
]
benches_sum_gpu = [
    [
        "sum",
        "f16-1",
        "--sdt=f16:f16:f16 --stag=abx:abx:abx --scales=1.25:3:0.5 16x2x6x4x3",
        False,  # Do not run graph for this benchmark
    ],
    [
        "sum",
        "f16-2",
        "--reset --ddt=f16 \
            --sdt=f16:f16:f16:f16:f16:f16:f16:f16:f16:f16 \
            --stag=abx:aBx16b:ABx16a16b:ABcd16b16a:BAcd16a16b:BAcd16b16a:aBCd16b16c:aBCd16c16b:aCBd16b16c:aCBd16c16b \
            --dtag=abx,aBx16b,ABx16a16b,ABcd16b16a,BAcd16a16b,BAcd16b16a,aBCd16b16c,aBCd16c16b,aCBd16b16c,aCBd16c16b \
            --scales=1.25:3:0.5:2:0.5:2:0.5:2:0.5:2 \
            16x32x48x5",
        False,  # Do not run graph for this benchmark
    ],
    [
        "sum",
        "f32-1",
        "--sdt=bf16:bf16:bf16 --stag=abx:abx:abx --scales=0.5:2:0.5    16x2x6x4x3",
        False,  # Do not run graph for this benchmark
    ],
    [
        "sum",
        "f32-2",
        "--reset --inplace=true,false --ddt=bf16 \
            --sdt=bf16:bf16:bf16:bf16:bf16:bf16:bf16:bf16:bf16:bf16 \
            --stag=abx:aBx16b:ABx16a16b:ABcd16b16a:BAcd16a16b:BAcd16b16a:aBCd16b16c:aBCd16c16b:aCBd16b16c:aCBd16c16b \
            --dtag=abx,aBx16b,ABx16a16b,ABcd16b16a,BAcd16a16b,BAcd16b16a,aBCd16b16c,aBCd16c16b,aCBd16b16c,aCBd16c16b \
            --scales=0.25:0.15:0.25:0.25:0.25:0.25:0.15:0.25:0.25:0.15 \
            16x32x48x5",
        False,  # Do not run graph for this benchmark
    ],
    [
        "sum",
        "padding-1",
        "--ddt=f32 --sdt=f32:f32 --stag=aBcd16b --dtag=aBcd16b 1x8x64x64 1x8x640x1024 1x24x640x1024",
    ],
    [
        "sum",
        "padding-2",
        "--sdt=bf16:bf16 --ddt=bf16 --stag=AB48a16b:AB48a16b --dtag=AB48a16b 512x1024",
    ],
    [
        "sum",
        "key-gpu",
        "--batch=option_set_fwks_key_gpu",
    ],
    [
        "sum",
        "key-ext",
        "--batch=option_set_fwks_ext_gpu",
    ],
]
benches_sum_ci = [
    [
        "sum",
        "ci",
        "--reset --ddt=f32,s8 --sdt=f32:u8:s8 --stag=abx:abx:abx,axb:axb:axb --scales=0.25:2:0.5 2x17x5x7x3 4x16x8x10x2",
    ],
]
benches_graph_fusion = [
    [
        "graph",
        "JAX-MHA-inf-bf16",
        "--reset --dt=bf16 --case=complex_fusion/mha/JAX-MHA-inf-fp32.json",
        False,  # Do not run SYCL graph for this benchmark
    ],
    [
        "graph",
        "JAX-MHA-inf-f16",
        "--reset --dt=f16 --case=complex_fusion/mha/JAX-MHA-inf-fp32.json",
        False,  # Do not run SYCL graph for this benchmark
    ],
    [
        "graph",
        "MHA-GPT-inf-bf16",
        "--reset --dt=bf16 --case=complex_fusion/mha/MHA-GPT-inf-fp32-bs1.json",
        False,  # Do not run SYCL graph for this benchmark
    ],
    [
        "graph",
        "MHA-bert_large-inf-f16",
        "--reset --dt=f16 --case=complex_fusion/mha/MHA-bert_large-inf-fp32-bs1.json",
        False,  # Do not run SYCL graph for this benchmark
    ],
    [
        "graph",
        "MHA-distill_bert-inf-f32",
        "--reset --dt=f32 --case=complex_fusion/mha/MHA-distill_bert-inf-fp32-bs1.json",
        False,  # Do not run SYCL graph for this benchmark
    ],
    [
        "graph",
        "sdpa-plain-simplified-f16-bf16",
        "--reset --dt=bf16 --case=complex_fusion/mha/sdpa-plain-simplified-f16.json",
        False,
    ],
    [
        "graph",
        "sdpa-plain-simplified-f16-f16",
        "--reset --dt=f16 --case=complex_fusion/mha/sdpa-plain-simplified-f16.json",
        False,  # Do not run SYCL graph for this benchmark
    ],
    [
        "graph",
        "sdpa-plain-simplified-f16-multiply",
        "--reset --dt=f32 --op-kind=1:Multiply --case=complex_fusion/mha/sdpa-plain-simplified-f16.json",
    ],
    [
        "graph",
        "sdpa-plain-wo-scale-f16-bs1-f16",
        "--reset --dt=f16 --case=complex_fusion/mha/sdpa-plain-wo-scale-f16-bs1.json",
        False,  # Do not run SYCL graph for this benchmark
    ],
    [
        "graph",
        "GQA-fp16-f32",
        "--reset --dt=f32 --case=complex_fusion/mha/GQA-fp16.json",
    ],
    [
        "graph",
        "GQA-fp16-bf16",
        "--reset --dt=bf16 --case=complex_fusion/mha/GQA-fp16.json",
        False,  # Do not run SYCL graph for this benchmark
    ],
    [
        "graph",
        "GQA-fp16-f16",
        "--reset --dt=f16 --case=complex_fusion/mha/GQA-fp16.json",
        False,  # Do not run SYCL graph for this benchmark
    ],
    [
        "graph",
        "GQA-fp16-v2-f32",
        "--reset --dt=f32 --case=complex_fusion/mha/GQA-fp16-v2.json",
    ],
    [
        "graph",
        "GQA-fp16-v2-bf16",
        "--reset --dt=bf16 --case=complex_fusion/mha/GQA-fp16-v2.json",
        False,  # Do not run SYCL graph for this benchmark
    ],
    [
        "graph",
        "sdpa-plain-wo-mask-f16-f32",
        "--reset --dt=f32 --case=complex_fusion/mha/sdpa-plain-wo-mask-f16.json",
    ],
    [
        "graph",
        "sdpa-plain-mask-f32",
        "--reset --dt=f32 --case=complex_fusion/mha/sdpa-plain-implicit-causal-mask-fp32-bs1.json",
    ],
    # int8 graphs
    [
        "graph",
        "MHA-bert_large-inf-int8-bs1",
        "--reset --case=complex_fusion/mha/MHA-bert_large-inf-int8-bs1.json",
        False,
    ],
    [
        "graph",
        "sdpa-plain-wo-scale-int8-bs1",
        "--reset --case=complex_fusion/mha/sdpa-plain-wo-scale-int8-bs1.json",
        False,
    ],
    [
        "graph",
        "sdpa-compressed-v-int8-gs32",
        "--reset --case=complex_fusion/mha/sdpa-compressed-v-int8-gs32.json",
        False,
    ],
    [
        "graph",
        "sdpa-compressed-kv-int4-gs32",
        "--reset --case=complex_fusion/mha/sdpa-compressed-kv-int4-gs32.json",
        False,
    ],
]
benches_graph_plain = [
    [
        "graph",
        "sdpa-plain-f32",
        "--reset --dt=f32 --case=complex_fusion/mha/sdpa-plain-implicit-causal-mask-fp32-bs1.json",
    ],
    [
        "graph",
        "sdpa-plain-bf16",
        "--reset --dt=bf16 --case=complex_fusion/mha/sdpa-plain-implicit-causal-mask-fp32-bs1.json",
        False,  # Do not run SYCL graph for this benchmark
    ],
    [
        "graph",
        "sdpa-plain-f16",
        "--reset --dt=f16 --case=complex_fusion/mha/sdpa-plain-implicit-causal-mask-fp32-bs1.json",
    ],
]

# quick benchmarks for testing purposes, not intended for CI
bench_test = [
    [
        "sum",
        "ci",
        "--reset --ddt=f32,s8 --sdt=f32:u8:s8 --stag=abx:abx:abx,axb:axb:axb --scales=0.25:2:0.5 2x17x5x7x3 4x16x8x10x2",
    ],
    [
        "sum",
        "f16-1",
        "--sdt=f16:f16:f16 --stag=abx:abx:abx --scales=1.25:3:0.5 16x2x6x4x3",
        False,  # Do not run graph for this benchmark
    ],
    [
        "sum",
        "f32-1",
        "--sdt=bf16:bf16:bf16 --stag=abx:abx:abx --scales=0.5:2:0.5    16x2x6x4x3",
        False,  # Do not run graph for this benchmark
    ],
    [
        "graph",
        "sdpa-plain-f32",
        "--reset --dt=f32 --case=complex_fusion/mha/sdpa-plain-implicit-causal-mask-fp32-bs1.json",
        False,
    ],
]


def get_bench_dnn_list():
    bench_list = []
    bench_list.extend(benches_final_set)
    return bench_list
