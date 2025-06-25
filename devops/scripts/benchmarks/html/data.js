benchmarkRuns = [
  {
    "results": [
      {
        "label": "onednn-sum-f16-1-eager",
        "value": 0.00944,
        "command": [
          "/home/mateuszpn/workdir/onednn-build/tests/benchdnn/benchdnn",
          "--sum",
          "--mode=P",
          "--engine=gpu",
          "--max-ms-per-prb=100",
          "--execution-mode=direct",
          "--sdt=f16:f16:f16",
          "--stag=abx:abx:abx",
          "--scales=1.25:3:0.5",
          "16x2x6x4x3"
        ],
        "env": {
          "ONEAPI_DEVICE_SELECTOR": "level_zero:*"
        },
        "stdout": "Output template: perf,%engine%,%impl%,%name%,%prb%,%Gops%,%+ctime%,%-time%,%-Gflops%,%0time%,%0Gflops%\nperf,gpu,multi_po_reorder_sum,,--mode=P --max-ms-per-prb=100 --sum --engine=gpu --sdt=f16:f16:f16 --scales=1.25:3:0.5 16x2x6x4x3,0,2.52173,0.00944,0,0.0128609,0\ntests:1 passed:1 skipped:0 mistrusted:0 unimplemented:0 invalid_arguments:0 failed:0 listed:0\ntotal perf: min(ms):0.00944 avg(ms):0.0128609\ntotal: 0.27s; create_pd: 0.00s (0%); create_prim: 0.00s (1%); fill: 0.01s (3%); execute: 0.00s (0%);\n",
        "passed": true,
        "unit": "ms",
        "stddev": 0.0,
        "git_url": "https://github.com/uxlfoundation/oneDNN.git",
        "git_hash": "v3.8",
        "name": "onednn-sum-f16-1-eager",
        "lower_is_better": true,
        "suite": "BenchDNN"
      },
      {
        "label": "onednn-sum-f16-2-eager",
        "value": 0.60928,
        "command": [
          "/home/mateuszpn/workdir/onednn-build/tests/benchdnn/benchdnn",
          "--sum",
          "--mode=P",
          "--engine=gpu",
          "--max-ms-per-prb=100",
          "--execution-mode=direct",
          "--reset",
          "--ddt=f16",
          "--sdt=f16:f16:f16:f16:f16:f16:f16:f16:f16:f16",
          "--stag=abx:aBx16b:ABx16a16b:ABcd16b16a:BAcd16a16b:BAcd16b16a:aBCd16b16c:aBCd16c16b:aCBd16b16c:aCBd16c16b",
          "--dtag=abx,aBx16b,ABx16a16b,ABcd16b16a,BAcd16a16b,BAcd16b16a,aBCd16b16c,aBCd16c16b,aCBd16b16c,aCBd16c16b",
          "--scales=1.25:3:0.5:2:0.5:2:0.5:2:0.5:2",
          "16x32x48x5"
        ],
        "env": {
          "ONEAPI_DEVICE_SELECTOR": "level_zero:*"
        },
        "stdout": "Output template: perf,%engine%,%impl%,%name%,%prb%,%Gops%,%+ctime%,%-time%,%-Gflops%,%0time%,%0Gflops%\nperf,gpu,multi_po_reorder_sum,,--mode=P --max-ms-per-prb=100 --sum --engine=gpu --sdt=f16:f16:f16:f16:f16:f16:f16:f16:f16:f16 --ddt=f16 --stag=abx:aBx16b:ABx16a16b:ABcd16b16a:BAcd16a16b:BAcd16b16a:aBCd16b16c:aBCd16c16b:aCBd16b16c:aCBd16c16b --dtag=abx --scales=1.25:3:0.5:2:0.5:2:0.5:2:0.5:2 16x32x48x5,0,48.9631,0.06448,0,0.0676806,0\nperf,gpu,multi_po_reorder_sum,,--mode=P --max-ms-per-prb=100 --sum --engine=gpu --sdt=f16:f16:f16:f16:f16:f16:f16:f16:f16:f16 --ddt=f16 --stag=abx:aBx16b:ABx16a16b:ABcd16b16a:BAcd16a16b:BAcd16b16a:aBCd16b16c:aBCd16c16b:aCBd16b16c:aCBd16c16b --dtag=aBx16b --scales=1.25:3:0.5:2:0.5:2:0.5:2:0.5:2 16x32x48x5,0,30.5063,0.05808,0,0.0612839,0\nperf,gpu,multi_po_reorder_sum,,--mode=P --max-ms-per-prb=100 --sum --engine=gpu --sdt=f16:f16:f16:f16:f16:f16:f16:f16:f16:f16 --ddt=f16 --stag=abx:aBx16b:ABx16a16b:ABcd16b16a:BAcd16a16b:BAcd16b16a:aBCd16b16c:aBCd16c16b:aCBd16b16c:aCBd16c16b --dtag=ABx16a16b --scales=1.25:3:0.5:2:0.5:2:0.5:2:0.5:2 16x32x48x5,0,47.4368,0.05888,0,0.0620269,0\nperf,gpu,multi_po_reorder_sum,,--mode=P --max-ms-per-prb=100 --sum --engine=gpu --sdt=f16:f16:f16:f16:f16:f16:f16:f16:f16:f16 --ddt=f16 --stag=abx:aBx16b:ABx16a16b:ABcd16b16a:BAcd16a16b:BAcd16b16a:aBCd16b16c:aBCd16c16b:aCBd16b16c:aCBd16c16b --dtag=ABcd16b16a --scales=1.25:3:0.5:2:0.5:2:0.5:2:0.5:2 16x32x48x5,0,46.4478,0.06368,0,0.0671496,0\nperf,gpu,multi_po_reorder_sum,,--mode=P --max-ms-per-prb=100 --sum --engine=gpu --sdt=f16:f16:f16:f16:f16:f16:f16:f16:f16:f16 --ddt=f16 --stag=abx:aBx16b:ABx16a16b:ABcd16b16a:BAcd16a16b:BAcd16b16a:aBCd16b16c:aBCd16c16b:aCBd16b16c:aCBd16c16b --dtag=BAcd16a16b --scales=1.25:3:0.5:2:0.5:2:0.5:2:0.5:2 16x32x48x5,0,46.657,0.05984,0,0.0630586,0\nperf,gpu,multi_po_reorder_sum,,--mode=P --max-ms-per-prb=100 --sum --engine=gpu --sdt=f16:f16:f16:f16:f16:f16:f16:f16:f16:f16 --ddt=f16 --stag=abx:aBx16b:ABx16a16b:ABcd16b16a:BAcd16a16b:BAcd16b16a:aBCd16b16c:aBCd16c16b:aCBd16b16c:aCBd16c16b --dtag=BAcd16b16a --scales=1.25:3:0.5:2:0.5:2:0.5:2:0.5:2 16x32x48x5,0,45.9631,0.06448,0,0.0679256,0\nperf,gpu,multi_po_reorder_sum,,--mode=P --max-ms-per-prb=100 --sum --engine=gpu --sdt=f16:f16:f16:f16:f16:f16:f16:f16:f16:f16 --ddt=f16 --stag=abx:aBx16b:ABx16a16b:ABcd16b16a:BAcd16a16b:BAcd16b16a:aBCd16b16c:aBCd16c16b:aCBd16b16c:aCBd16c16b --dtag=aBCd16b16c --scales=1.25:3:0.5:2:0.5:2:0.5:2:0.5:2 16x32x48x5,0,41.9988,0.06208,0,0.0652478,0\nperf,gpu,multi_po_reorder_sum,,--mode=P --max-ms-per-prb=100 --sum --engine=gpu --sdt=f16:f16:f16:f16:f16:f16:f16:f16:f16:f16 --ddt=f16 --stag=abx:aBx16b:ABx16a16b:ABcd16b16a:BAcd16a16b:BAcd16b16a:aBCd16b16c:aBCd16c16b:aCBd16b16c:aCBd16c16b --dtag=aBCd16c16b --scales=1.25:3:0.5:2:0.5:2:0.5:2:0.5:2 16x32x48x5,0,47.5825,0.05808,0,0.061508,0\nperf,gpu,multi_po_reorder_sum,,--mode=P --max-ms-per-prb=100 --sum --engine=gpu --sdt=f16:f16:f16:f16:f16:f16:f16:f16:f16:f16 --ddt=f16 --stag=abx:aBx16b:ABx16a16b:ABcd16b16a:BAcd16a16b:BAcd16b16a:aBCd16b16c:aBCd16c16b:aCBd16b16c:aCBd16c16b --dtag=aCBd16b16c --scales=1.25:3:0.5:2:0.5:2:0.5:2:0.5:2 16x32x48x5,0,51.822,0.06288,0,0.0659863,0\nperf,gpu,multi_po_reorder_sum,,--mode=P --max-ms-per-prb=100 --sum --engine=gpu --sdt=f16:f16:f16:f16:f16:f16:f16:f16:f16:f16 --ddt=f16 --stag=abx:aBx16b:ABx16a16b:ABcd16b16a:BAcd16a16b:BAcd16b16a:aBCd16b16c:aBCd16c16b:aCBd16b16c:aCBd16c16b --dtag=aCBd16c16b --scales=1.25:3:0.5:2:0.5:2:0.5:2:0.5:2 16x32x48x5,0,50.4551,0.0568,0,0.0609149,0\ntests:10 passed:10 skipped:0 mistrusted:0 unimplemented:0 invalid_arguments:0 failed:0 listed:0\ntotal perf: min(ms):0.60928 avg(ms):0.642782\ntotal: 2.47s; create_pd: 0.01s (0%); create_prim: 0.45s (18%); fill: 0.08s (3%); execute: 0.02s (1%);\n",
        "passed": true,
        "unit": "ms",
        "stddev": 0.0021996363335788104,
        "git_url": "https://github.com/uxlfoundation/oneDNN.git",
        "git_hash": "v3.8",
        "name": "onednn-sum-f16-2-eager",
        "lower_is_better": true,
        "suite": "BenchDNN"
      },
      {
        "label": "onednn-sum-f32-1-eager",
        "value": 0.0088,
        "command": [
          "/home/mateuszpn/workdir/onednn-build/tests/benchdnn/benchdnn",
          "--sum",
          "--mode=P",
          "--engine=gpu",
          "--max-ms-per-prb=100",
          "--execution-mode=direct",
          "--sdt=bf16:bf16:bf16",
          "--stag=abx:abx:abx",
          "--scales=0.5:2:0.5",
          "16x2x6x4x3"
        ],
        "env": {
          "ONEAPI_DEVICE_SELECTOR": "level_zero:*"
        },
        "stdout": "Output template: perf,%engine%,%impl%,%name%,%prb%,%Gops%,%+ctime%,%-time%,%-Gflops%,%0time%,%0Gflops%\nperf,gpu,multi_po_reorder_sum,,--mode=P --max-ms-per-prb=100 --sum --engine=gpu --sdt=bf16:bf16:bf16 --scales=0.5:2:0.5 16x2x6x4x3,0,2.42236,0.0088,0,0.0129955,0\ntests:1 passed:1 skipped:0 mistrusted:0 unimplemented:0 invalid_arguments:0 failed:0 listed:0\ntotal perf: min(ms):0.0088 avg(ms):0.0129955\ntotal: 0.28s; create_pd: 0.00s (0%); create_prim: 0.00s (1%); fill: 0.01s (3%); execute: 0.00s (0%);\n",
        "passed": true,
        "unit": "ms",
        "stddev": 0.00017486502731471965,
        "git_url": "https://github.com/uxlfoundation/oneDNN.git",
        "git_hash": "v3.8",
        "name": "onednn-sum-f32-1-eager",
        "lower_is_better": true,
        "suite": "BenchDNN"
      },
      {
        "label": "onednn-sum-f32-2-eager",
        "value": 0.6441600000000001,
        "command": [
          "/home/mateuszpn/workdir/onednn-build/tests/benchdnn/benchdnn",
          "--sum",
          "--mode=P",
          "--engine=gpu",
          "--max-ms-per-prb=100",
          "--execution-mode=direct",
          "--reset",
          "--inplace=true,false",
          "--ddt=bf16",
          "--sdt=bf16:bf16:bf16:bf16:bf16:bf16:bf16:bf16:bf16:bf16",
          "--stag=abx:aBx16b:ABx16a16b:ABcd16b16a:BAcd16a16b:BAcd16b16a:aBCd16b16c:aBCd16c16b:aCBd16b16c:aCBd16c16b",
          "--dtag=abx,aBx16b,ABx16a16b,ABcd16b16a,BAcd16a16b,BAcd16b16a,aBCd16b16c,aBCd16c16b,aCBd16b16c,aCBd16c16b",
          "--scales=0.25:0.15:0.25:0.25:0.25:0.25:0.15:0.25:0.25:0.15",
          "16x32x48x5"
        ],
        "env": {
          "ONEAPI_DEVICE_SELECTOR": "level_zero:*"
        },
        "stdout": "Output template: perf,%engine%,%impl%,%name%,%prb%,%Gops%,%+ctime%,%-time%,%-Gflops%,%0time%,%0Gflops%\nperf,gpu,multi_po_reorder_sum,,--mode=P --max-ms-per-prb=100 --sum --engine=gpu --sdt=bf16:bf16:bf16:bf16:bf16:bf16:bf16:bf16:bf16:bf16 --ddt=bf16 --stag=abx:aBx16b:ABx16a16b:ABcd16b16a:BAcd16a16b:BAcd16b16a:aBCd16b16c:aBCd16c16b:aCBd16b16c:aCBd16c16b --dtag=abx --scales=0.25:0.15:0.25:0.25:0.25:0.25:0.15:0.25:0.25:0.15 --inplace=true 16x32x48x5,0,47.5034,0.0648,0,0.0682694,0\nperf,gpu,multi_po_reorder_sum,,--mode=P --max-ms-per-prb=100 --sum --engine=gpu --sdt=bf16:bf16:bf16:bf16:bf16:bf16:bf16:bf16:bf16:bf16 --ddt=bf16 --stag=abx:aBx16b:ABx16a16b:ABcd16b16a:BAcd16a16b:BAcd16b16a:aBCd16b16c:aBCd16c16b:aCBd16b16c:aCBd16c16b --dtag=abx --scales=0.25:0.15:0.25:0.25:0.25:0.25:0.15:0.25:0.25:0.15 16x32x48x5,0,0.0119629,0.06512,0,0.068156,0\n2:SKIPPED (Invalid case) (0 ms) __REPRO: --mode=P --max-ms-per-prb=100 --sum --engine=gpu --sdt=bf16:bf16:bf16:bf16:bf16:bf16:bf16:bf16:bf16:bf16 --ddt=bf16 --stag=abx:aBx16b:ABx16a16b:ABcd16b16a:BAcd16a16b:BAcd16b16a:aBCd16b16c:aBCd16c16b:aCBd16b16c:aCBd16c16b --dtag=aBx16b --scales=0.25:0.15:0.25:0.25:0.25:0.25:0.15:0.25:0.25:0.15 --inplace=true 16x32x48x5\nperf,gpu,,,--mode=P --max-ms-per-prb=100 --sum --engine=gpu --sdt=bf16:bf16:bf16:bf16:bf16:bf16:bf16:bf16:bf16:bf16 --ddt=bf16 --stag=abx:aBx16b:ABx16a16b:ABcd16b16a:BAcd16a16b:BAcd16b16a:aBCd16b16c:aBCd16c16b:aCBd16b16c:aCBd16c16b --dtag=aBx16b --scales=0.25:0.15:0.25:0.25:0.25:0.25:0.15:0.25:0.25:0.15 --inplace=true 16x32x48x5,0,0,0,0,0,0\nperf,gpu,multi_po_reorder_sum,,--mode=P --max-ms-per-prb=100 --sum --engine=gpu --sdt=bf16:bf16:bf16:bf16:bf16:bf16:bf16:bf16:bf16:bf16 --ddt=bf16 --stag=abx:aBx16b:ABx16a16b:ABcd16b16a:BAcd16a16b:BAcd16b16a:aBCd16b16c:aBCd16c16b:aCBd16b16c:aCBd16c16b --dtag=aBx16b --scales=0.25:0.15:0.25:0.25:0.25:0.25:0.15:0.25:0.25:0.15 16x32x48x5,0,29.7659,0.0568,0,0.060283,0\n4:SKIPPED (Invalid case) (0 ms) __REPRO: --mode=P --max-ms-per-prb=100 --sum --engine=gpu --sdt=bf16:bf16:bf16:bf16:bf16:bf16:bf16:bf16:bf16:bf16 --ddt=bf16 --stag=abx:aBx16b:ABx16a16b:ABcd16b16a:BAcd16a16b:BAcd16b16a:aBCd16b16c:aBCd16c16b:aCBd16b16c:aCBd16c16b --dtag=ABx16a16b --scales=0.25:0.15:0.25:0.25:0.25:0.25:0.15:0.25:0.25:0.15 --inplace=true 16x32x48x5\nperf,gpu,,,--mode=P --max-ms-per-prb=100 --sum --engine=gpu --sdt=bf16:bf16:bf16:bf16:bf16:bf16:bf16:bf16:bf16:bf16 --ddt=bf16 --stag=abx:aBx16b:ABx16a16b:ABcd16b16a:BAcd16a16b:BAcd16b16a:aBCd16b16c:aBCd16c16b:aCBd16b16c:aCBd16c16b --dtag=ABx16a16b --scales=0.25:0.15:0.25:0.25:0.25:0.25:0.15:0.25:0.25:0.15 --inplace=true 16x32x48x5,0,0,0,0,0,0\nperf,gpu,multi_po_reorder_sum,,--mode=P --max-ms-per-prb=100 --sum --engine=gpu --sdt=bf16:bf16:bf16:bf16:bf16:bf16:bf16:bf16:bf16:bf16 --ddt=bf16 --stag=abx:aBx16b:ABx16a16b:ABcd16b16a:BAcd16a16b:BAcd16b16a:aBCd16b16c:aBCd16c16b:aCBd16b16c:aCBd16c16b --dtag=ABx16a16b --scales=0.25:0.15:0.25:0.25:0.25:0.25:0.15:0.25:0.25:0.15 16x32x48x5,0,46.333,0.0568,0,0.0603173,0\n6:SKIPPED (Invalid case) (0 ms) __REPRO: --mode=P --max-ms-per-prb=100 --sum --engine=gpu --sdt=bf16:bf16:bf16:bf16:bf16:bf16:bf16:bf16:bf16:bf16 --ddt=bf16 --stag=abx:aBx16b:ABx16a16b:ABcd16b16a:BAcd16a16b:BAcd16b16a:aBCd16b16c:aBCd16c16b:aCBd16b16c:aCBd16c16b --dtag=ABcd16b16a --scales=0.25:0.15:0.25:0.25:0.25:0.25:0.15:0.25:0.25:0.15 --inplace=true 16x32x48x5\nperf,gpu,,,--mode=P --max-ms-per-prb=100 --sum --engine=gpu --sdt=bf16:bf16:bf16:bf16:bf16:bf16:bf16:bf16:bf16:bf16 --ddt=bf16 --stag=abx:aBx16b:ABx16a16b:ABcd16b16a:BAcd16a16b:BAcd16b16a:aBCd16b16c:aBCd16c16b:aCBd16b16c:aCBd16c16b --dtag=ABcd16b16a --scales=0.25:0.15:0.25:0.25:0.25:0.25:0.15:0.25:0.25:0.15 --inplace=true 16x32x48x5,0,0,0,0,0,0\nperf,gpu,multi_po_reorder_sum,,--mode=P --max-ms-per-prb=100 --sum --engine=gpu --sdt=bf16:bf16:bf16:bf16:bf16:bf16:bf16:bf16:bf16:bf16 --ddt=bf16 --stag=abx:aBx16b:ABx16a16b:ABcd16b16a:BAcd16a16b:BAcd16b16a:aBCd16b16c:aBCd16c16b:aCBd16b16c:aCBd16c16b --dtag=ABcd16b16a --scales=0.25:0.15:0.25:0.25:0.25:0.25:0.15:0.25:0.25:0.15 16x32x48x5,0,44.2969,0.05808,0,0.0609991,0\n8:SKIPPED (Invalid case) (0 ms) __REPRO: --mode=P --max-ms-per-prb=100 --sum --engine=gpu --sdt=bf16:bf16:bf16:bf16:bf16:bf16:bf16:bf16:bf16:bf16 --ddt=bf16 --stag=abx:aBx16b:ABx16a16b:ABcd16b16a:BAcd16a16b:BAcd16b16a:aBCd16b16c:aBCd16c16b:aCBd16b16c:aCBd16c16b --dtag=BAcd16a16b --scales=0.25:0.15:0.25:0.25:0.25:0.25:0.15:0.25:0.25:0.15 --inplace=true 16x32x48x5\nperf,gpu,,,--mode=P --max-ms-per-prb=100 --sum --engine=gpu --sdt=bf16:bf16:bf16:bf16:bf16:bf16:bf16:bf16:bf16:bf16 --ddt=bf16 --stag=abx:aBx16b:ABx16a16b:ABcd16b16a:BAcd16a16b:BAcd16b16a:aBCd16b16c:aBCd16c16b:aCBd16b16c:aCBd16c16b --dtag=BAcd16a16b --scales=0.25:0.15:0.25:0.25:0.25:0.25:0.15:0.25:0.25:0.15 --inplace=true 16x32x48x5,0,0,0,0,0,0\nperf,gpu,multi_po_reorder_sum,,--mode=P --max-ms-per-prb=100 --sum --engine=gpu --sdt=bf16:bf16:bf16:bf16:bf16:bf16:bf16:bf16:bf16:bf16 --ddt=bf16 --stag=abx:aBx16b:ABx16a16b:ABcd16b16a:BAcd16a16b:BAcd16b16a:aBCd16b16c:aBCd16c16b:aCBd16b16c:aCBd16c16b --dtag=BAcd16a16b --scales=0.25:0.15:0.25:0.25:0.25:0.25:0.15:0.25:0.25:0.15 16x32x48x5,0,44.946,0.05248,0,0.055674,0\n10:SKIPPED (Invalid case) (0 ms) __REPRO: --mode=P --max-ms-per-prb=100 --sum --engine=gpu --sdt=bf16:bf16:bf16:bf16:bf16:bf16:bf16:bf16:bf16:bf16 --ddt=bf16 --stag=abx:aBx16b:ABx16a16b:ABcd16b16a:BAcd16a16b:BAcd16b16a:aBCd16b16c:aBCd16c16b:aCBd16b16c:aCBd16c16b --dtag=BAcd16b16a --scales=0.25:0.15:0.25:0.25:0.25:0.25:0.15:0.25:0.25:0.15 --inplace=true 16x32x48x5\nperf,gpu,,,--mode=P --max-ms-per-prb=100 --sum --engine=gpu --sdt=bf16:bf16:bf16:bf16:bf16:bf16:bf16:bf16:bf16:bf16 --ddt=bf16 --stag=abx:aBx16b:ABx16a16b:ABcd16b16a:BAcd16a16b:BAcd16b16a:aBCd16b16c:aBCd16c16b:aCBd16b16c:aCBd16c16b --dtag=BAcd16b16a --scales=0.25:0.15:0.25:0.25:0.25:0.25:0.15:0.25:0.25:0.15 --inplace=true 16x32x48x5,0,0,0,0,0,0\nperf,gpu,multi_po_reorder_sum,,--mode=P --max-ms-per-prb=100 --sum --engine=gpu --sdt=bf16:bf16:bf16:bf16:bf16:bf16:bf16:bf16:bf16:bf16 --ddt=bf16 --stag=abx:aBx16b:ABx16a16b:ABcd16b16a:BAcd16a16b:BAcd16b16a:aBCd16b16c:aBCd16c16b:aCBd16b16c:aCBd16c16b --dtag=BAcd16b16a --scales=0.25:0.15:0.25:0.25:0.25:0.25:0.15:0.25:0.25:0.15 16x32x48x5,0,44.5508,0.05904,0,0.0620182,0\n12:SKIPPED (Invalid case) (0 ms) __REPRO: --mode=P --max-ms-per-prb=100 --sum --engine=gpu --sdt=bf16:bf16:bf16:bf16:bf16:bf16:bf16:bf16:bf16:bf16 --ddt=bf16 --stag=abx:aBx16b:ABx16a16b:ABcd16b16a:BAcd16a16b:BAcd16b16a:aBCd16b16c:aBCd16c16b:aCBd16b16c:aCBd16c16b --dtag=aBCd16b16c --scales=0.25:0.15:0.25:0.25:0.25:0.25:0.15:0.25:0.25:0.15 --inplace=true 16x32x48x5\nperf,gpu,,,--mode=P --max-ms-per-prb=100 --sum --engine=gpu --sdt=bf16:bf16:bf16:bf16:bf16:bf16:bf16:bf16:bf16:bf16 --ddt=bf16 --stag=abx:aBx16b:ABx16a16b:ABcd16b16a:BAcd16a16b:BAcd16b16a:aBCd16b16c:aBCd16c16b:aCBd16b16c:aCBd16c16b --dtag=aBCd16b16c --scales=0.25:0.15:0.25:0.25:0.25:0.25:0.15:0.25:0.25:0.15 --inplace=true 16x32x48x5,0,0,0,0,0,0\nperf,gpu,multi_po_reorder_sum,,--mode=P --max-ms-per-prb=100 --sum --engine=gpu --sdt=bf16:bf16:bf16:bf16:bf16:bf16:bf16:bf16:bf16:bf16 --ddt=bf16 --stag=abx:aBx16b:ABx16a16b:ABcd16b16a:BAcd16a16b:BAcd16b16a:aBCd16b16c:aBCd16c16b:aCBd16b16c:aCBd16c16b --dtag=aBCd16b16c --scales=0.25:0.15:0.25:0.25:0.25:0.25:0.15:0.25:0.25:0.15 16x32x48x5,0,41.147,0.06112,0,0.0643661,0\n14:SKIPPED (Invalid case) (0 ms) __REPRO: --mode=P --max-ms-per-prb=100 --sum --engine=gpu --sdt=bf16:bf16:bf16:bf16:bf16:bf16:bf16:bf16:bf16:bf16 --ddt=bf16 --stag=abx:aBx16b:ABx16a16b:ABcd16b16a:BAcd16a16b:BAcd16b16a:aBCd16b16c:aBCd16c16b:aCBd16b16c:aCBd16c16b --dtag=aBCd16c16b --scales=0.25:0.15:0.25:0.25:0.25:0.25:0.15:0.25:0.25:0.15 --inplace=true 16x32x48x5\nperf,gpu,,,--mode=P --max-ms-per-prb=100 --sum --engine=gpu --sdt=bf16:bf16:bf16:bf16:bf16:bf16:bf16:bf16:bf16:bf16 --ddt=bf16 --stag=abx:aBx16b:ABx16a16b:ABcd16b16a:BAcd16a16b:BAcd16b16a:aBCd16b16c:aBCd16c16b:aCBd16b16c:aCBd16c16b --dtag=aBCd16c16b --scales=0.25:0.15:0.25:0.25:0.25:0.25:0.15:0.25:0.25:0.15 --inplace=true 16x32x48x5,0,0,0,0,0,0\nperf,gpu,multi_po_reorder_sum,,--mode=P --max-ms-per-prb=100 --sum --engine=gpu --sdt=bf16:bf16:bf16:bf16:bf16:bf16:bf16:bf16:bf16:bf16 --ddt=bf16 --stag=abx:aBx16b:ABx16a16b:ABcd16b16a:BAcd16a16b:BAcd16b16a:aBCd16b16c:aBCd16c16b:aCBd16b16c:aCBd16c16b --dtag=aBCd16c16b --scales=0.25:0.15:0.25:0.25:0.25:0.25:0.15:0.25:0.25:0.15 16x32x48x5,0,46.8096,0.05728,0,0.0602824,0\n16:SKIPPED (Invalid case) (0 ms) __REPRO: --mode=P --max-ms-per-prb=100 --sum --engine=gpu --sdt=bf16:bf16:bf16:bf16:bf16:bf16:bf16:bf16:bf16:bf16 --ddt=bf16 --stag=abx:aBx16b:ABx16a16b:ABcd16b16a:BAcd16a16b:BAcd16b16a:aBCd16b16c:aBCd16c16b:aCBd16b16c:aCBd16c16b --dtag=aCBd16b16c --scales=0.25:0.15:0.25:0.25:0.25:0.25:0.15:0.25:0.25:0.15 --inplace=true 16x32x48x5\nperf,gpu,,,--mode=P --max-ms-per-prb=100 --sum --engine=gpu --sdt=bf16:bf16:bf16:bf16:bf16:bf16:bf16:bf16:bf16:bf16 --ddt=bf16 --stag=abx:aBx16b:ABx16a16b:ABcd16b16a:BAcd16a16b:BAcd16b16a:aBCd16b16c:aBCd16c16b:aCBd16b16c:aCBd16c16b --dtag=aCBd16b16c --scales=0.25:0.15:0.25:0.25:0.25:0.25:0.15:0.25:0.25:0.15 --inplace=true 16x32x48x5,0,0,0,0,0,0\nperf,gpu,multi_po_reorder_sum,,--mode=P --max-ms-per-prb=100 --sum --engine=gpu --sdt=bf16:bf16:bf16:bf16:bf16:bf16:bf16:bf16:bf16:bf16 --ddt=bf16 --stag=abx:aBx16b:ABx16a16b:ABcd16b16a:BAcd16a16b:BAcd16b16a:aBCd16b16c:aBCd16c16b:aCBd16b16c:aCBd16c16b --dtag=aCBd16b16c --scales=0.25:0.15:0.25:0.25:0.25:0.25:0.15:0.25:0.25:0.15 16x32x48x5,0,50.3113,0.05664,0,0.0600053,0\n18:SKIPPED (Invalid case) (0 ms) __REPRO: --mode=P --max-ms-per-prb=100 --sum --engine=gpu --sdt=bf16:bf16:bf16:bf16:bf16:bf16:bf16:bf16:bf16:bf16 --ddt=bf16 --stag=abx:aBx16b:ABx16a16b:ABcd16b16a:BAcd16a16b:BAcd16b16a:aBCd16b16c:aBCd16c16b:aCBd16b16c:aCBd16c16b --dtag=aCBd16c16b --scales=0.25:0.15:0.25:0.25:0.25:0.25:0.15:0.25:0.25:0.15 --inplace=true 16x32x48x5\nperf,gpu,,,--mode=P --max-ms-per-prb=100 --sum --engine=gpu --sdt=bf16:bf16:bf16:bf16:bf16:bf16:bf16:bf16:bf16:bf16 --ddt=bf16 --stag=abx:aBx16b:ABx16a16b:ABcd16b16a:BAcd16a16b:BAcd16b16a:aBCd16b16c:aBCd16c16b:aCBd16b16c:aCBd16c16b --dtag=aCBd16c16b --scales=0.25:0.15:0.25:0.25:0.25:0.25:0.15:0.25:0.25:0.15 --inplace=true 16x32x48x5,0,0,0,0,0,0\nperf,gpu,multi_po_reorder_sum,,--mode=P --max-ms-per-prb=100 --sum --engine=gpu --sdt=bf16:bf16:bf16:bf16:bf16:bf16:bf16:bf16:bf16:bf16 --ddt=bf16 --stag=abx:aBx16b:ABx16a16b:ABcd16b16a:BAcd16a16b:BAcd16b16a:aBCd16b16c:aBCd16c16b:aCBd16b16c:aCBd16c16b --dtag=aCBd16c16b --scales=0.25:0.15:0.25:0.25:0.25:0.25:0.15:0.25:0.25:0.15 16x32x48x5,0,49.8357,0.056,0,0.0593853,0\ntests:20 passed:11 skipped:9 mistrusted:0 unimplemented:0 invalid_arguments:0 failed:0 listed:0\ntotal perf: min(ms):0.64416 avg(ms):0.679756\ntotal: 2.68s; create_pd: 0.01s (0%); create_prim: 0.44s (16%); fill: 0.08s (3%); execute: 0.02s (1%);\n",
        "passed": true,
        "unit": "ms",
        "stddev": 0.004735567547823622,
        "git_url": "https://github.com/uxlfoundation/oneDNN.git",
        "git_hash": "v3.8",
        "name": "onednn-sum-f32-2-eager",
        "lower_is_better": true,
        "suite": "BenchDNN"
      },
      {
        "label": "onednn-sum-padding-1-eager",
        "value": 0.3904,
        "command": [
          "/home/mateuszpn/workdir/onednn-build/tests/benchdnn/benchdnn",
          "--sum",
          "--mode=P",
          "--engine=gpu",
          "--max-ms-per-prb=100",
          "--execution-mode=direct",
          "--ddt=f32",
          "--sdt=f32:f32",
          "--stag=aBcd16b",
          "--dtag=aBcd16b",
          "1x8x64x64",
          "1x8x640x1024",
          "1x24x640x1024"
        ],
        "env": {
          "ONEAPI_DEVICE_SELECTOR": "level_zero:*"
        },
        "stdout": "Output template: perf,%engine%,%impl%,%name%,%prb%,%Gops%,%+ctime%,%-time%,%-Gflops%,%0time%,%0Gflops%\nperf,gpu,multi_po_reorder_sum,,--mode=P --max-ms-per-prb=100 --sum --engine=gpu --stag=aBcd16b:aBcd16b --dtag=aBcd16b --scales=1 1x8x64x64,0,1.58545,0.00192,0,0.00269551,0\nperf,gpu,multi_po_reorder_sum,,--mode=P --max-ms-per-prb=100 --sum --engine=gpu --stag=aBcd16b:aBcd16b --dtag=aBcd16b --scales=1 1x8x640x1024,0,0.890869,0.08528,0,0.0932233,0\nperf,gpu,multi_po_reorder_sum,,--mode=P --max-ms-per-prb=100 --sum --engine=gpu --stag=aBcd16b:aBcd16b --dtag=aBcd16b --scales=1 1x24x640x1024,0,1.29517,0.3032,0,0.32437,0\ntests:3 passed:3 skipped:0 mistrusted:0 unimplemented:0 invalid_arguments:0 failed:0 listed:0\ntotal perf: min(ms):0.3904 avg(ms):0.420289\ntotal: 0.91s; create_pd: 0.00s (0%); create_prim: 0.00s (0%); fill: 0.15s (16%); execute: 0.02s (2%);\n",
        "passed": true,
        "unit": "ms",
        "stddev": 0.001602664448140469,
        "git_url": "https://github.com/uxlfoundation/oneDNN.git",
        "git_hash": "v3.8",
        "name": "onednn-sum-padding-1-eager",
        "lower_is_better": true,
        "suite": "BenchDNN"
      },
      {
        "label": "onednn-sum-padding-1-graph",
        "value": 0.39216,
        "command": [
          "/home/mateuszpn/workdir/onednn-build/tests/benchdnn/benchdnn",
          "--sum",
          "--mode=P",
          "--engine=gpu",
          "--max-ms-per-prb=100",
          "--execution-mode=graph",
          "--ddt=f32",
          "--sdt=f32:f32",
          "--stag=aBcd16b",
          "--dtag=aBcd16b",
          "1x8x64x64",
          "1x8x640x1024",
          "1x24x640x1024"
        ],
        "env": {
          "ONEAPI_DEVICE_SELECTOR": "level_zero:*"
        },
        "stdout": "Output template: perf,%engine%,%impl%,%name%,%prb%,%Gops%,%+ctime%,%-time%,%-Gflops%,%0time%,%0Gflops%\nperf,gpu,multi_po_reorder_sum,,--mode=P --max-ms-per-prb=100 --sum --engine=gpu --execution-mode=graph --stag=aBcd16b:aBcd16b --dtag=aBcd16b --scales=1 1x8x64x64,0,1.43994,0.00192,0,0.00268973,0\nperf,gpu,multi_po_reorder_sum,,--mode=P --max-ms-per-prb=100 --sum --engine=gpu --execution-mode=graph --stag=aBcd16b:aBcd16b --dtag=aBcd16b --scales=1 1x8x640x1024,0,0.874268,0.08656,0,0.094599,0\nperf,gpu,multi_po_reorder_sum,,--mode=P --max-ms-per-prb=100 --sum --engine=gpu --execution-mode=graph --stag=aBcd16b:aBcd16b --dtag=aBcd16b --scales=1 1x24x640x1024,0,1.27124,0.30368,0,0.325998,0\ntests:3 passed:3 skipped:0 mistrusted:0 unimplemented:0 invalid_arguments:0 failed:0 listed:0\ntotal perf: min(ms):0.39216 avg(ms):0.423287\ntotal: 0.89s; create_pd: 0.00s (0%); create_prim: 0.00s (0%); fill: 0.13s (15%); execute: 0.02s (2%);\n",
        "passed": true,
        "unit": "ms",
        "stddev": 0.0009097985124923661,
        "git_url": "https://github.com/uxlfoundation/oneDNN.git",
        "git_hash": "v3.8",
        "name": "onednn-sum-padding-1-graph",
        "lower_is_better": true,
        "suite": "BenchDNN"
      },
      {
        "label": "onednn-sum-padding-2-eager",
        "value": 0.00336,
        "command": [
          "/home/mateuszpn/workdir/onednn-build/tests/benchdnn/benchdnn",
          "--sum",
          "--mode=P",
          "--engine=gpu",
          "--max-ms-per-prb=100",
          "--execution-mode=direct",
          "--sdt=bf16:bf16",
          "--ddt=bf16",
          "--stag=AB48a16b:AB48a16b",
          "--dtag=AB48a16b",
          "512x1024"
        ],
        "env": {
          "ONEAPI_DEVICE_SELECTOR": "level_zero:*"
        },
        "stdout": "Output template: perf,%engine%,%impl%,%name%,%prb%,%Gops%,%+ctime%,%-time%,%-Gflops%,%0time%,%0Gflops%\nperf,gpu,multi_po_reorder_sum,,--mode=P --max-ms-per-prb=100 --sum --engine=gpu --sdt=bf16:bf16 --ddt=bf16 --stag=AB48a16b:AB48a16b --dtag=AB48a16b --scales=1 512x1024,0,1.21216,0.00336,0,0.00399908,0\ntests:1 passed:1 skipped:0 mistrusted:0 unimplemented:0 invalid_arguments:0 failed:0 listed:0\ntotal perf: min(ms):0.00336 avg(ms):0.00399908\ntotal: 0.33s; create_pd: 0.00s (0%); create_prim: 0.00s (0%); fill: 0.02s (6%); execute: 0.00s (0%);\n",
        "passed": true,
        "unit": "ms",
        "stddev": 8.262364471909155e-05,
        "git_url": "https://github.com/uxlfoundation/oneDNN.git",
        "git_hash": "v3.8",
        "name": "onednn-sum-padding-2-eager",
        "lower_is_better": true,
        "suite": "BenchDNN"
      },
      {
        "label": "onednn-sum-padding-2-graph",
        "value": 0.00352,
        "command": [
          "/home/mateuszpn/workdir/onednn-build/tests/benchdnn/benchdnn",
          "--sum",
          "--mode=P",
          "--engine=gpu",
          "--max-ms-per-prb=100",
          "--execution-mode=graph",
          "--sdt=bf16:bf16",
          "--ddt=bf16",
          "--stag=AB48a16b:AB48a16b",
          "--dtag=AB48a16b",
          "512x1024"
        ],
        "env": {
          "ONEAPI_DEVICE_SELECTOR": "level_zero:*"
        },
        "stdout": "Output template: perf,%engine%,%impl%,%name%,%prb%,%Gops%,%+ctime%,%-time%,%-Gflops%,%0time%,%0Gflops%\nperf,gpu,multi_po_reorder_sum,,--mode=P --max-ms-per-prb=100 --sum --engine=gpu --execution-mode=graph --sdt=bf16:bf16 --ddt=bf16 --stag=AB48a16b:AB48a16b --dtag=AB48a16b --scales=1 512x1024,0,1.24072,0.00352,0,0.00398547,0\ntests:1 passed:1 skipped:0 mistrusted:0 unimplemented:0 invalid_arguments:0 failed:0 listed:0\ntotal perf: min(ms):0.00352 avg(ms):0.00398547\ntotal: 0.33s; create_pd: 0.00s (0%); create_prim: 0.00s (0%); fill: 0.02s (7%); execute: 0.00s (0%);\n",
        "passed": true,
        "unit": "ms",
        "stddev": 0.0,
        "git_url": "https://github.com/uxlfoundation/oneDNN.git",
        "git_hash": "v3.8",
        "name": "onednn-sum-padding-2-graph",
        "lower_is_better": true,
        "suite": "BenchDNN"
      },
      {
        "label": "onednn-graph-sdpa-plain-f16-eager",
        "value": 0.33968,
        "command": [
          "/home/mateuszpn/workdir/onednn-build/tests/benchdnn/benchdnn",
          "--graph",
          "--mode=P",
          "--engine=gpu",
          "--max-ms-per-prb=100",
          "--execution-mode=direct",
          "--reset",
          "--dt=f16",
          "--case=complex_fusion/mha/sdpa-plain-implicit-causal-mask-fp32-bs1.json"
        ],
        "env": {
          "ONEAPI_DEVICE_SELECTOR": "level_zero:*"
        },
        "stdout": "Output template: perf,%engine%,%prb%,%-time%,%0time%\nperf,gpu,--mode=P --max-ms-per-prb=100 --graph --engine=gpu --dt=f16 --case=complex_fusion/mha/sdpa-plain-implicit-causal-mask-fp32-bs1.json,0.33968,0.342391\ntests:1 passed:1 skipped:0 mistrusted:0 unimplemented:0 invalid_arguments:0 failed:0 listed:0\ntotal perf: min(ms):0.33968 avg(ms):0.342391\ntotal: 0.54s; create_pd: 0.00s (0%); create_prim: 0.07s (13%); fill: 0.00s (0%); execute: 0.00s (0%);\n",
        "passed": true,
        "unit": "ms",
        "stddev": 0.00855442631178792,
        "git_url": "https://github.com/uxlfoundation/oneDNN.git",
        "git_hash": "v3.8",
        "name": "onednn-graph-sdpa-plain-f16-eager",
        "lower_is_better": true,
        "suite": "BenchDNN"
      },
      {
        "label": "onednn-graph-sdpa-plain-f32-eager",
        "value": 0.38512,
        "command": [
          "/home/mateuszpn/workdir/onednn-build/tests/benchdnn/benchdnn",
          "--graph",
          "--mode=P",
          "--engine=gpu",
          "--max-ms-per-prb=100",
          "--execution-mode=direct",
          "--reset",
          "--dt=f32",
          "--case=complex_fusion/mha/sdpa-plain-implicit-causal-mask-fp32-bs1.json"
        ],
        "env": {
          "ONEAPI_DEVICE_SELECTOR": "level_zero:*"
        },
        "stdout": "Output template: perf,%engine%,%prb%,%-time%,%0time%\nperf,gpu,--mode=P --max-ms-per-prb=100 --graph --engine=gpu --dt=f32 --case=complex_fusion/mha/sdpa-plain-implicit-causal-mask-fp32-bs1.json,0.38512,0.388208\ntests:1 passed:1 skipped:0 mistrusted:0 unimplemented:0 invalid_arguments:0 failed:0 listed:0\ntotal perf: min(ms):0.38512 avg(ms):0.388208\ntotal: 0.60s; create_pd: 0.00s (0%); create_prim: 0.07s (11%); fill: 0.00s (0%); execute: 0.00s (0%);\n",
        "passed": true,
        "unit": "ms",
        "stddev": 0.0066990148529466635,
        "git_url": "https://github.com/uxlfoundation/oneDNN.git",
        "git_hash": "v3.8",
        "name": "onednn-graph-sdpa-plain-f32-eager",
        "lower_is_better": true,
        "suite": "BenchDNN"
      },
      {
        "label": "onednn-graph-sdpa-plain-f32-graph",
        "value": 0.37952,
        "command": [
          "/home/mateuszpn/workdir/onednn-build/tests/benchdnn/benchdnn",
          "--graph",
          "--mode=P",
          "--engine=gpu",
          "--max-ms-per-prb=100",
          "--execution-mode=graph",
          "--reset",
          "--dt=f32",
          "--case=complex_fusion/mha/sdpa-plain-implicit-causal-mask-fp32-bs1.json"
        ],
        "env": {
          "ONEAPI_DEVICE_SELECTOR": "level_zero:*"
        },
        "stdout": "Output template: perf,%engine%,%prb%,%-time%,%0time%\nperf,gpu,--mode=P --max-ms-per-prb=100 --graph --engine=gpu --execution-mode=graph --dt=f32 --case=complex_fusion/mha/sdpa-plain-implicit-causal-mask-fp32-bs1.json,0.37952,0.382662\ntests:1 passed:1 skipped:0 mistrusted:0 unimplemented:0 invalid_arguments:0 failed:0 listed:0\ntotal perf: min(ms):0.37952 avg(ms):0.382662\ntotal: 0.58s; create_pd: 0.00s (0%); create_prim: 0.07s (11%); fill: 0.00s (0%); execute: 0.00s (0%);\n",
        "passed": true,
        "unit": "ms",
        "stddev": 0.011297102873450952,
        "git_url": "https://github.com/uxlfoundation/oneDNN.git",
        "git_hash": "v3.8",
        "name": "onednn-graph-sdpa-plain-f32-graph",
        "lower_is_better": true,
        "suite": "BenchDNN"
      }
    ],
    "name": "This PR",
    "hostname": "gkdse-pre-dnp-02",
    "git_hash": "1eb1026ad0ef",
    "github_repo": "mateuszpn/llvm",
    "date": "2025-06-27T09:56:15.698275+00:00",
    "compute_runtime": "unknown"
  }
];

benchmarkMetadata = {
  "SubmitKernel": {
    "type": "group",
    "description": "Measures CPU time overhead of submitting kernels through different APIs.",
    "notes": "Each layer builds on top of the previous layer, adding functionality and overhead.\nThe first layer is the Level Zero API, the second is the Unified Runtime API, and the third is the SYCL API.\nThe UR v2 adapter noticeably reduces UR layer overhead, also improving SYCL performance.\nWork is ongoing to reduce the overhead of the SYCL API\n",
    "unstable": null,
    "tags": [
      "submit",
      "micro",
      "SYCL",
      "UR",
      "L0"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": null,
    "explicit_group": null
  },
  "SinKernelGraph": {
    "type": "group",
    "description": null,
    "notes": null,
    "unstable": "This benchmark combines both eager and graph execution, and may not be representative of real use cases.",
    "tags": [
      "submit",
      "memory",
      "proxy",
      "SYCL",
      "UR",
      "L0",
      "graph"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": null,
    "explicit_group": null
  },
  "SubmitGraph": {
    "type": "group",
    "description": null,
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "micro",
      "SYCL",
      "UR",
      "L0",
      "graph"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": null,
    "explicit_group": null
  },
  "FinalizeGraph": {
    "type": "group",
    "description": null,
    "notes": null,
    "unstable": null,
    "tags": [
      "finalize",
      "micro",
      "SYCL",
      "graph"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": null,
    "explicit_group": null
  },
  "api_overhead_benchmark_syclpreview SubmitKernel out of order not using events": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting out-of-order kernels through SYCL Preview API, excluding kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 1 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "SYCL",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "SYCLPREVIEW SubmitKernel out of order, NumKernels 10",
    "explicit_group": "SubmitKernel out of order"
  },
  "api_overhead_benchmark_syclpreview SubmitKernel out of order not using events CPU count": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting out-of-order kernels through SYCL Preview API, excluding kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 1 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "SYCL",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "SYCLPREVIEW SubmitKernel out of order, NumKernels 10, CPU count",
    "explicit_group": "SubmitKernel out of order, CPU count"
  },
  "api_overhead_benchmark_syclpreview SubmitKernel out of order not using events KernelExecTime=20": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting out-of-order kernels through SYCL Preview API, excluding kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 20 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "SYCL",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "SYCLPREVIEW SubmitKernel out of order KernelExecTime=20, NumKernels 10",
    "explicit_group": "SubmitKernel out of order KernelExecTime=20"
  },
  "api_overhead_benchmark_syclpreview SubmitKernel out of order not using events KernelExecTime=20 CPU count": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting out-of-order kernels through SYCL Preview API, excluding kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 20 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "SYCL",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "SYCLPREVIEW SubmitKernel out of order KernelExecTime=20, NumKernels 10, CPU count",
    "explicit_group": "SubmitKernel out of order KernelExecTime=20, CPU count"
  },
  "api_overhead_benchmark_syclpreview SubmitKernel out of order": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting out-of-order kernels through SYCL Preview API, excluding kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 1 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "SYCL",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "SYCLPREVIEW SubmitKernel out of order using events, NumKernels 10",
    "explicit_group": "SubmitKernel out of order using events"
  },
  "api_overhead_benchmark_syclpreview SubmitKernel out of order CPU count": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting out-of-order kernels through SYCL Preview API, excluding kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 1 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "SYCL",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "SYCLPREVIEW SubmitKernel out of order using events, NumKernels 10, CPU count",
    "explicit_group": "SubmitKernel out of order using events, CPU count"
  },
  "api_overhead_benchmark_syclpreview SubmitKernel out of order KernelExecTime=20": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting out-of-order kernels through SYCL Preview API, excluding kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 20 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "SYCL",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "SYCLPREVIEW SubmitKernel out of order using events KernelExecTime=20, NumKernels 10",
    "explicit_group": "SubmitKernel out of order using events KernelExecTime=20"
  },
  "api_overhead_benchmark_syclpreview SubmitKernel out of order KernelExecTime=20 CPU count": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting out-of-order kernels through SYCL Preview API, excluding kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 20 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "SYCL",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "SYCLPREVIEW SubmitKernel out of order using events KernelExecTime=20, NumKernels 10, CPU count",
    "explicit_group": "SubmitKernel out of order using events KernelExecTime=20, CPU count"
  },
  "api_overhead_benchmark_syclpreview SubmitKernel out of order with measure completion not using events": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting out-of-order kernels through SYCL Preview API, including kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 1 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "SYCL",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "SYCLPREVIEW SubmitKernel out of order with measure completion, NumKernels 10",
    "explicit_group": "SubmitKernel out of order with completion"
  },
  "api_overhead_benchmark_syclpreview SubmitKernel out of order with measure completion not using events CPU count": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting out-of-order kernels through SYCL Preview API, including kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 1 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "SYCL",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "SYCLPREVIEW SubmitKernel out of order with measure completion, NumKernels 10, CPU count",
    "explicit_group": "SubmitKernel out of order with completion, CPU count"
  },
  "api_overhead_benchmark_syclpreview SubmitKernel out of order with measure completion not using events KernelExecTime=20": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting out-of-order kernels through SYCL Preview API, including kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 20 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "SYCL",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "SYCLPREVIEW SubmitKernel out of order with measure completion KernelExecTime=20, NumKernels 10",
    "explicit_group": "SubmitKernel out of order with completion KernelExecTime=20"
  },
  "api_overhead_benchmark_syclpreview SubmitKernel out of order with measure completion not using events KernelExecTime=20 CPU count": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting out-of-order kernels through SYCL Preview API, including kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 20 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "SYCL",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "SYCLPREVIEW SubmitKernel out of order with measure completion KernelExecTime=20, NumKernels 10, CPU count",
    "explicit_group": "SubmitKernel out of order with completion KernelExecTime=20, CPU count"
  },
  "api_overhead_benchmark_syclpreview SubmitKernel out of order with measure completion": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting out-of-order kernels through SYCL Preview API, including kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 1 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "SYCL",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "SYCLPREVIEW SubmitKernel out of order with measure completion using events, NumKernels 10",
    "explicit_group": "SubmitKernel out of order with completion using events"
  },
  "api_overhead_benchmark_syclpreview SubmitKernel out of order with measure completion CPU count": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting out-of-order kernels through SYCL Preview API, including kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 1 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "SYCL",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "SYCLPREVIEW SubmitKernel out of order with measure completion using events, NumKernels 10, CPU count",
    "explicit_group": "SubmitKernel out of order with completion using events, CPU count"
  },
  "api_overhead_benchmark_syclpreview SubmitKernel out of order with measure completion KernelExecTime=20": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting out-of-order kernels through SYCL Preview API, including kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 20 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "SYCL",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "SYCLPREVIEW SubmitKernel out of order with measure completion using events KernelExecTime=20, NumKernels 10",
    "explicit_group": "SubmitKernel out of order with completion using events KernelExecTime=20"
  },
  "api_overhead_benchmark_syclpreview SubmitKernel out of order with measure completion KernelExecTime=20 CPU count": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting out-of-order kernels through SYCL Preview API, including kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 20 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "SYCL",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "SYCLPREVIEW SubmitKernel out of order with measure completion using events KernelExecTime=20, NumKernels 10, CPU count",
    "explicit_group": "SubmitKernel out of order with completion using events KernelExecTime=20, CPU count"
  },
  "api_overhead_benchmark_syclpreview SubmitKernel in order not using events": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting in-order kernels through SYCL Preview API, excluding kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 1 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "SYCL",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "SYCLPREVIEW SubmitKernel in order, NumKernels 10",
    "explicit_group": "SubmitKernel in order"
  },
  "api_overhead_benchmark_syclpreview SubmitKernel in order not using events CPU count": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting in-order kernels through SYCL Preview API, excluding kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 1 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "SYCL",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "SYCLPREVIEW SubmitKernel in order, NumKernels 10, CPU count",
    "explicit_group": "SubmitKernel in order, CPU count"
  },
  "api_overhead_benchmark_syclpreview SubmitKernel in order not using events KernelExecTime=20": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting in-order kernels through SYCL Preview API, excluding kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 20 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "SYCL",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "SYCLPREVIEW SubmitKernel in order KernelExecTime=20, NumKernels 10",
    "explicit_group": "SubmitKernel in order KernelExecTime=20"
  },
  "api_overhead_benchmark_syclpreview SubmitKernel in order not using events KernelExecTime=20 CPU count": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting in-order kernels through SYCL Preview API, excluding kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 20 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "SYCL",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "SYCLPREVIEW SubmitKernel in order KernelExecTime=20, NumKernels 10, CPU count",
    "explicit_group": "SubmitKernel in order KernelExecTime=20, CPU count"
  },
  "api_overhead_benchmark_syclpreview SubmitKernel in order": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting in-order kernels through SYCL Preview API, excluding kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 1 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "SYCL",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "SYCLPREVIEW SubmitKernel in order using events, NumKernels 10",
    "explicit_group": "SubmitKernel in order using events"
  },
  "api_overhead_benchmark_syclpreview SubmitKernel in order CPU count": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting in-order kernels through SYCL Preview API, excluding kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 1 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "SYCL",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "SYCLPREVIEW SubmitKernel in order using events, NumKernels 10, CPU count",
    "explicit_group": "SubmitKernel in order using events, CPU count"
  },
  "api_overhead_benchmark_syclpreview SubmitKernel in order KernelExecTime=20": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting in-order kernels through SYCL Preview API, excluding kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 20 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "SYCL",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "SYCLPREVIEW SubmitKernel in order using events KernelExecTime=20, NumKernels 10",
    "explicit_group": "SubmitKernel in order using events KernelExecTime=20"
  },
  "api_overhead_benchmark_syclpreview SubmitKernel in order KernelExecTime=20 CPU count": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting in-order kernels through SYCL Preview API, excluding kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 20 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "SYCL",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "SYCLPREVIEW SubmitKernel in order using events KernelExecTime=20, NumKernels 10, CPU count",
    "explicit_group": "SubmitKernel in order using events KernelExecTime=20, CPU count"
  },
  "api_overhead_benchmark_syclpreview SubmitKernel in order with measure completion not using events": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting in-order kernels through SYCL Preview API, including kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 1 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "SYCL",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "SYCLPREVIEW SubmitKernel in order with measure completion, NumKernels 10",
    "explicit_group": "SubmitKernel in order with completion"
  },
  "api_overhead_benchmark_syclpreview SubmitKernel in order with measure completion not using events CPU count": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting in-order kernels through SYCL Preview API, including kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 1 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "SYCL",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "SYCLPREVIEW SubmitKernel in order with measure completion, NumKernels 10, CPU count",
    "explicit_group": "SubmitKernel in order with completion, CPU count"
  },
  "api_overhead_benchmark_syclpreview SubmitKernel in order with measure completion not using events KernelExecTime=20": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting in-order kernels through SYCL Preview API, including kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 20 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "SYCL",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "SYCLPREVIEW SubmitKernel in order with measure completion KernelExecTime=20, NumKernels 10",
    "explicit_group": "SubmitKernel in order with completion KernelExecTime=20"
  },
  "api_overhead_benchmark_syclpreview SubmitKernel in order with measure completion not using events KernelExecTime=20 CPU count": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting in-order kernels through SYCL Preview API, including kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 20 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "SYCL",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "SYCLPREVIEW SubmitKernel in order with measure completion KernelExecTime=20, NumKernels 10, CPU count",
    "explicit_group": "SubmitKernel in order with completion KernelExecTime=20, CPU count"
  },
  "api_overhead_benchmark_syclpreview SubmitKernel in order with measure completion": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting in-order kernels through SYCL Preview API, including kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 1 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "SYCL",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "SYCLPREVIEW SubmitKernel in order with measure completion using events, NumKernels 10",
    "explicit_group": "SubmitKernel in order with completion using events"
  },
  "api_overhead_benchmark_syclpreview SubmitKernel in order with measure completion CPU count": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting in-order kernels through SYCL Preview API, including kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 1 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "SYCL",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "SYCLPREVIEW SubmitKernel in order with measure completion using events, NumKernels 10, CPU count",
    "explicit_group": "SubmitKernel in order with completion using events, CPU count"
  },
  "api_overhead_benchmark_syclpreview SubmitKernel in order with measure completion KernelExecTime=20": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting in-order kernels through SYCL Preview API, including kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 20 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "SYCL",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "SYCLPREVIEW SubmitKernel in order with measure completion using events KernelExecTime=20, NumKernels 10",
    "explicit_group": "SubmitKernel in order with completion using events KernelExecTime=20"
  },
  "api_overhead_benchmark_syclpreview SubmitKernel in order with measure completion KernelExecTime=20 CPU count": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting in-order kernels through SYCL Preview API, including kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 20 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "SYCL",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "SYCLPREVIEW SubmitKernel in order with measure completion using events KernelExecTime=20, NumKernels 10, CPU count",
    "explicit_group": "SubmitKernel in order with completion using events KernelExecTime=20, CPU count"
  },
  "graph_api_benchmark_syclpreview SinKernelGraph graphs:0, numKernels:5": {
    "type": "benchmark",
    "description": "Measures SYCLPREVIEW performance when executing 5 sin kernels without graphs. Tests overhead and benefits of graph-based execution.",
    "notes": null,
    "unstable": "This benchmark combines both eager and graph execution, and may not be representative of real use cases.",
    "tags": [
      "graph",
      "SYCL",
      "proxy",
      "submit",
      "memory",
      "latency"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCLPREVIEW SinKernelGraph, graphs 0, numKernels 5",
    "explicit_group": "SinKernelGraph, numKernels: 5"
  },
  "graph_api_benchmark_syclpreview SinKernelGraph graphs:0, numKernels:100": {
    "type": "benchmark",
    "description": "Measures SYCLPREVIEW performance when executing 100 sin kernels without graphs. Tests overhead and benefits of graph-based execution.",
    "notes": null,
    "unstable": "This benchmark combines both eager and graph execution, and may not be representative of real use cases.",
    "tags": [
      "graph",
      "SYCL",
      "proxy",
      "submit",
      "memory",
      "latency"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCLPREVIEW SinKernelGraph, graphs 0, numKernels 100",
    "explicit_group": "SinKernelGraph, numKernels: 100"
  },
  "graph_api_benchmark_syclpreview SinKernelGraph graphs:1, numKernels:5": {
    "type": "benchmark",
    "description": "Measures SYCLPREVIEW performance when executing 5 sin kernels using graphs. Tests overhead and benefits of graph-based execution.",
    "notes": null,
    "unstable": "This benchmark combines both eager and graph execution, and may not be representative of real use cases.",
    "tags": [
      "graph",
      "SYCL",
      "proxy",
      "submit",
      "memory",
      "latency"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCLPREVIEW SinKernelGraph, graphs 1, numKernels 5",
    "explicit_group": "SinKernelGraph, numKernels: 5"
  },
  "graph_api_benchmark_syclpreview SinKernelGraph graphs:1, numKernels:100": {
    "type": "benchmark",
    "description": "Measures SYCLPREVIEW performance when executing 100 sin kernels using graphs. Tests overhead and benefits of graph-based execution.",
    "notes": null,
    "unstable": "This benchmark combines both eager and graph execution, and may not be representative of real use cases.",
    "tags": [
      "graph",
      "SYCL",
      "proxy",
      "submit",
      "memory",
      "latency"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCLPREVIEW SinKernelGraph, graphs 1, numKernels 100",
    "explicit_group": "SinKernelGraph, numKernels: 100"
  },
  "ulls_benchmark_syclpreview EmptyKernel wgc:1000, wgs:256": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "SYCL",
      "micro",
      "latency",
      "submit"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCLPREVIEW EmptyKernel, wgc 1000, wgs 256",
    "explicit_group": "EmptyKernel, wgc: 1000, wgs: 256"
  },
  "ulls_benchmark_syclpreview KernelSwitch count 8 kernelTime 200": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "SYCL",
      "micro",
      "latency",
      "submit"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCLPREVIEW KernelSwitch, count 8, kernelTime 200",
    "explicit_group": "KernelSwitch, count: 8, kernelTime: 200"
  },
  "graph_api_benchmark_syclpreview SubmitGraph numKernels:4 ioq 0 measureCompletion 0": {
    "type": "benchmark",
    "description": "Measures SYCLPREVIEW performance when executing 4 trivial kernels using graphs. Tests overhead and benefits of graph-based execution.",
    "notes": null,
    "unstable": null,
    "tags": [
      "graph",
      "SYCL",
      "micro",
      "submit",
      "latency"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCLPREVIEW SubmitGraph, numKernels 4, ioq 0, measureCompletion 0",
    "explicit_group": "SubmitGraph, numKernels: 4"
  },
  "graph_api_benchmark_syclpreview SubmitGraph numKernels:4 ioq 0 measureCompletion 1": {
    "type": "benchmark",
    "description": "Measures SYCLPREVIEW performance when executing 4 trivial kernels using graphs. Tests overhead and benefits of graph-based execution.",
    "notes": null,
    "unstable": null,
    "tags": [
      "graph",
      "SYCL",
      "micro",
      "submit",
      "latency"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCLPREVIEW SubmitGraph, numKernels 4, ioq 0, measureCompletion 1",
    "explicit_group": "SubmitGraph, numKernels: 4"
  },
  "graph_api_benchmark_syclpreview SubmitGraph numKernels:10 ioq 0 measureCompletion 0": {
    "type": "benchmark",
    "description": "Measures SYCLPREVIEW performance when executing 10 trivial kernels using graphs. Tests overhead and benefits of graph-based execution.",
    "notes": null,
    "unstable": null,
    "tags": [
      "graph",
      "SYCL",
      "micro",
      "submit",
      "latency"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCLPREVIEW SubmitGraph, numKernels 10, ioq 0, measureCompletion 0",
    "explicit_group": "SubmitGraph, numKernels: 10"
  },
  "graph_api_benchmark_syclpreview SubmitGraph numKernels:10 ioq 0 measureCompletion 1": {
    "type": "benchmark",
    "description": "Measures SYCLPREVIEW performance when executing 10 trivial kernels using graphs. Tests overhead and benefits of graph-based execution.",
    "notes": null,
    "unstable": null,
    "tags": [
      "graph",
      "SYCL",
      "micro",
      "submit",
      "latency"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCLPREVIEW SubmitGraph, numKernels 10, ioq 0, measureCompletion 1",
    "explicit_group": "SubmitGraph, numKernels: 10"
  },
  "graph_api_benchmark_syclpreview SubmitGraph numKernels:32 ioq 0 measureCompletion 0": {
    "type": "benchmark",
    "description": "Measures SYCLPREVIEW performance when executing 32 trivial kernels using graphs. Tests overhead and benefits of graph-based execution.",
    "notes": null,
    "unstable": null,
    "tags": [
      "graph",
      "SYCL",
      "micro",
      "submit",
      "latency"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCLPREVIEW SubmitGraph, numKernels 32, ioq 0, measureCompletion 0",
    "explicit_group": "SubmitGraph, numKernels: 32"
  },
  "graph_api_benchmark_syclpreview SubmitGraph numKernels:32 ioq 0 measureCompletion 1": {
    "type": "benchmark",
    "description": "Measures SYCLPREVIEW performance when executing 32 trivial kernels using graphs. Tests overhead and benefits of graph-based execution.",
    "notes": null,
    "unstable": null,
    "tags": [
      "graph",
      "SYCL",
      "micro",
      "submit",
      "latency"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCLPREVIEW SubmitGraph, numKernels 32, ioq 0, measureCompletion 1",
    "explicit_group": "SubmitGraph, numKernels: 32"
  },
  "graph_api_benchmark_syclpreview SubmitGraph numKernels:4 ioq 1 measureCompletion 0": {
    "type": "benchmark",
    "description": "Measures SYCLPREVIEW performance when executing 4 trivial kernels using graphs. Tests overhead and benefits of graph-based execution.",
    "notes": null,
    "unstable": null,
    "tags": [
      "graph",
      "SYCL",
      "micro",
      "submit",
      "latency"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCLPREVIEW SubmitGraph, numKernels 4, ioq 1, measureCompletion 0",
    "explicit_group": "SubmitGraph, numKernels: 4"
  },
  "graph_api_benchmark_syclpreview SubmitGraph numKernels:4 ioq 1 measureCompletion 1": {
    "type": "benchmark",
    "description": "Measures SYCLPREVIEW performance when executing 4 trivial kernels using graphs. Tests overhead and benefits of graph-based execution.",
    "notes": null,
    "unstable": null,
    "tags": [
      "graph",
      "SYCL",
      "micro",
      "submit",
      "latency"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCLPREVIEW SubmitGraph, numKernels 4, ioq 1, measureCompletion 1",
    "explicit_group": "SubmitGraph, numKernels: 4"
  },
  "graph_api_benchmark_syclpreview SubmitGraph numKernels:10 ioq 1 measureCompletion 0": {
    "type": "benchmark",
    "description": "Measures SYCLPREVIEW performance when executing 10 trivial kernels using graphs. Tests overhead and benefits of graph-based execution.",
    "notes": null,
    "unstable": null,
    "tags": [
      "graph",
      "SYCL",
      "micro",
      "submit",
      "latency"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCLPREVIEW SubmitGraph, numKernels 10, ioq 1, measureCompletion 0",
    "explicit_group": "SubmitGraph, numKernels: 10"
  },
  "graph_api_benchmark_syclpreview SubmitGraph numKernels:10 ioq 1 measureCompletion 1": {
    "type": "benchmark",
    "description": "Measures SYCLPREVIEW performance when executing 10 trivial kernels using graphs. Tests overhead and benefits of graph-based execution.",
    "notes": null,
    "unstable": null,
    "tags": [
      "graph",
      "SYCL",
      "micro",
      "submit",
      "latency"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCLPREVIEW SubmitGraph, numKernels 10, ioq 1, measureCompletion 1",
    "explicit_group": "SubmitGraph, numKernels: 10"
  },
  "graph_api_benchmark_syclpreview SubmitGraph numKernels:32 ioq 1 measureCompletion 0": {
    "type": "benchmark",
    "description": "Measures SYCLPREVIEW performance when executing 32 trivial kernels using graphs. Tests overhead and benefits of graph-based execution.",
    "notes": null,
    "unstable": null,
    "tags": [
      "graph",
      "SYCL",
      "micro",
      "submit",
      "latency"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCLPREVIEW SubmitGraph, numKernels 32, ioq 1, measureCompletion 0",
    "explicit_group": "SubmitGraph, numKernels: 32"
  },
  "graph_api_benchmark_syclpreview SubmitGraph numKernels:32 ioq 1 measureCompletion 1": {
    "type": "benchmark",
    "description": "Measures SYCLPREVIEW performance when executing 32 trivial kernels using graphs. Tests overhead and benefits of graph-based execution.",
    "notes": null,
    "unstable": null,
    "tags": [
      "graph",
      "SYCL",
      "micro",
      "submit",
      "latency"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCLPREVIEW SubmitGraph, numKernels 32, ioq 1, measureCompletion 1",
    "explicit_group": "SubmitGraph, numKernels: 32"
  },
  "api_overhead_benchmark_sycl SubmitKernel out of order not using events": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting out-of-order kernels through SYCL API, excluding kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 1 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "SYCL",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "SYCL SubmitKernel out of order, NumKernels 10",
    "explicit_group": "SubmitKernel out of order"
  },
  "api_overhead_benchmark_sycl SubmitKernel out of order not using events CPU count": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting out-of-order kernels through SYCL API, excluding kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 1 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "SYCL",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "SYCL SubmitKernel out of order, NumKernels 10, CPU count",
    "explicit_group": "SubmitKernel out of order, CPU count"
  },
  "api_overhead_benchmark_sycl SubmitKernel out of order not using events KernelExecTime=20": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting out-of-order kernels through SYCL API, excluding kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 20 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "SYCL",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "SYCL SubmitKernel out of order KernelExecTime=20, NumKernels 10",
    "explicit_group": "SubmitKernel out of order KernelExecTime=20"
  },
  "api_overhead_benchmark_sycl SubmitKernel out of order not using events KernelExecTime=20 CPU count": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting out-of-order kernels through SYCL API, excluding kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 20 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "SYCL",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "SYCL SubmitKernel out of order KernelExecTime=20, NumKernels 10, CPU count",
    "explicit_group": "SubmitKernel out of order KernelExecTime=20, CPU count"
  },
  "api_overhead_benchmark_sycl SubmitKernel out of order": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting out-of-order kernels through SYCL API, excluding kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 1 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "SYCL",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "SYCL SubmitKernel out of order using events, NumKernels 10",
    "explicit_group": "SubmitKernel out of order using events"
  },
  "api_overhead_benchmark_sycl SubmitKernel out of order CPU count": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting out-of-order kernels through SYCL API, excluding kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 1 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "SYCL",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "SYCL SubmitKernel out of order using events, NumKernels 10, CPU count",
    "explicit_group": "SubmitKernel out of order using events, CPU count"
  },
  "api_overhead_benchmark_sycl SubmitKernel out of order KernelExecTime=20": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting out-of-order kernels through SYCL API, excluding kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 20 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "SYCL",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "SYCL SubmitKernel out of order using events KernelExecTime=20, NumKernels 10",
    "explicit_group": "SubmitKernel out of order using events KernelExecTime=20"
  },
  "api_overhead_benchmark_sycl SubmitKernel out of order KernelExecTime=20 CPU count": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting out-of-order kernels through SYCL API, excluding kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 20 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "SYCL",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "SYCL SubmitKernel out of order using events KernelExecTime=20, NumKernels 10, CPU count",
    "explicit_group": "SubmitKernel out of order using events KernelExecTime=20, CPU count"
  },
  "api_overhead_benchmark_sycl SubmitKernel out of order with measure completion not using events": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting out-of-order kernels through SYCL API, including kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 1 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "SYCL",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "SYCL SubmitKernel out of order with measure completion, NumKernels 10",
    "explicit_group": "SubmitKernel out of order with completion"
  },
  "api_overhead_benchmark_sycl SubmitKernel out of order with measure completion not using events CPU count": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting out-of-order kernels through SYCL API, including kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 1 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "SYCL",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "SYCL SubmitKernel out of order with measure completion, NumKernels 10, CPU count",
    "explicit_group": "SubmitKernel out of order with completion, CPU count"
  },
  "api_overhead_benchmark_sycl SubmitKernel out of order with measure completion not using events KernelExecTime=20": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting out-of-order kernels through SYCL API, including kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 20 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "SYCL",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "SYCL SubmitKernel out of order with measure completion KernelExecTime=20, NumKernels 10",
    "explicit_group": "SubmitKernel out of order with completion KernelExecTime=20"
  },
  "api_overhead_benchmark_sycl SubmitKernel out of order with measure completion not using events KernelExecTime=20 CPU count": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting out-of-order kernels through SYCL API, including kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 20 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "SYCL",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "SYCL SubmitKernel out of order with measure completion KernelExecTime=20, NumKernels 10, CPU count",
    "explicit_group": "SubmitKernel out of order with completion KernelExecTime=20, CPU count"
  },
  "api_overhead_benchmark_sycl SubmitKernel out of order with measure completion": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting out-of-order kernels through SYCL API, including kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 1 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "SYCL",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "SYCL SubmitKernel out of order with measure completion using events, NumKernels 10",
    "explicit_group": "SubmitKernel out of order with completion using events"
  },
  "api_overhead_benchmark_sycl SubmitKernel out of order with measure completion CPU count": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting out-of-order kernels through SYCL API, including kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 1 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "SYCL",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "SYCL SubmitKernel out of order with measure completion using events, NumKernels 10, CPU count",
    "explicit_group": "SubmitKernel out of order with completion using events, CPU count"
  },
  "api_overhead_benchmark_sycl SubmitKernel out of order with measure completion KernelExecTime=20": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting out-of-order kernels through SYCL API, including kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 20 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "SYCL",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "SYCL SubmitKernel out of order with measure completion using events KernelExecTime=20, NumKernels 10",
    "explicit_group": "SubmitKernel out of order with completion using events KernelExecTime=20"
  },
  "api_overhead_benchmark_sycl SubmitKernel out of order with measure completion KernelExecTime=20 CPU count": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting out-of-order kernels through SYCL API, including kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 20 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "SYCL",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "SYCL SubmitKernel out of order with measure completion using events KernelExecTime=20, NumKernels 10, CPU count",
    "explicit_group": "SubmitKernel out of order with completion using events KernelExecTime=20, CPU count"
  },
  "api_overhead_benchmark_sycl SubmitKernel in order not using events": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting in-order kernels through SYCL API, excluding kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 1 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "SYCL",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "SYCL SubmitKernel in order, NumKernels 10",
    "explicit_group": "SubmitKernel in order"
  },
  "api_overhead_benchmark_sycl SubmitKernel in order not using events CPU count": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting in-order kernels through SYCL API, excluding kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 1 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "SYCL",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "SYCL SubmitKernel in order, NumKernels 10, CPU count",
    "explicit_group": "SubmitKernel in order, CPU count"
  },
  "api_overhead_benchmark_sycl SubmitKernel in order not using events KernelExecTime=20": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting in-order kernels through SYCL API, excluding kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 20 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "SYCL",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "SYCL SubmitKernel in order KernelExecTime=20, NumKernels 10",
    "explicit_group": "SubmitKernel in order KernelExecTime=20"
  },
  "api_overhead_benchmark_sycl SubmitKernel in order not using events KernelExecTime=20 CPU count": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting in-order kernels through SYCL API, excluding kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 20 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "SYCL",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "SYCL SubmitKernel in order KernelExecTime=20, NumKernels 10, CPU count",
    "explicit_group": "SubmitKernel in order KernelExecTime=20, CPU count"
  },
  "api_overhead_benchmark_sycl SubmitKernel in order": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting in-order kernels through SYCL API, excluding kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 1 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "SYCL",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "SYCL SubmitKernel in order using events, NumKernels 10",
    "explicit_group": "SubmitKernel in order using events"
  },
  "api_overhead_benchmark_sycl SubmitKernel in order CPU count": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting in-order kernels through SYCL API, excluding kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 1 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "SYCL",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "SYCL SubmitKernel in order using events, NumKernels 10, CPU count",
    "explicit_group": "SubmitKernel in order using events, CPU count"
  },
  "api_overhead_benchmark_sycl SubmitKernel in order KernelExecTime=20": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting in-order kernels through SYCL API, excluding kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 20 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "SYCL",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "SYCL SubmitKernel in order using events KernelExecTime=20, NumKernels 10",
    "explicit_group": "SubmitKernel in order using events KernelExecTime=20"
  },
  "api_overhead_benchmark_sycl SubmitKernel in order KernelExecTime=20 CPU count": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting in-order kernels through SYCL API, excluding kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 20 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "SYCL",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "SYCL SubmitKernel in order using events KernelExecTime=20, NumKernels 10, CPU count",
    "explicit_group": "SubmitKernel in order using events KernelExecTime=20, CPU count"
  },
  "api_overhead_benchmark_sycl SubmitKernel in order with measure completion not using events": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting in-order kernels through SYCL API, including kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 1 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "SYCL",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "SYCL SubmitKernel in order with measure completion, NumKernels 10",
    "explicit_group": "SubmitKernel in order with completion"
  },
  "api_overhead_benchmark_sycl SubmitKernel in order with measure completion not using events CPU count": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting in-order kernels through SYCL API, including kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 1 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "SYCL",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "SYCL SubmitKernel in order with measure completion, NumKernels 10, CPU count",
    "explicit_group": "SubmitKernel in order with completion, CPU count"
  },
  "api_overhead_benchmark_sycl SubmitKernel in order with measure completion not using events KernelExecTime=20": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting in-order kernels through SYCL API, including kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 20 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "SYCL",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "SYCL SubmitKernel in order with measure completion KernelExecTime=20, NumKernels 10",
    "explicit_group": "SubmitKernel in order with completion KernelExecTime=20"
  },
  "api_overhead_benchmark_sycl SubmitKernel in order with measure completion not using events KernelExecTime=20 CPU count": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting in-order kernels through SYCL API, including kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 20 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "SYCL",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "SYCL SubmitKernel in order with measure completion KernelExecTime=20, NumKernels 10, CPU count",
    "explicit_group": "SubmitKernel in order with completion KernelExecTime=20, CPU count"
  },
  "api_overhead_benchmark_sycl SubmitKernel in order with measure completion": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting in-order kernels through SYCL API, including kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 1 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "SYCL",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "SYCL SubmitKernel in order with measure completion using events, NumKernels 10",
    "explicit_group": "SubmitKernel in order with completion using events"
  },
  "api_overhead_benchmark_sycl SubmitKernel in order with measure completion CPU count": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting in-order kernels through SYCL API, including kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 1 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "SYCL",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "SYCL SubmitKernel in order with measure completion using events, NumKernels 10, CPU count",
    "explicit_group": "SubmitKernel in order with completion using events, CPU count"
  },
  "api_overhead_benchmark_sycl SubmitKernel in order with measure completion KernelExecTime=20": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting in-order kernels through SYCL API, including kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 20 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "SYCL",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "SYCL SubmitKernel in order with measure completion using events KernelExecTime=20, NumKernels 10",
    "explicit_group": "SubmitKernel in order with completion using events KernelExecTime=20"
  },
  "api_overhead_benchmark_sycl SubmitKernel in order with measure completion KernelExecTime=20 CPU count": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting in-order kernels through SYCL API, including kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 20 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "SYCL",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "SYCL SubmitKernel in order with measure completion using events KernelExecTime=20, NumKernels 10, CPU count",
    "explicit_group": "SubmitKernel in order with completion using events KernelExecTime=20, CPU count"
  },
  "graph_api_benchmark_sycl SinKernelGraph graphs:0, numKernels:5": {
    "type": "benchmark",
    "description": "Measures SYCL performance when executing 5 sin kernels without graphs. Tests overhead and benefits of graph-based execution.",
    "notes": null,
    "unstable": "This benchmark combines both eager and graph execution, and may not be representative of real use cases.",
    "tags": [
      "graph",
      "SYCL",
      "proxy",
      "submit",
      "memory",
      "latency"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCL SinKernelGraph, graphs 0, numKernels 5",
    "explicit_group": "SinKernelGraph, numKernels: 5"
  },
  "graph_api_benchmark_sycl SinKernelGraph graphs:0, numKernels:100": {
    "type": "benchmark",
    "description": "Measures SYCL performance when executing 100 sin kernels without graphs. Tests overhead and benefits of graph-based execution.",
    "notes": null,
    "unstable": "This benchmark combines both eager and graph execution, and may not be representative of real use cases.",
    "tags": [
      "graph",
      "SYCL",
      "proxy",
      "submit",
      "memory",
      "latency"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCL SinKernelGraph, graphs 0, numKernels 100",
    "explicit_group": "SinKernelGraph, numKernels: 100"
  },
  "graph_api_benchmark_sycl SinKernelGraph graphs:1, numKernels:5": {
    "type": "benchmark",
    "description": "Measures SYCL performance when executing 5 sin kernels using graphs. Tests overhead and benefits of graph-based execution.",
    "notes": null,
    "unstable": "This benchmark combines both eager and graph execution, and may not be representative of real use cases.",
    "tags": [
      "graph",
      "SYCL",
      "proxy",
      "submit",
      "memory",
      "latency"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCL SinKernelGraph, graphs 1, numKernels 5",
    "explicit_group": "SinKernelGraph, numKernels: 5"
  },
  "graph_api_benchmark_sycl SinKernelGraph graphs:1, numKernels:100": {
    "type": "benchmark",
    "description": "Measures SYCL performance when executing 100 sin kernels using graphs. Tests overhead and benefits of graph-based execution.",
    "notes": null,
    "unstable": "This benchmark combines both eager and graph execution, and may not be representative of real use cases.",
    "tags": [
      "graph",
      "SYCL",
      "proxy",
      "submit",
      "memory",
      "latency"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCL SinKernelGraph, graphs 1, numKernels 100",
    "explicit_group": "SinKernelGraph, numKernels: 100"
  },
  "ulls_benchmark_sycl EmptyKernel wgc:1000, wgs:256": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "SYCL",
      "micro",
      "latency",
      "submit"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCL EmptyKernel, wgc 1000, wgs 256",
    "explicit_group": "EmptyKernel, wgc: 1000, wgs: 256"
  },
  "ulls_benchmark_sycl KernelSwitch count 8 kernelTime 200": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "SYCL",
      "micro",
      "latency",
      "submit"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCL KernelSwitch, count 8, kernelTime 200",
    "explicit_group": "KernelSwitch, count: 8, kernelTime: 200"
  },
  "graph_api_benchmark_sycl SubmitGraph numKernels:4 ioq 0 measureCompletion 0": {
    "type": "benchmark",
    "description": "Measures SYCL performance when executing 4 trivial kernels using graphs. Tests overhead and benefits of graph-based execution.",
    "notes": null,
    "unstable": null,
    "tags": [
      "graph",
      "SYCL",
      "micro",
      "submit",
      "latency"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCL SubmitGraph, numKernels 4, ioq 0, measureCompletion 0",
    "explicit_group": "SubmitGraph, numKernels: 4"
  },
  "graph_api_benchmark_sycl SubmitGraph numKernels:4 ioq 0 measureCompletion 1": {
    "type": "benchmark",
    "description": "Measures SYCL performance when executing 4 trivial kernels using graphs. Tests overhead and benefits of graph-based execution.",
    "notes": null,
    "unstable": null,
    "tags": [
      "graph",
      "SYCL",
      "micro",
      "submit",
      "latency"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCL SubmitGraph, numKernels 4, ioq 0, measureCompletion 1",
    "explicit_group": "SubmitGraph, numKernels: 4"
  },
  "graph_api_benchmark_sycl SubmitGraph numKernels:10 ioq 0 measureCompletion 0": {
    "type": "benchmark",
    "description": "Measures SYCL performance when executing 10 trivial kernels using graphs. Tests overhead and benefits of graph-based execution.",
    "notes": null,
    "unstable": null,
    "tags": [
      "graph",
      "SYCL",
      "micro",
      "submit",
      "latency"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCL SubmitGraph, numKernels 10, ioq 0, measureCompletion 0",
    "explicit_group": "SubmitGraph, numKernels: 10"
  },
  "graph_api_benchmark_sycl SubmitGraph numKernels:10 ioq 0 measureCompletion 1": {
    "type": "benchmark",
    "description": "Measures SYCL performance when executing 10 trivial kernels using graphs. Tests overhead and benefits of graph-based execution.",
    "notes": null,
    "unstable": null,
    "tags": [
      "graph",
      "SYCL",
      "micro",
      "submit",
      "latency"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCL SubmitGraph, numKernels 10, ioq 0, measureCompletion 1",
    "explicit_group": "SubmitGraph, numKernels: 10"
  },
  "graph_api_benchmark_sycl SubmitGraph numKernels:32 ioq 0 measureCompletion 0": {
    "type": "benchmark",
    "description": "Measures SYCL performance when executing 32 trivial kernels using graphs. Tests overhead and benefits of graph-based execution.",
    "notes": null,
    "unstable": null,
    "tags": [
      "graph",
      "SYCL",
      "micro",
      "submit",
      "latency"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCL SubmitGraph, numKernels 32, ioq 0, measureCompletion 0",
    "explicit_group": "SubmitGraph, numKernels: 32"
  },
  "graph_api_benchmark_sycl SubmitGraph numKernels:32 ioq 0 measureCompletion 1": {
    "type": "benchmark",
    "description": "Measures SYCL performance when executing 32 trivial kernels using graphs. Tests overhead and benefits of graph-based execution.",
    "notes": null,
    "unstable": null,
    "tags": [
      "graph",
      "SYCL",
      "micro",
      "submit",
      "latency"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCL SubmitGraph, numKernels 32, ioq 0, measureCompletion 1",
    "explicit_group": "SubmitGraph, numKernels: 32"
  },
  "graph_api_benchmark_sycl SubmitGraph numKernels:4 ioq 1 measureCompletion 0": {
    "type": "benchmark",
    "description": "Measures SYCL performance when executing 4 trivial kernels using graphs. Tests overhead and benefits of graph-based execution.",
    "notes": null,
    "unstable": null,
    "tags": [
      "graph",
      "SYCL",
      "micro",
      "submit",
      "latency"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCL SubmitGraph, numKernels 4, ioq 1, measureCompletion 0",
    "explicit_group": "SubmitGraph, numKernels: 4"
  },
  "graph_api_benchmark_sycl SubmitGraph numKernels:4 ioq 1 measureCompletion 1": {
    "type": "benchmark",
    "description": "Measures SYCL performance when executing 4 trivial kernels using graphs. Tests overhead and benefits of graph-based execution.",
    "notes": null,
    "unstable": null,
    "tags": [
      "graph",
      "SYCL",
      "micro",
      "submit",
      "latency"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCL SubmitGraph, numKernels 4, ioq 1, measureCompletion 1",
    "explicit_group": "SubmitGraph, numKernels: 4"
  },
  "graph_api_benchmark_sycl SubmitGraph numKernels:10 ioq 1 measureCompletion 0": {
    "type": "benchmark",
    "description": "Measures SYCL performance when executing 10 trivial kernels using graphs. Tests overhead and benefits of graph-based execution.",
    "notes": null,
    "unstable": null,
    "tags": [
      "graph",
      "SYCL",
      "micro",
      "submit",
      "latency"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCL SubmitGraph, numKernels 10, ioq 1, measureCompletion 0",
    "explicit_group": "SubmitGraph, numKernels: 10"
  },
  "graph_api_benchmark_sycl SubmitGraph numKernels:10 ioq 1 measureCompletion 1": {
    "type": "benchmark",
    "description": "Measures SYCL performance when executing 10 trivial kernels using graphs. Tests overhead and benefits of graph-based execution.",
    "notes": null,
    "unstable": null,
    "tags": [
      "graph",
      "SYCL",
      "micro",
      "submit",
      "latency"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCL SubmitGraph, numKernels 10, ioq 1, measureCompletion 1",
    "explicit_group": "SubmitGraph, numKernels: 10"
  },
  "graph_api_benchmark_sycl SubmitGraph numKernels:32 ioq 1 measureCompletion 0": {
    "type": "benchmark",
    "description": "Measures SYCL performance when executing 32 trivial kernels using graphs. Tests overhead and benefits of graph-based execution.",
    "notes": null,
    "unstable": null,
    "tags": [
      "graph",
      "SYCL",
      "micro",
      "submit",
      "latency"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCL SubmitGraph, numKernels 32, ioq 1, measureCompletion 0",
    "explicit_group": "SubmitGraph, numKernels: 32"
  },
  "graph_api_benchmark_sycl SubmitGraph numKernels:32 ioq 1 measureCompletion 1": {
    "type": "benchmark",
    "description": "Measures SYCL performance when executing 32 trivial kernels using graphs. Tests overhead and benefits of graph-based execution.",
    "notes": null,
    "unstable": null,
    "tags": [
      "graph",
      "SYCL",
      "micro",
      "submit",
      "latency"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCL SubmitGraph, numKernels 32, ioq 1, measureCompletion 1",
    "explicit_group": "SubmitGraph, numKernels: 32"
  },
  "api_overhead_benchmark_l0 SubmitKernel out of order not using events": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting out-of-order kernels through Level Zero API, excluding kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 1 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "L0",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "L0 SubmitKernel out of order, NumKernels 10",
    "explicit_group": "SubmitKernel out of order"
  },
  "api_overhead_benchmark_l0 SubmitKernel out of order not using events CPU count": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting out-of-order kernels through Level Zero API, excluding kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 1 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "L0",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "L0 SubmitKernel out of order, NumKernels 10, CPU count",
    "explicit_group": "SubmitKernel out of order, CPU count"
  },
  "api_overhead_benchmark_l0 SubmitKernel out of order not using events KernelExecTime=20": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting out-of-order kernels through Level Zero API, excluding kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 20 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "L0",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "L0 SubmitKernel out of order KernelExecTime=20, NumKernels 10",
    "explicit_group": "SubmitKernel out of order KernelExecTime=20"
  },
  "api_overhead_benchmark_l0 SubmitKernel out of order not using events KernelExecTime=20 CPU count": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting out-of-order kernels through Level Zero API, excluding kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 20 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "L0",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "L0 SubmitKernel out of order KernelExecTime=20, NumKernels 10, CPU count",
    "explicit_group": "SubmitKernel out of order KernelExecTime=20, CPU count"
  },
  "api_overhead_benchmark_l0 SubmitKernel out of order": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting out-of-order kernels through Level Zero API, excluding kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 1 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "L0",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "L0 SubmitKernel out of order using events, NumKernels 10",
    "explicit_group": "SubmitKernel out of order using events"
  },
  "api_overhead_benchmark_l0 SubmitKernel out of order CPU count": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting out-of-order kernels through Level Zero API, excluding kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 1 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "L0",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "L0 SubmitKernel out of order using events, NumKernels 10, CPU count",
    "explicit_group": "SubmitKernel out of order using events, CPU count"
  },
  "api_overhead_benchmark_l0 SubmitKernel out of order KernelExecTime=20": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting out-of-order kernels through Level Zero API, excluding kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 20 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "L0",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "L0 SubmitKernel out of order using events KernelExecTime=20, NumKernels 10",
    "explicit_group": "SubmitKernel out of order using events KernelExecTime=20"
  },
  "api_overhead_benchmark_l0 SubmitKernel out of order KernelExecTime=20 CPU count": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting out-of-order kernels through Level Zero API, excluding kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 20 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "L0",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "L0 SubmitKernel out of order using events KernelExecTime=20, NumKernels 10, CPU count",
    "explicit_group": "SubmitKernel out of order using events KernelExecTime=20, CPU count"
  },
  "api_overhead_benchmark_l0 SubmitKernel out of order with measure completion not using events": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting out-of-order kernels through Level Zero API, including kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 1 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "L0",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "L0 SubmitKernel out of order with measure completion, NumKernels 10",
    "explicit_group": "SubmitKernel out of order with completion"
  },
  "api_overhead_benchmark_l0 SubmitKernel out of order with measure completion not using events CPU count": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting out-of-order kernels through Level Zero API, including kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 1 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "L0",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "L0 SubmitKernel out of order with measure completion, NumKernels 10, CPU count",
    "explicit_group": "SubmitKernel out of order with completion, CPU count"
  },
  "api_overhead_benchmark_l0 SubmitKernel out of order with measure completion not using events KernelExecTime=20": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting out-of-order kernels through Level Zero API, including kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 20 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "L0",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "L0 SubmitKernel out of order with measure completion KernelExecTime=20, NumKernels 10",
    "explicit_group": "SubmitKernel out of order with completion KernelExecTime=20"
  },
  "api_overhead_benchmark_l0 SubmitKernel out of order with measure completion not using events KernelExecTime=20 CPU count": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting out-of-order kernels through Level Zero API, including kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 20 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "L0",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "L0 SubmitKernel out of order with measure completion KernelExecTime=20, NumKernels 10, CPU count",
    "explicit_group": "SubmitKernel out of order with completion KernelExecTime=20, CPU count"
  },
  "api_overhead_benchmark_l0 SubmitKernel out of order with measure completion": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting out-of-order kernels through Level Zero API, including kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 1 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "L0",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "L0 SubmitKernel out of order with measure completion using events, NumKernels 10",
    "explicit_group": "SubmitKernel out of order with completion using events"
  },
  "api_overhead_benchmark_l0 SubmitKernel out of order with measure completion CPU count": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting out-of-order kernels through Level Zero API, including kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 1 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "L0",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "L0 SubmitKernel out of order with measure completion using events, NumKernels 10, CPU count",
    "explicit_group": "SubmitKernel out of order with completion using events, CPU count"
  },
  "api_overhead_benchmark_l0 SubmitKernel out of order with measure completion KernelExecTime=20": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting out-of-order kernels through Level Zero API, including kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 20 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "L0",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "L0 SubmitKernel out of order with measure completion using events KernelExecTime=20, NumKernels 10",
    "explicit_group": "SubmitKernel out of order with completion using events KernelExecTime=20"
  },
  "api_overhead_benchmark_l0 SubmitKernel out of order with measure completion KernelExecTime=20 CPU count": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting out-of-order kernels through Level Zero API, including kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 20 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "L0",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "L0 SubmitKernel out of order with measure completion using events KernelExecTime=20, NumKernels 10, CPU count",
    "explicit_group": "SubmitKernel out of order with completion using events KernelExecTime=20, CPU count"
  },
  "api_overhead_benchmark_l0 SubmitKernel in order not using events": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting in-order kernels through Level Zero API, excluding kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 1 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "L0",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "L0 SubmitKernel in order, NumKernels 10",
    "explicit_group": "SubmitKernel in order"
  },
  "api_overhead_benchmark_l0 SubmitKernel in order not using events CPU count": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting in-order kernels through Level Zero API, excluding kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 1 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "L0",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "L0 SubmitKernel in order, NumKernels 10, CPU count",
    "explicit_group": "SubmitKernel in order, CPU count"
  },
  "api_overhead_benchmark_l0 SubmitKernel in order not using events KernelExecTime=20": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting in-order kernels through Level Zero API, excluding kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 20 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "L0",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "L0 SubmitKernel in order KernelExecTime=20, NumKernels 10",
    "explicit_group": "SubmitKernel in order KernelExecTime=20"
  },
  "api_overhead_benchmark_l0 SubmitKernel in order not using events KernelExecTime=20 CPU count": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting in-order kernels through Level Zero API, excluding kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 20 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "L0",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "L0 SubmitKernel in order KernelExecTime=20, NumKernels 10, CPU count",
    "explicit_group": "SubmitKernel in order KernelExecTime=20, CPU count"
  },
  "api_overhead_benchmark_l0 SubmitKernel in order": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting in-order kernels through Level Zero API, excluding kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 1 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "L0",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "L0 SubmitKernel in order using events, NumKernels 10",
    "explicit_group": "SubmitKernel in order using events"
  },
  "api_overhead_benchmark_l0 SubmitKernel in order CPU count": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting in-order kernels through Level Zero API, excluding kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 1 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "L0",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "L0 SubmitKernel in order using events, NumKernels 10, CPU count",
    "explicit_group": "SubmitKernel in order using events, CPU count"
  },
  "api_overhead_benchmark_l0 SubmitKernel in order KernelExecTime=20": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting in-order kernels through Level Zero API, excluding kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 20 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "L0",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "L0 SubmitKernel in order using events KernelExecTime=20, NumKernels 10",
    "explicit_group": "SubmitKernel in order using events KernelExecTime=20"
  },
  "api_overhead_benchmark_l0 SubmitKernel in order KernelExecTime=20 CPU count": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting in-order kernels through Level Zero API, excluding kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 20 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "L0",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "L0 SubmitKernel in order using events KernelExecTime=20, NumKernels 10, CPU count",
    "explicit_group": "SubmitKernel in order using events KernelExecTime=20, CPU count"
  },
  "api_overhead_benchmark_l0 SubmitKernel in order with measure completion not using events": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting in-order kernels through Level Zero API, including kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 1 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "L0",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "L0 SubmitKernel in order with measure completion, NumKernels 10",
    "explicit_group": "SubmitKernel in order with completion"
  },
  "api_overhead_benchmark_l0 SubmitKernel in order with measure completion not using events CPU count": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting in-order kernels through Level Zero API, including kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 1 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "L0",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "L0 SubmitKernel in order with measure completion, NumKernels 10, CPU count",
    "explicit_group": "SubmitKernel in order with completion, CPU count"
  },
  "api_overhead_benchmark_l0 SubmitKernel in order with measure completion not using events KernelExecTime=20": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting in-order kernels through Level Zero API, including kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 20 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "L0",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "L0 SubmitKernel in order with measure completion KernelExecTime=20, NumKernels 10",
    "explicit_group": "SubmitKernel in order with completion KernelExecTime=20"
  },
  "api_overhead_benchmark_l0 SubmitKernel in order with measure completion not using events KernelExecTime=20 CPU count": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting in-order kernels through Level Zero API, including kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 20 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "L0",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "L0 SubmitKernel in order with measure completion KernelExecTime=20, NumKernels 10, CPU count",
    "explicit_group": "SubmitKernel in order with completion KernelExecTime=20, CPU count"
  },
  "api_overhead_benchmark_l0 SubmitKernel in order with measure completion": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting in-order kernels through Level Zero API, including kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 1 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "L0",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "L0 SubmitKernel in order with measure completion using events, NumKernels 10",
    "explicit_group": "SubmitKernel in order with completion using events"
  },
  "api_overhead_benchmark_l0 SubmitKernel in order with measure completion CPU count": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting in-order kernels through Level Zero API, including kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 1 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "L0",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "L0 SubmitKernel in order with measure completion using events, NumKernels 10, CPU count",
    "explicit_group": "SubmitKernel in order with completion using events, CPU count"
  },
  "api_overhead_benchmark_l0 SubmitKernel in order with measure completion KernelExecTime=20": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting in-order kernels through Level Zero API, including kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 20 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "L0",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "L0 SubmitKernel in order with measure completion using events KernelExecTime=20, NumKernels 10",
    "explicit_group": "SubmitKernel in order with completion using events KernelExecTime=20"
  },
  "api_overhead_benchmark_l0 SubmitKernel in order with measure completion KernelExecTime=20 CPU count": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting in-order kernels through Level Zero API, including kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 20 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "L0",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "L0 SubmitKernel in order with measure completion using events KernelExecTime=20, NumKernels 10, CPU count",
    "explicit_group": "SubmitKernel in order with completion using events KernelExecTime=20, CPU count"
  },
  "graph_api_benchmark_l0 SinKernelGraph graphs:0, numKernels:5": {
    "type": "benchmark",
    "description": "Measures L0 performance when executing 5 sin kernels without graphs. Tests overhead and benefits of graph-based execution.",
    "notes": null,
    "unstable": "This benchmark combines both eager and graph execution, and may not be representative of real use cases.",
    "tags": [
      "graph",
      "L0",
      "proxy",
      "submit",
      "memory",
      "latency"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "L0 SinKernelGraph, graphs 0, numKernels 5",
    "explicit_group": "SinKernelGraph, numKernels: 5"
  },
  "graph_api_benchmark_l0 SinKernelGraph graphs:0, numKernels:100": {
    "type": "benchmark",
    "description": "Measures L0 performance when executing 100 sin kernels without graphs. Tests overhead and benefits of graph-based execution.",
    "notes": null,
    "unstable": "This benchmark combines both eager and graph execution, and may not be representative of real use cases.",
    "tags": [
      "graph",
      "L0",
      "proxy",
      "submit",
      "memory",
      "latency"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "L0 SinKernelGraph, graphs 0, numKernels 100",
    "explicit_group": "SinKernelGraph, numKernels: 100"
  },
  "graph_api_benchmark_l0 SinKernelGraph graphs:1, numKernels:5": {
    "type": "benchmark",
    "description": "Measures L0 performance when executing 5 sin kernels using graphs. Tests overhead and benefits of graph-based execution.",
    "notes": null,
    "unstable": "This benchmark combines both eager and graph execution, and may not be representative of real use cases.",
    "tags": [
      "graph",
      "L0",
      "proxy",
      "submit",
      "memory",
      "latency"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "L0 SinKernelGraph, graphs 1, numKernels 5",
    "explicit_group": "SinKernelGraph, numKernels: 5"
  },
  "graph_api_benchmark_l0 SinKernelGraph graphs:1, numKernels:100": {
    "type": "benchmark",
    "description": "Measures L0 performance when executing 100 sin kernels using graphs. Tests overhead and benefits of graph-based execution.",
    "notes": null,
    "unstable": "This benchmark combines both eager and graph execution, and may not be representative of real use cases.",
    "tags": [
      "graph",
      "L0",
      "proxy",
      "submit",
      "memory",
      "latency"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "L0 SinKernelGraph, graphs 1, numKernels 100",
    "explicit_group": "SinKernelGraph, numKernels: 100"
  },
  "ulls_benchmark_l0 EmptyKernel wgc:1000, wgs:256": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "L0",
      "micro",
      "latency",
      "submit"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "L0 EmptyKernel, wgc 1000, wgs 256",
    "explicit_group": "EmptyKernel, wgc: 1000, wgs: 256"
  },
  "ulls_benchmark_l0 KernelSwitch count 8 kernelTime 200": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "L0",
      "micro",
      "latency",
      "submit"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "L0 KernelSwitch, count 8, kernelTime 200",
    "explicit_group": "KernelSwitch, count: 8, kernelTime: 200"
  },
  "graph_api_benchmark_l0 SubmitGraph numKernels:4 ioq 0 measureCompletion 0": {
    "type": "benchmark",
    "description": "Measures L0 performance when executing 4 trivial kernels using graphs. Tests overhead and benefits of graph-based execution.",
    "notes": null,
    "unstable": null,
    "tags": [
      "graph",
      "L0",
      "micro",
      "submit",
      "latency"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "L0 SubmitGraph, numKernels 4, ioq 0, measureCompletion 0",
    "explicit_group": "SubmitGraph, numKernels: 4"
  },
  "graph_api_benchmark_l0 SubmitGraph numKernels:4 ioq 0 measureCompletion 1": {
    "type": "benchmark",
    "description": "Measures L0 performance when executing 4 trivial kernels using graphs. Tests overhead and benefits of graph-based execution.",
    "notes": null,
    "unstable": null,
    "tags": [
      "graph",
      "L0",
      "micro",
      "submit",
      "latency"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "L0 SubmitGraph, numKernels 4, ioq 0, measureCompletion 1",
    "explicit_group": "SubmitGraph, numKernels: 4"
  },
  "graph_api_benchmark_l0 SubmitGraph numKernels:10 ioq 0 measureCompletion 0": {
    "type": "benchmark",
    "description": "Measures L0 performance when executing 10 trivial kernels using graphs. Tests overhead and benefits of graph-based execution.",
    "notes": null,
    "unstable": null,
    "tags": [
      "graph",
      "L0",
      "micro",
      "submit",
      "latency"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "L0 SubmitGraph, numKernels 10, ioq 0, measureCompletion 0",
    "explicit_group": "SubmitGraph, numKernels: 10"
  },
  "graph_api_benchmark_l0 SubmitGraph numKernels:10 ioq 0 measureCompletion 1": {
    "type": "benchmark",
    "description": "Measures L0 performance when executing 10 trivial kernels using graphs. Tests overhead and benefits of graph-based execution.",
    "notes": null,
    "unstable": null,
    "tags": [
      "graph",
      "L0",
      "micro",
      "submit",
      "latency"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "L0 SubmitGraph, numKernels 10, ioq 0, measureCompletion 1",
    "explicit_group": "SubmitGraph, numKernels: 10"
  },
  "graph_api_benchmark_l0 SubmitGraph numKernels:32 ioq 0 measureCompletion 0": {
    "type": "benchmark",
    "description": "Measures L0 performance when executing 32 trivial kernels using graphs. Tests overhead and benefits of graph-based execution.",
    "notes": null,
    "unstable": null,
    "tags": [
      "graph",
      "L0",
      "micro",
      "submit",
      "latency"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "L0 SubmitGraph, numKernels 32, ioq 0, measureCompletion 0",
    "explicit_group": "SubmitGraph, numKernels: 32"
  },
  "graph_api_benchmark_l0 SubmitGraph numKernels:32 ioq 0 measureCompletion 1": {
    "type": "benchmark",
    "description": "Measures L0 performance when executing 32 trivial kernels using graphs. Tests overhead and benefits of graph-based execution.",
    "notes": null,
    "unstable": null,
    "tags": [
      "graph",
      "L0",
      "micro",
      "submit",
      "latency"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "L0 SubmitGraph, numKernels 32, ioq 0, measureCompletion 1",
    "explicit_group": "SubmitGraph, numKernels: 32"
  },
  "graph_api_benchmark_l0 SubmitGraph numKernels:4 ioq 1 measureCompletion 0": {
    "type": "benchmark",
    "description": "Measures L0 performance when executing 4 trivial kernels using graphs. Tests overhead and benefits of graph-based execution.",
    "notes": null,
    "unstable": null,
    "tags": [
      "graph",
      "L0",
      "micro",
      "submit",
      "latency"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "L0 SubmitGraph, numKernels 4, ioq 1, measureCompletion 0",
    "explicit_group": "SubmitGraph, numKernels: 4"
  },
  "graph_api_benchmark_l0 SubmitGraph numKernels:4 ioq 1 measureCompletion 1": {
    "type": "benchmark",
    "description": "Measures L0 performance when executing 4 trivial kernels using graphs. Tests overhead and benefits of graph-based execution.",
    "notes": null,
    "unstable": null,
    "tags": [
      "graph",
      "L0",
      "micro",
      "submit",
      "latency"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "L0 SubmitGraph, numKernels 4, ioq 1, measureCompletion 1",
    "explicit_group": "SubmitGraph, numKernels: 4"
  },
  "graph_api_benchmark_l0 SubmitGraph numKernels:10 ioq 1 measureCompletion 0": {
    "type": "benchmark",
    "description": "Measures L0 performance when executing 10 trivial kernels using graphs. Tests overhead and benefits of graph-based execution.",
    "notes": null,
    "unstable": null,
    "tags": [
      "graph",
      "L0",
      "micro",
      "submit",
      "latency"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "L0 SubmitGraph, numKernels 10, ioq 1, measureCompletion 0",
    "explicit_group": "SubmitGraph, numKernels: 10"
  },
  "graph_api_benchmark_l0 SubmitGraph numKernels:10 ioq 1 measureCompletion 1": {
    "type": "benchmark",
    "description": "Measures L0 performance when executing 10 trivial kernels using graphs. Tests overhead and benefits of graph-based execution.",
    "notes": null,
    "unstable": null,
    "tags": [
      "graph",
      "L0",
      "micro",
      "submit",
      "latency"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "L0 SubmitGraph, numKernels 10, ioq 1, measureCompletion 1",
    "explicit_group": "SubmitGraph, numKernels: 10"
  },
  "graph_api_benchmark_l0 SubmitGraph numKernels:32 ioq 1 measureCompletion 0": {
    "type": "benchmark",
    "description": "Measures L0 performance when executing 32 trivial kernels using graphs. Tests overhead and benefits of graph-based execution.",
    "notes": null,
    "unstable": null,
    "tags": [
      "graph",
      "L0",
      "micro",
      "submit",
      "latency"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "L0 SubmitGraph, numKernels 32, ioq 1, measureCompletion 0",
    "explicit_group": "SubmitGraph, numKernels: 32"
  },
  "graph_api_benchmark_l0 SubmitGraph numKernels:32 ioq 1 measureCompletion 1": {
    "type": "benchmark",
    "description": "Measures L0 performance when executing 32 trivial kernels using graphs. Tests overhead and benefits of graph-based execution.",
    "notes": null,
    "unstable": null,
    "tags": [
      "graph",
      "L0",
      "micro",
      "submit",
      "latency"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "L0 SubmitGraph, numKernels 32, ioq 1, measureCompletion 1",
    "explicit_group": "SubmitGraph, numKernels: 32"
  },
  "api_overhead_benchmark_ur SubmitKernel out of order not using events": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting out-of-order kernels through Unified Runtime API, excluding kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 1 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "UR",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "UR SubmitKernel out of order, NumKernels 10",
    "explicit_group": "SubmitKernel out of order"
  },
  "api_overhead_benchmark_ur SubmitKernel out of order not using events CPU count": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting out-of-order kernels through Unified Runtime API, excluding kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 1 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "UR",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "UR SubmitKernel out of order, NumKernels 10, CPU count",
    "explicit_group": "SubmitKernel out of order, CPU count"
  },
  "api_overhead_benchmark_ur SubmitKernel out of order not using events KernelExecTime=20": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting out-of-order kernels through Unified Runtime API, excluding kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 20 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "UR",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "UR SubmitKernel out of order KernelExecTime=20, NumKernels 10",
    "explicit_group": "SubmitKernel out of order KernelExecTime=20"
  },
  "api_overhead_benchmark_ur SubmitKernel out of order not using events KernelExecTime=20 CPU count": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting out-of-order kernels through Unified Runtime API, excluding kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 20 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "UR",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "UR SubmitKernel out of order KernelExecTime=20, NumKernels 10, CPU count",
    "explicit_group": "SubmitKernel out of order KernelExecTime=20, CPU count"
  },
  "api_overhead_benchmark_ur SubmitKernel out of order": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting out-of-order kernels through Unified Runtime API, excluding kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 1 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "UR",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "UR SubmitKernel out of order using events, NumKernels 10",
    "explicit_group": "SubmitKernel out of order using events"
  },
  "api_overhead_benchmark_ur SubmitKernel out of order CPU count": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting out-of-order kernels through Unified Runtime API, excluding kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 1 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "UR",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "UR SubmitKernel out of order using events, NumKernels 10, CPU count",
    "explicit_group": "SubmitKernel out of order using events, CPU count"
  },
  "api_overhead_benchmark_ur SubmitKernel out of order KernelExecTime=20": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting out-of-order kernels through Unified Runtime API, excluding kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 20 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "UR",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "UR SubmitKernel out of order using events KernelExecTime=20, NumKernels 10",
    "explicit_group": "SubmitKernel out of order using events KernelExecTime=20"
  },
  "api_overhead_benchmark_ur SubmitKernel out of order KernelExecTime=20 CPU count": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting out-of-order kernels through Unified Runtime API, excluding kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 20 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "UR",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "UR SubmitKernel out of order using events KernelExecTime=20, NumKernels 10, CPU count",
    "explicit_group": "SubmitKernel out of order using events KernelExecTime=20, CPU count"
  },
  "api_overhead_benchmark_ur SubmitKernel out of order with measure completion not using events": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting out-of-order kernels through Unified Runtime API, including kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 1 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "UR",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "UR SubmitKernel out of order with measure completion, NumKernels 10",
    "explicit_group": "SubmitKernel out of order with completion"
  },
  "api_overhead_benchmark_ur SubmitKernel out of order with measure completion not using events CPU count": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting out-of-order kernels through Unified Runtime API, including kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 1 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "UR",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "UR SubmitKernel out of order with measure completion, NumKernels 10, CPU count",
    "explicit_group": "SubmitKernel out of order with completion, CPU count"
  },
  "api_overhead_benchmark_ur SubmitKernel out of order with measure completion not using events KernelExecTime=20": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting out-of-order kernels through Unified Runtime API, including kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 20 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "UR",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "UR SubmitKernel out of order with measure completion KernelExecTime=20, NumKernels 10",
    "explicit_group": "SubmitKernel out of order with completion KernelExecTime=20"
  },
  "api_overhead_benchmark_ur SubmitKernel out of order with measure completion not using events KernelExecTime=20 CPU count": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting out-of-order kernels through Unified Runtime API, including kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 20 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "UR",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "UR SubmitKernel out of order with measure completion KernelExecTime=20, NumKernels 10, CPU count",
    "explicit_group": "SubmitKernel out of order with completion KernelExecTime=20, CPU count"
  },
  "api_overhead_benchmark_ur SubmitKernel out of order with measure completion": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting out-of-order kernels through Unified Runtime API, including kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 1 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "UR",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "UR SubmitKernel out of order with measure completion using events, NumKernels 10",
    "explicit_group": "SubmitKernel out of order with completion using events"
  },
  "api_overhead_benchmark_ur SubmitKernel out of order with measure completion CPU count": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting out-of-order kernels through Unified Runtime API, including kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 1 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "UR",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "UR SubmitKernel out of order with measure completion using events, NumKernels 10, CPU count",
    "explicit_group": "SubmitKernel out of order with completion using events, CPU count"
  },
  "api_overhead_benchmark_ur SubmitKernel out of order with measure completion KernelExecTime=20": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting out-of-order kernels through Unified Runtime API, including kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 20 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "UR",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "UR SubmitKernel out of order with measure completion using events KernelExecTime=20, NumKernels 10",
    "explicit_group": "SubmitKernel out of order with completion using events KernelExecTime=20"
  },
  "api_overhead_benchmark_ur SubmitKernel out of order with measure completion KernelExecTime=20 CPU count": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting out-of-order kernels through Unified Runtime API, including kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 20 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "UR",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "UR SubmitKernel out of order with measure completion using events KernelExecTime=20, NumKernels 10, CPU count",
    "explicit_group": "SubmitKernel out of order with completion using events KernelExecTime=20, CPU count"
  },
  "api_overhead_benchmark_ur SubmitKernel in order not using events": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting in-order kernels through Unified Runtime API, excluding kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 1 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "UR",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "UR SubmitKernel in order, NumKernels 10",
    "explicit_group": "SubmitKernel in order"
  },
  "api_overhead_benchmark_ur SubmitKernel in order not using events CPU count": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting in-order kernels through Unified Runtime API, excluding kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 1 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "UR",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "UR SubmitKernel in order, NumKernels 10, CPU count",
    "explicit_group": "SubmitKernel in order, CPU count"
  },
  "api_overhead_benchmark_ur SubmitKernel in order not using events KernelExecTime=20": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting in-order kernels through Unified Runtime API, excluding kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 20 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "UR",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "UR SubmitKernel in order KernelExecTime=20, NumKernels 10",
    "explicit_group": "SubmitKernel in order KernelExecTime=20"
  },
  "api_overhead_benchmark_ur SubmitKernel in order not using events KernelExecTime=20 CPU count": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting in-order kernels through Unified Runtime API, excluding kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 20 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "UR",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "UR SubmitKernel in order KernelExecTime=20, NumKernels 10, CPU count",
    "explicit_group": "SubmitKernel in order KernelExecTime=20, CPU count"
  },
  "api_overhead_benchmark_ur SubmitKernel in order": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting in-order kernels through Unified Runtime API, excluding kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 1 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "UR",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "UR SubmitKernel in order using events, NumKernels 10",
    "explicit_group": "SubmitKernel in order using events"
  },
  "api_overhead_benchmark_ur SubmitKernel in order CPU count": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting in-order kernels through Unified Runtime API, excluding kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 1 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "UR",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "UR SubmitKernel in order using events, NumKernels 10, CPU count",
    "explicit_group": "SubmitKernel in order using events, CPU count"
  },
  "api_overhead_benchmark_ur SubmitKernel in order KernelExecTime=20": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting in-order kernels through Unified Runtime API, excluding kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 20 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "UR",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "UR SubmitKernel in order using events KernelExecTime=20, NumKernels 10",
    "explicit_group": "SubmitKernel in order using events KernelExecTime=20"
  },
  "api_overhead_benchmark_ur SubmitKernel in order KernelExecTime=20 CPU count": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting in-order kernels through Unified Runtime API, excluding kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 20 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "UR",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "UR SubmitKernel in order using events KernelExecTime=20, NumKernels 10, CPU count",
    "explicit_group": "SubmitKernel in order using events KernelExecTime=20, CPU count"
  },
  "api_overhead_benchmark_ur SubmitKernel in order with measure completion not using events": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting in-order kernels through Unified Runtime API, including kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 1 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "UR",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "UR SubmitKernel in order with measure completion, NumKernels 10",
    "explicit_group": "SubmitKernel in order with completion"
  },
  "api_overhead_benchmark_ur SubmitKernel in order with measure completion not using events CPU count": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting in-order kernels through Unified Runtime API, including kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 1 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "UR",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "UR SubmitKernel in order with measure completion, NumKernels 10, CPU count",
    "explicit_group": "SubmitKernel in order with completion, CPU count"
  },
  "api_overhead_benchmark_ur SubmitKernel in order with measure completion not using events KernelExecTime=20": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting in-order kernels through Unified Runtime API, including kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 20 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "UR",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "UR SubmitKernel in order with measure completion KernelExecTime=20, NumKernels 10",
    "explicit_group": "SubmitKernel in order with completion KernelExecTime=20"
  },
  "api_overhead_benchmark_ur SubmitKernel in order with measure completion not using events KernelExecTime=20 CPU count": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting in-order kernels through Unified Runtime API, including kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 20 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "UR",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "UR SubmitKernel in order with measure completion KernelExecTime=20, NumKernels 10, CPU count",
    "explicit_group": "SubmitKernel in order with completion KernelExecTime=20, CPU count"
  },
  "api_overhead_benchmark_ur SubmitKernel in order with measure completion": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting in-order kernels through Unified Runtime API, including kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 1 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "UR",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "UR SubmitKernel in order with measure completion using events, NumKernels 10",
    "explicit_group": "SubmitKernel in order with completion using events"
  },
  "api_overhead_benchmark_ur SubmitKernel in order with measure completion CPU count": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting in-order kernels through Unified Runtime API, including kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 1 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "UR",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "UR SubmitKernel in order with measure completion using events, NumKernels 10, CPU count",
    "explicit_group": "SubmitKernel in order with completion using events, CPU count"
  },
  "api_overhead_benchmark_ur SubmitKernel in order with measure completion KernelExecTime=20": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting in-order kernels through Unified Runtime API, including kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 20 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "UR",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "UR SubmitKernel in order with measure completion using events KernelExecTime=20, NumKernels 10",
    "explicit_group": "SubmitKernel in order with completion using events KernelExecTime=20"
  },
  "api_overhead_benchmark_ur SubmitKernel in order with measure completion KernelExecTime=20 CPU count": {
    "type": "benchmark",
    "description": "Measures CPU time overhead of submitting in-order kernels through Unified Runtime API, including kernel completion time. Runs 10 simple kernels with minimal execution time to isolate API overhead from kernel execution time.Each kernel executes for approximately 20 micro seconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "submit",
      "latency",
      "UR",
      "micro"
    ],
    "range_min": 0.0,
    "range_max": null,
    "display_name": "UR SubmitKernel in order with measure completion using events KernelExecTime=20, NumKernels 10, CPU count",
    "explicit_group": "SubmitKernel in order with completion using events KernelExecTime=20, CPU count"
  },
  "graph_api_benchmark_ur SinKernelGraph graphs:0, numKernels:5": {
    "type": "benchmark",
    "description": "Measures UR performance when executing 5 sin kernels without graphs. Tests overhead and benefits of graph-based execution.",
    "notes": null,
    "unstable": "This benchmark combines both eager and graph execution, and may not be representative of real use cases.",
    "tags": [
      "graph",
      "UR",
      "proxy",
      "submit",
      "memory",
      "latency"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "UR SinKernelGraph, graphs 0, numKernels 5",
    "explicit_group": "SinKernelGraph, numKernels: 5"
  },
  "graph_api_benchmark_ur SinKernelGraph graphs:0, numKernels:100": {
    "type": "benchmark",
    "description": "Measures UR performance when executing 100 sin kernels without graphs. Tests overhead and benefits of graph-based execution.",
    "notes": null,
    "unstable": "This benchmark combines both eager and graph execution, and may not be representative of real use cases.",
    "tags": [
      "graph",
      "UR",
      "proxy",
      "submit",
      "memory",
      "latency"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "UR SinKernelGraph, graphs 0, numKernels 100",
    "explicit_group": "SinKernelGraph, numKernels: 100"
  },
  "graph_api_benchmark_ur SinKernelGraph graphs:1, numKernels:5": {
    "type": "benchmark",
    "description": "Measures UR performance when executing 5 sin kernels using graphs. Tests overhead and benefits of graph-based execution.",
    "notes": null,
    "unstable": "This benchmark combines both eager and graph execution, and may not be representative of real use cases.",
    "tags": [
      "graph",
      "UR",
      "proxy",
      "submit",
      "memory",
      "latency"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "UR SinKernelGraph, graphs 1, numKernels 5",
    "explicit_group": "SinKernelGraph, numKernels: 5"
  },
  "graph_api_benchmark_ur SinKernelGraph graphs:1, numKernels:100": {
    "type": "benchmark",
    "description": "Measures UR performance when executing 100 sin kernels using graphs. Tests overhead and benefits of graph-based execution.",
    "notes": null,
    "unstable": "This benchmark combines both eager and graph execution, and may not be representative of real use cases.",
    "tags": [
      "graph",
      "UR",
      "proxy",
      "submit",
      "memory",
      "latency"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "UR SinKernelGraph, graphs 1, numKernels 100",
    "explicit_group": "SinKernelGraph, numKernels: 100"
  },
  "ulls_benchmark_ur EmptyKernel wgc:1000, wgs:256": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "UR",
      "micro",
      "latency",
      "submit"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "UR EmptyKernel, wgc 1000, wgs 256",
    "explicit_group": "EmptyKernel, wgc: 1000, wgs: 256"
  },
  "ulls_benchmark_ur KernelSwitch count 8 kernelTime 200": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "UR",
      "micro",
      "latency",
      "submit"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "UR KernelSwitch, count 8, kernelTime 200",
    "explicit_group": "KernelSwitch, count: 8, kernelTime: 200"
  },
  "graph_api_benchmark_ur SubmitGraph numKernels:4 ioq 0 measureCompletion 0": {
    "type": "benchmark",
    "description": "Measures UR performance when executing 4 trivial kernels using graphs. Tests overhead and benefits of graph-based execution.",
    "notes": null,
    "unstable": null,
    "tags": [
      "graph",
      "UR",
      "micro",
      "submit",
      "latency"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "UR SubmitGraph, numKernels 4, ioq 0, measureCompletion 0",
    "explicit_group": "SubmitGraph, numKernels: 4"
  },
  "graph_api_benchmark_ur SubmitGraph numKernels:4 ioq 0 measureCompletion 1": {
    "type": "benchmark",
    "description": "Measures UR performance when executing 4 trivial kernels using graphs. Tests overhead and benefits of graph-based execution.",
    "notes": null,
    "unstable": null,
    "tags": [
      "graph",
      "UR",
      "micro",
      "submit",
      "latency"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "UR SubmitGraph, numKernels 4, ioq 0, measureCompletion 1",
    "explicit_group": "SubmitGraph, numKernels: 4"
  },
  "graph_api_benchmark_ur SubmitGraph numKernels:10 ioq 0 measureCompletion 0": {
    "type": "benchmark",
    "description": "Measures UR performance when executing 10 trivial kernels using graphs. Tests overhead and benefits of graph-based execution.",
    "notes": null,
    "unstable": null,
    "tags": [
      "graph",
      "UR",
      "micro",
      "submit",
      "latency"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "UR SubmitGraph, numKernels 10, ioq 0, measureCompletion 0",
    "explicit_group": "SubmitGraph, numKernels: 10"
  },
  "graph_api_benchmark_ur SubmitGraph numKernels:10 ioq 0 measureCompletion 1": {
    "type": "benchmark",
    "description": "Measures UR performance when executing 10 trivial kernels using graphs. Tests overhead and benefits of graph-based execution.",
    "notes": null,
    "unstable": null,
    "tags": [
      "graph",
      "UR",
      "micro",
      "submit",
      "latency"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "UR SubmitGraph, numKernels 10, ioq 0, measureCompletion 1",
    "explicit_group": "SubmitGraph, numKernels: 10"
  },
  "graph_api_benchmark_ur SubmitGraph numKernels:32 ioq 0 measureCompletion 0": {
    "type": "benchmark",
    "description": "Measures UR performance when executing 32 trivial kernels using graphs. Tests overhead and benefits of graph-based execution.",
    "notes": null,
    "unstable": null,
    "tags": [
      "graph",
      "UR",
      "micro",
      "submit",
      "latency"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "UR SubmitGraph, numKernels 32, ioq 0, measureCompletion 0",
    "explicit_group": "SubmitGraph, numKernels: 32"
  },
  "graph_api_benchmark_ur SubmitGraph numKernels:32 ioq 0 measureCompletion 1": {
    "type": "benchmark",
    "description": "Measures UR performance when executing 32 trivial kernels using graphs. Tests overhead and benefits of graph-based execution.",
    "notes": null,
    "unstable": null,
    "tags": [
      "graph",
      "UR",
      "micro",
      "submit",
      "latency"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "UR SubmitGraph, numKernels 32, ioq 0, measureCompletion 1",
    "explicit_group": "SubmitGraph, numKernels: 32"
  },
  "graph_api_benchmark_ur SubmitGraph numKernels:4 ioq 1 measureCompletion 0": {
    "type": "benchmark",
    "description": "Measures UR performance when executing 4 trivial kernels using graphs. Tests overhead and benefits of graph-based execution.",
    "notes": null,
    "unstable": null,
    "tags": [
      "graph",
      "UR",
      "micro",
      "submit",
      "latency"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "UR SubmitGraph, numKernels 4, ioq 1, measureCompletion 0",
    "explicit_group": "SubmitGraph, numKernels: 4"
  },
  "graph_api_benchmark_ur SubmitGraph numKernels:4 ioq 1 measureCompletion 1": {
    "type": "benchmark",
    "description": "Measures UR performance when executing 4 trivial kernels using graphs. Tests overhead and benefits of graph-based execution.",
    "notes": null,
    "unstable": null,
    "tags": [
      "graph",
      "UR",
      "micro",
      "submit",
      "latency"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "UR SubmitGraph, numKernels 4, ioq 1, measureCompletion 1",
    "explicit_group": "SubmitGraph, numKernels: 4"
  },
  "graph_api_benchmark_ur SubmitGraph numKernels:10 ioq 1 measureCompletion 0": {
    "type": "benchmark",
    "description": "Measures UR performance when executing 10 trivial kernels using graphs. Tests overhead and benefits of graph-based execution.",
    "notes": null,
    "unstable": null,
    "tags": [
      "graph",
      "UR",
      "micro",
      "submit",
      "latency"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "UR SubmitGraph, numKernels 10, ioq 1, measureCompletion 0",
    "explicit_group": "SubmitGraph, numKernels: 10"
  },
  "graph_api_benchmark_ur SubmitGraph numKernels:10 ioq 1 measureCompletion 1": {
    "type": "benchmark",
    "description": "Measures UR performance when executing 10 trivial kernels using graphs. Tests overhead and benefits of graph-based execution.",
    "notes": null,
    "unstable": null,
    "tags": [
      "graph",
      "UR",
      "micro",
      "submit",
      "latency"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "UR SubmitGraph, numKernels 10, ioq 1, measureCompletion 1",
    "explicit_group": "SubmitGraph, numKernels: 10"
  },
  "graph_api_benchmark_ur SubmitGraph numKernels:32 ioq 1 measureCompletion 0": {
    "type": "benchmark",
    "description": "Measures UR performance when executing 32 trivial kernels using graphs. Tests overhead and benefits of graph-based execution.",
    "notes": null,
    "unstable": null,
    "tags": [
      "graph",
      "UR",
      "micro",
      "submit",
      "latency"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "UR SubmitGraph, numKernels 32, ioq 1, measureCompletion 0",
    "explicit_group": "SubmitGraph, numKernels: 32"
  },
  "graph_api_benchmark_ur SubmitGraph numKernels:32 ioq 1 measureCompletion 1": {
    "type": "benchmark",
    "description": "Measures UR performance when executing 32 trivial kernels using graphs. Tests overhead and benefits of graph-based execution.",
    "notes": null,
    "unstable": null,
    "tags": [
      "graph",
      "UR",
      "micro",
      "submit",
      "latency"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "UR SubmitGraph, numKernels 32, ioq 1, measureCompletion 1",
    "explicit_group": "SubmitGraph, numKernels: 32"
  },
  "memory_benchmark_sycl QueueInOrderMemcpy from Device to Device, size 1024": {
    "type": "benchmark",
    "description": "Measures SYCL in-order queue memory copy performance for copy and command submission from Device to Device with 1024 bytes, executed 100 times per iteration.",
    "notes": null,
    "unstable": null,
    "tags": [
      "memory",
      "latency",
      "SYCL",
      "micro"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCL QueueInOrderMemcpy from Device to Device, size 1024",
    "explicit_group": ""
  },
  "memory_benchmark_sycl QueueInOrderMemcpy from Host to Device, size 1024": {
    "type": "benchmark",
    "description": "Measures SYCL in-order queue memory copy performance for copy and command submission from Host to Device with 1024 bytes, executed 100 times per iteration.",
    "notes": null,
    "unstable": null,
    "tags": [
      "memory",
      "latency",
      "SYCL",
      "micro"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCL QueueInOrderMemcpy from Host to Device, size 1024",
    "explicit_group": ""
  },
  "memory_benchmark_sycl QueueMemcpy from Device to Device, size 1024": {
    "type": "benchmark",
    "description": "Measures general SYCL queue memory copy performance from Device to Device with 1024 bytes per operation.",
    "notes": null,
    "unstable": null,
    "tags": [
      "memory",
      "latency",
      "SYCL",
      "micro"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCL QueueMemcpy from Device to Device, size 1024",
    "explicit_group": ""
  },
  "memory_benchmark_sycl StreamMemory, placement Device, type Triad, size 10240": {
    "type": "benchmark",
    "description": "Measures Device memory bandwidth using Triad pattern with 10240 bytes. Higher values (GB/s) indicate better performance.",
    "notes": null,
    "unstable": null,
    "tags": [
      "memory",
      "throughput",
      "SYCL",
      "micro"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCL StreamMemory, placement Device, type Triad, size 10240",
    "explicit_group": ""
  },
  "api_overhead_benchmark_sycl ExecImmediateCopyQueue out of order from Device to Device, size 1024": {
    "type": "benchmark",
    "description": "Measures SYCL out-of-order queue overhead for copy-only from Device to Device memory with 1024 bytes. Tests immediate execution overheads.",
    "notes": null,
    "unstable": null,
    "tags": [
      "memory",
      "submit",
      "latency",
      "SYCL",
      "micro"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCL ExecImmediateCopyQueue out of order from Device to Device, size 1024",
    "explicit_group": ""
  },
  "api_overhead_benchmark_sycl ExecImmediateCopyQueue in order from Device to Host, size 1024": {
    "type": "benchmark",
    "description": "Measures SYCL in-order queue overhead for copy-only from Device to Host memory with 1024 bytes. Tests immediate execution overheads.",
    "notes": null,
    "unstable": null,
    "tags": [
      "memory",
      "submit",
      "latency",
      "SYCL",
      "micro"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCL ExecImmediateCopyQueue in order from Device to Host, size 1024",
    "explicit_group": ""
  },
  "miscellaneous_benchmark_sycl VectorSum": {
    "type": "benchmark",
    "description": "Measures performance of vector addition across 3D grid (512x256x256 elements) using SYCL.",
    "notes": null,
    "unstable": null,
    "tags": [
      "math",
      "throughput",
      "SYCL",
      "micro"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCL VectorSum",
    "explicit_group": ""
  },
  "graph_api_benchmark_sycl FinalizeGraph rebuildGraphEveryIter:0 graphStructure:Gromacs": {
    "type": "benchmark",
    "description": "Measures the time taken to finalize a SYCL graph, using a graph structure based on the usage of graphs in Gromacs. It measures finalizing the same modifiable graph repeatedly over multiple iterations.",
    "notes": null,
    "unstable": null,
    "tags": [
      "graph",
      "SYCL",
      "micro",
      "finalize",
      "latency"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCL FinalizeGraph, rebuildGraphEveryIter 0, graphStructure Gromacs",
    "explicit_group": "FinalizeGraph, GraphStructure: Gromacs"
  },
  "graph_api_benchmark_sycl FinalizeGraph rebuildGraphEveryIter:1 graphStructure:Gromacs": {
    "type": "benchmark",
    "description": "Measures the time taken to finalize a SYCL graph, using a graph structure based on the usage of graphs in Gromacs. It measures finalizing a unique modifiable graph per iteration.",
    "notes": null,
    "unstable": null,
    "tags": [
      "graph",
      "SYCL",
      "micro",
      "finalize",
      "latency"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCL FinalizeGraph, rebuildGraphEveryIter 1, graphStructure Gromacs",
    "explicit_group": "FinalizeGraph, GraphStructure: Gromacs"
  },
  "graph_api_benchmark_sycl FinalizeGraph rebuildGraphEveryIter:0 graphStructure:Llama": {
    "type": "benchmark",
    "description": "Measures the time taken to finalize a SYCL graph, using a graph structure based on the usage of graphs in Llama. It measures finalizing the same modifiable graph repeatedly over multiple iterations.",
    "notes": null,
    "unstable": null,
    "tags": [
      "graph",
      "SYCL",
      "micro",
      "finalize",
      "latency"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCL FinalizeGraph, rebuildGraphEveryIter 0, graphStructure Llama",
    "explicit_group": "FinalizeGraph, GraphStructure: Llama"
  },
  "graph_api_benchmark_sycl FinalizeGraph rebuildGraphEveryIter:1 graphStructure:Llama": {
    "type": "benchmark",
    "description": "Measures the time taken to finalize a SYCL graph, using a graph structure based on the usage of graphs in Llama. It measures finalizing a unique modifiable graph per iteration.",
    "notes": null,
    "unstable": null,
    "tags": [
      "graph",
      "SYCL",
      "micro",
      "finalize",
      "latency"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCL FinalizeGraph, rebuildGraphEveryIter 1, graphStructure Llama",
    "explicit_group": "FinalizeGraph, GraphStructure: Llama"
  },
  "multithread_benchmark_ur MemcpyExecute opsPerThread:400, numThreads:1, allocSize:102400 srcUSM:1 dstUSM:1": {
    "type": "benchmark",
    "description": "Measures multithreaded memory copy performance with 1 threads each performing 400 operations on 102400 bytes from device to device memory with events with driver copy offload without barrier. ",
    "notes": null,
    "unstable": null,
    "tags": [
      "memory",
      "latency",
      "UR",
      "micro"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "UR MemcpyExecute, opsPerThread 400, numThreads 1, allocSize 102400, srcUSM 1, dstUSM 1",
    "explicit_group": "MemcpyExecute, opsPerThread: 400, numThreads: 1, allocSize: 102400"
  },
  "multithread_benchmark_ur MemcpyExecute opsPerThread:400, numThreads:1, allocSize:102400 srcUSM:0 dstUSM:1": {
    "type": "benchmark",
    "description": "Measures multithreaded memory copy performance with 1 threads each performing 400 operations on 102400 bytes from host to device memory with events with driver copy offload without barrier. ",
    "notes": null,
    "unstable": null,
    "tags": [
      "memory",
      "latency",
      "UR",
      "micro"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "UR MemcpyExecute, opsPerThread 400, numThreads 1, allocSize 102400, srcUSM 0, dstUSM 1",
    "explicit_group": "MemcpyExecute, opsPerThread: 400, numThreads: 1, allocSize: 102400"
  },
  "multithread_benchmark_ur MemcpyExecute opsPerThread:100, numThreads:4, allocSize:102400 srcUSM:1 dstUSM:1 without events": {
    "type": "benchmark",
    "description": "Measures multithreaded memory copy performance with 4 threads each performing 100 operations on 102400 bytes from device to device memory without events with driver copy offload without barrier. ",
    "notes": null,
    "unstable": null,
    "tags": [
      "memory",
      "latency",
      "UR",
      "micro"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "UR MemcpyExecute, opsPerThread 100, numThreads 4, allocSize 102400, srcUSM 1, dstUSM 1, without events",
    "explicit_group": "MemcpyExecute, opsPerThread: 100, numThreads: 4, allocSize: 102400"
  },
  "multithread_benchmark_ur MemcpyExecute opsPerThread:100, numThreads:4, allocSize:102400 srcUSM:1 dstUSM:1 without events without copy offload": {
    "type": "benchmark",
    "description": "Measures multithreaded memory copy performance with 4 threads each performing 100 operations on 102400 bytes from device to device memory without events without driver copy offload without barrier. ",
    "notes": null,
    "unstable": null,
    "tags": [
      "memory",
      "latency",
      "UR",
      "micro"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "UR MemcpyExecute, opsPerThread 100, numThreads 4, allocSize 102400, srcUSM 1, dstUSM 1, without events without copy offload",
    "explicit_group": "MemcpyExecute, opsPerThread: 100, numThreads: 4, allocSize: 102400"
  },
  "multithread_benchmark_ur MemcpyExecute opsPerThread:4096, numThreads:4, allocSize:1024 srcUSM:0 dstUSM:1 without events": {
    "type": "benchmark",
    "description": "Measures multithreaded memory copy performance with 4 threads each performing 4096 operations on 1024 bytes from host to device memory without events with driver copy offload without barrier. ",
    "notes": null,
    "unstable": null,
    "tags": [
      "memory",
      "latency",
      "UR",
      "micro"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "UR MemcpyExecute, opsPerThread 4096, numThreads 4, allocSize 1024, srcUSM 0, dstUSM 1, without events",
    "explicit_group": "MemcpyExecute, opsPerThread: 4096, numThreads: 4, allocSize: 1024"
  },
  "multithread_benchmark_ur MemcpyExecute opsPerThread:4096, numThreads:4, allocSize:1024 srcUSM:0 dstUSM:1 without events with barrier": {
    "type": "benchmark",
    "description": "Measures multithreaded memory copy performance with 4 threads each performing 4096 operations on 1024 bytes from host to device memory without events with driver copy offload with barrier. ",
    "notes": null,
    "unstable": null,
    "tags": [
      "memory",
      "latency",
      "UR",
      "micro"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "UR MemcpyExecute, opsPerThread 4096, numThreads 4, allocSize 1024, srcUSM 0, dstUSM 1, without events",
    "explicit_group": "MemcpyExecute, opsPerThread: 4096, numThreads: 4, allocSize: 1024"
  },
  "api_overhead_benchmark_ur UsmMemoryAllocation usmMemoryPlacement:Device size:256 measureMode:Both": {
    "type": "benchmark",
    "description": "Measures memory allocation overhead by allocating 256 bytes of usm Device memory and free'ing it immediately. Both memory allocation and memory free are timed. ",
    "notes": null,
    "unstable": null,
    "tags": [
      "UR",
      "micro",
      "latency",
      "memory"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "UR UsmMemoryAllocation, usmMemoryPlacement Device, size 256, measureMode Both",
    "explicit_group": "UsmMemoryAllocation"
  },
  "api_overhead_benchmark_ur UsmMemoryAllocation usmMemoryPlacement:Device size:262144 measureMode:Both": {
    "type": "benchmark",
    "description": "Measures memory allocation overhead by allocating 262144 bytes of usm Device memory and free'ing it immediately. Both memory allocation and memory free are timed. ",
    "notes": null,
    "unstable": null,
    "tags": [
      "UR",
      "micro",
      "latency",
      "memory"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "UR UsmMemoryAllocation, usmMemoryPlacement Device, size 262144, measureMode Both",
    "explicit_group": "UsmMemoryAllocation"
  },
  "api_overhead_benchmark_ur UsmBatchMemoryAllocation usmMemoryPlacement:Device allocationCount:128 size:256 measureMode:Both": {
    "type": "benchmark",
    "description": "Measures memory allocation overhead by allocating 256 bytes of usm Device memory 128 times, then free'ing it all at once. Both memory allocation and memory free are timed. ",
    "notes": null,
    "unstable": null,
    "tags": [
      "UR",
      "micro",
      "latency",
      "memory"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "UR UsmBatchMemoryAllocation, usmMemoryPlacement Device, allocationCount 128, size 256, measureMode Both",
    "explicit_group": "UsmBatchMemoryAllocation"
  },
  "api_overhead_benchmark_ur UsmBatchMemoryAllocation usmMemoryPlacement:Device allocationCount:128 size:16384 measureMode:Both": {
    "type": "benchmark",
    "description": "Measures memory allocation overhead by allocating 16384 bytes of usm Device memory 128 times, then free'ing it all at once. Both memory allocation and memory free are timed. ",
    "notes": null,
    "unstable": null,
    "tags": [
      "UR",
      "micro",
      "latency",
      "memory"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "UR UsmBatchMemoryAllocation, usmMemoryPlacement Device, allocationCount 128, size 16384, measureMode Both",
    "explicit_group": "UsmBatchMemoryAllocation"
  },
  "api_overhead_benchmark_ur UsmBatchMemoryAllocation usmMemoryPlacement:Device allocationCount:128 size:131072 measureMode:Both": {
    "type": "benchmark",
    "description": "Measures memory allocation overhead by allocating 131072 bytes of usm Device memory 128 times, then free'ing it all at once. Both memory allocation and memory free are timed. ",
    "notes": null,
    "unstable": null,
    "tags": [
      "UR",
      "micro",
      "latency",
      "memory"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "UR UsmBatchMemoryAllocation, usmMemoryPlacement Device, allocationCount 128, size 131072, measureMode Both",
    "explicit_group": "UsmBatchMemoryAllocation"
  },
  "multithread_benchmark_syclpreview MemcpyExecute opsPerThread:4096, numThreads:1, allocSize:1024 srcUSM:1 dstUSM:1 without events": {
    "type": "benchmark",
    "description": "Measures multithreaded memory copy performance with 1 threads each performing 4096 operations on 1024 bytes from device to device memory without events with driver copy offload without barrier. ",
    "notes": null,
    "unstable": null,
    "tags": [
      "memory",
      "latency",
      "SYCL",
      "micro"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "UR MemcpyExecute, opsPerThread 4096, numThreads 1, allocSize 1024, srcUSM 1, dstUSM 1, without events",
    "explicit_group": "MemcpyExecute, opsPerThread: 4096, numThreads: 1, allocSize: 1024"
  },
  "multithread_benchmark_syclpreview MemcpyExecute opsPerThread:4096, numThreads:1, allocSize:1024 srcUSM:1 dstUSM:1 without events with barrier": {
    "type": "benchmark",
    "description": "Measures multithreaded memory copy performance with 1 threads each performing 4096 operations on 1024 bytes from device to device memory without events with driver copy offload with barrier. ",
    "notes": null,
    "unstable": null,
    "tags": [
      "memory",
      "latency",
      "SYCL",
      "micro"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "UR MemcpyExecute, opsPerThread 4096, numThreads 1, allocSize 1024, srcUSM 1, dstUSM 1, without events",
    "explicit_group": "MemcpyExecute, opsPerThread: 4096, numThreads: 1, allocSize: 1024"
  },
  "multithread_benchmark_syclpreview MemcpyExecute opsPerThread:4096, numThreads:4, allocSize:1024 srcUSM:1 dstUSM:1 without events": {
    "type": "benchmark",
    "description": "Measures multithreaded memory copy performance with 4 threads each performing 4096 operations on 1024 bytes from device to device memory without events with driver copy offload without barrier. ",
    "notes": null,
    "unstable": null,
    "tags": [
      "memory",
      "latency",
      "SYCL",
      "micro"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "UR MemcpyExecute, opsPerThread 4096, numThreads 4, allocSize 1024, srcUSM 1, dstUSM 1, without events",
    "explicit_group": "MemcpyExecute, opsPerThread: 4096, numThreads: 4, allocSize: 1024"
  },
  "multithread_benchmark_syclpreview MemcpyExecute opsPerThread:4096, numThreads:4, allocSize:1024 srcUSM:1 dstUSM:1 without events with barrier": {
    "type": "benchmark",
    "description": "Measures multithreaded memory copy performance with 4 threads each performing 4096 operations on 1024 bytes from device to device memory without events with driver copy offload with barrier. ",
    "notes": null,
    "unstable": null,
    "tags": [
      "memory",
      "latency",
      "SYCL",
      "micro"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "UR MemcpyExecute, opsPerThread 4096, numThreads 4, allocSize 1024, srcUSM 1, dstUSM 1, without events",
    "explicit_group": "MemcpyExecute, opsPerThread: 4096, numThreads: 4, allocSize: 1024"
  },
  "Velocity-Bench Hashtable": {
    "type": "benchmark",
    "description": "Measures hash table search performance using an efficient lock-free algorithm with linear probing. Reports throughput in millions of keys processed per second. Higher values indicate better performance.",
    "notes": null,
    "unstable": null,
    "tags": [
      "SYCL",
      "application",
      "throughput"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "Velocity-Bench Hashtable",
    "explicit_group": ""
  },
  "Velocity-Bench Bitcracker": {
    "type": "benchmark",
    "description": "Password-cracking application for BitLocker-encrypted memory units. Uses dictionary attack to find user or recovery passwords. Measures total time required to process 60000 passwords.",
    "notes": null,
    "unstable": null,
    "tags": [
      "SYCL",
      "application",
      "throughput"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "Velocity-Bench Bitcracker",
    "explicit_group": ""
  },
  "Velocity-Bench CudaSift": {
    "type": "benchmark",
    "description": "Implementation of the SIFT (Scale Invariant Feature Transform) algorithm for detecting, describing, and matching local features in images. Measures average processing time in milliseconds.",
    "notes": null,
    "unstable": null,
    "tags": [
      "SYCL",
      "application",
      "image"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "Velocity-Bench CudaSift",
    "explicit_group": ""
  },
  "Velocity-Bench Easywave": {
    "type": "benchmark",
    "description": "A tsunami wave simulator used for researching tsunami generation and wave propagation. Measures the elapsed time in milliseconds to simulate a specified tsunami event based on real-world data.",
    "notes": null,
    "unstable": null,
    "tags": [
      "SYCL",
      "application",
      "simulation"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "Velocity-Bench Easywave",
    "explicit_group": ""
  },
  "Velocity-Bench QuickSilver": {
    "type": "benchmark",
    "description": "Solves a simplified dynamic Monte Carlo particle-transport problem used in HPC. Replicates memory access patterns, communication patterns, and branching of Mercury workloads. Reports a figure of merit in MMS/CTT where higher values indicate better performance.",
    "notes": null,
    "unstable": null,
    "tags": [
      "SYCL",
      "application",
      "simulation",
      "throughput"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "Velocity-Bench QuickSilver",
    "explicit_group": ""
  },
  "Velocity-Bench Sobel Filter": {
    "type": "benchmark",
    "description": "Popular RGB-to-grayscale image conversion technique that applies a gaussian filter to reduce edge artifacts. Processes a large 32K x 32K image and measures the time required to apply the filter.",
    "notes": null,
    "unstable": null,
    "tags": [
      "SYCL",
      "application",
      "image",
      "throughput"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "Velocity-Bench Sobel Filter",
    "explicit_group": ""
  },
  "Velocity-Bench dl-cifar": {
    "type": "benchmark",
    "description": "Deep learning image classification workload based on the CIFAR-10 dataset of 60,000 32x32 color images in 10 classes. Uses neural networks to classify input images and measures total calculation time.",
    "notes": null,
    "unstable": null,
    "tags": [
      "SYCL",
      "application",
      "inference",
      "image"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "Velocity-Bench dl-cifar",
    "explicit_group": ""
  },
  "Velocity-Bench dl-mnist": {
    "type": "benchmark",
    "description": "Digit recognition based on the MNIST database, one of the oldest and most popular databases of handwritten digits. Uses neural networks to identify digits and measures total calculation time.",
    "notes": null,
    "unstable": null,
    "tags": [
      "SYCL",
      "application",
      "inference",
      "image"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "Velocity-Bench dl-mnist",
    "explicit_group": ""
  },
  "Velocity-Bench svm": {
    "type": "benchmark",
    "description": "Implementation of Support Vector Machine, a popular classical machine learning technique. Uses supervised learning models with associated algorithms to analyze data for classification and regression analysis. Measures total elapsed time.",
    "notes": null,
    "unstable": null,
    "tags": [
      "SYCL",
      "application",
      "inference"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "Velocity-Bench svm",
    "explicit_group": ""
  },
  "SYCL-Bench IndependentDAGTaskThroughput_multi": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "SYCL",
      "micro"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCL-Bench IndependentDAGTaskThroughput_multi",
    "explicit_group": ""
  },
  "SYCL-Bench DAGTaskThroughput_multi": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "SYCL",
      "micro"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCL-Bench DAGTaskThroughput_multi",
    "explicit_group": ""
  },
  "SYCL-Bench HostDeviceBandwidth_multi": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "SYCL",
      "micro"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCL-Bench HostDeviceBandwidth_multi",
    "explicit_group": ""
  },
  "SYCL-Bench LocalMem_multi": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "SYCL",
      "micro",
      "memory"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCL-Bench LocalMem_multi",
    "explicit_group": ""
  },
  "SYCL-Bench ScalarProduct_multi": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "SYCL",
      "micro"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCL-Bench ScalarProduct_multi",
    "explicit_group": ""
  },
  "SYCL-Bench Pattern_SegmentedReduction_multi": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "SYCL",
      "micro"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCL-Bench Pattern_SegmentedReduction_multi",
    "explicit_group": ""
  },
  "SYCL-Bench USM_Allocation_latency_multi": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "SYCL",
      "micro"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCL-Bench USM_Allocation_latency_multi",
    "explicit_group": ""
  },
  "SYCL-Bench VectorAddition_multi": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "SYCL",
      "micro"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCL-Bench VectorAddition_multi",
    "explicit_group": ""
  },
  "SYCL-Bench 2mm": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "SYCL",
      "micro"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCL-Bench 2mm",
    "explicit_group": ""
  },
  "SYCL-Bench 3mm": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "SYCL",
      "micro"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCL-Bench 3mm",
    "explicit_group": ""
  },
  "SYCL-Bench Atax": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "SYCL",
      "micro"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCL-Bench Atax",
    "explicit_group": ""
  },
  "SYCL-Bench Bicg": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "SYCL",
      "micro"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCL-Bench Bicg",
    "explicit_group": ""
  },
  "SYCL-Bench Kmeans": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "SYCL",
      "micro"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCL-Bench Kmeans",
    "explicit_group": ""
  },
  "SYCL-Bench LinearRegressionCoeff": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "SYCL",
      "micro"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCL-Bench LinearRegressionCoeff",
    "explicit_group": ""
  },
  "SYCL-Bench MolecularDynamics": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "SYCL",
      "micro"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCL-Bench MolecularDynamics",
    "explicit_group": ""
  },
  "SYCL-Bench sf_16": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [
      "SYCL",
      "micro"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "SYCL-Bench sf_16",
    "explicit_group": ""
  },
  "llama.cpp DeepSeek-R1-Distill-Qwen-1.5B-Q4_0.gguf": {
    "type": "benchmark",
    "description": "Performance testing tool for llama.cpp that measures LLM inference speed in tokens per second. Runs both prompt processing (initial context processing) and text generation benchmarks with different batch sizes. Higher values indicate better performance. Uses the DeepSeek-R1-Distill-Qwen-1.5B-Q4_0.gguf quantized model and leverages SYCL with oneDNN for acceleration.",
    "notes": null,
    "unstable": null,
    "tags": [
      "SYCL",
      "application",
      "inference",
      "throughput"
    ],
    "range_min": null,
    "range_max": null,
    "display_name": "llama.cpp DeepSeek-R1-Distill-Qwen-1.5B-Q4_0.gguf",
    "explicit_group": ""
  },
  "umf-benchmark": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [],
    "range_min": null,
    "range_max": null,
    "display_name": "umf-benchmark",
    "explicit_group": ""
  },
  "gromacs-0006-pme-graphs": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [],
    "range_min": null,
    "range_max": null,
    "display_name": "gromacs-0006-pme-graphs",
    "explicit_group": ""
  },
  "gromacs-0006-pme-eager": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [],
    "range_min": null,
    "range_max": null,
    "display_name": "gromacs-0006-pme-eager",
    "explicit_group": ""
  },
  "gromacs-0006-rf-graphs": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [],
    "range_min": null,
    "range_max": null,
    "display_name": "gromacs-0006-rf-graphs",
    "explicit_group": ""
  },
  "gromacs-0006-rf-eager": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [],
    "range_min": null,
    "range_max": null,
    "display_name": "gromacs-0006-rf-eager",
    "explicit_group": ""
  },
  "onednn-sum-f16-1-eager": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [],
    "range_min": null,
    "range_max": null,
    "display_name": "onednn-sum-f16-1-eager",
    "explicit_group": "sum-f16-1"
  },
  "onednn-sum-f16-2-eager": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [],
    "range_min": null,
    "range_max": null,
    "display_name": "onednn-sum-f16-2-eager",
    "explicit_group": "sum-f16-2"
  },
  "onednn-sum-f32-1-eager": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [],
    "range_min": null,
    "range_max": null,
    "display_name": "onednn-sum-f32-1-eager",
    "explicit_group": "sum-f32-1"
  },
  "onednn-sum-f32-2-eager": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [],
    "range_min": null,
    "range_max": null,
    "display_name": "onednn-sum-f32-2-eager",
    "explicit_group": "sum-f32-2"
  },
  "onednn-sum-padding-1-eager": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [],
    "range_min": null,
    "range_max": null,
    "display_name": "onednn-sum-padding-1-eager",
    "explicit_group": "sum-padding-1"
  },
  "onednn-sum-padding-1-graph": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [],
    "range_min": null,
    "range_max": null,
    "display_name": "onednn-sum-padding-1-graph",
    "explicit_group": "sum-padding-1"
  },
  "onednn-sum-padding-2-eager": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [],
    "range_min": null,
    "range_max": null,
    "display_name": "onednn-sum-padding-2-eager",
    "explicit_group": "sum-padding-2"
  },
  "onednn-sum-padding-2-graph": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [],
    "range_min": null,
    "range_max": null,
    "display_name": "onednn-sum-padding-2-graph",
    "explicit_group": "sum-padding-2"
  },
  "onednn-graph-sdpa-plain-f16-eager": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [],
    "range_min": null,
    "range_max": null,
    "display_name": "onednn-graph-sdpa-plain-f16-eager",
    "explicit_group": "graph-sdpa-plain-f16"
  },
  "onednn-graph-sdpa-plain-f16-graph": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [],
    "range_min": null,
    "range_max": null,
    "display_name": "onednn-graph-sdpa-plain-f16-graph",
    "explicit_group": "graph-sdpa-plain-f16"
  },
  "onednn-graph-sdpa-plain-f32-eager": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [],
    "range_min": null,
    "range_max": null,
    "display_name": "onednn-graph-sdpa-plain-f32-eager",
    "explicit_group": "graph-sdpa-plain-f32"
  },
  "onednn-graph-sdpa-plain-f32-graph": {
    "type": "benchmark",
    "description": "",
    "notes": null,
    "unstable": null,
    "tags": [],
    "range_min": null,
    "range_max": null,
    "display_name": "onednn-graph-sdpa-plain-f32-graph",
    "explicit_group": "graph-sdpa-plain-f32"
  },
  "Foo Group": {
    "type": "group",
    "description": "This is a test benchmark for Foo Group.",
    "notes": "This is a test note for Foo Group.\nLook, multiple lines!",
    "unstable": null,
    "tags": [],
    "range_min": null,
    "range_max": null,
    "display_name": null,
    "explicit_group": null
  },
  "Bar Group": {
    "type": "group",
    "description": "This is a test benchmark for Bar Group.",
    "notes": null,
    "unstable": "This is an unstable note for Bar Group.",
    "tags": [],
    "range_min": null,
    "range_max": null,
    "display_name": null,
    "explicit_group": null
  },
  "Memory Bandwidth 1": {
    "type": "benchmark",
    "description": "This is a test benchmark for Memory Bandwidth 1.",
    "notes": null,
    "unstable": null,
    "tags": [],
    "range_min": null,
    "range_max": null,
    "display_name": "Memory Bandwidth 1",
    "explicit_group": ""
  },
  "Memory Bandwidth 2": {
    "type": "benchmark",
    "description": "This is a test benchmark for Memory Bandwidth 2.",
    "notes": null,
    "unstable": null,
    "tags": [],
    "range_min": null,
    "range_max": null,
    "display_name": "Memory Bandwidth 2",
    "explicit_group": ""
  },
  "Memory Bandwidth 3": {
    "type": "benchmark",
    "description": "This is a test benchmark for Memory Bandwidth 3.",
    "notes": null,
    "unstable": null,
    "tags": [],
    "range_min": null,
    "range_max": null,
    "display_name": "Memory Bandwidth 3",
    "explicit_group": ""
  },
  "Memory Bandwidth 4": {
    "type": "benchmark",
    "description": "This is a test benchmark for Memory Bandwidth 4.",
    "notes": null,
    "unstable": null,
    "tags": [],
    "range_min": null,
    "range_max": null,
    "display_name": "Memory Bandwidth 4",
    "explicit_group": ""
  },
  "Memory Bandwidth 5": {
    "type": "benchmark",
    "description": "This is a test benchmark for Memory Bandwidth 5.",
    "notes": null,
    "unstable": null,
    "tags": [],
    "range_min": null,
    "range_max": null,
    "display_name": "Memory Bandwidth 5",
    "explicit_group": ""
  },
  "Memory Bandwidth 6": {
    "type": "benchmark",
    "description": "This is a test benchmark for Memory Bandwidth 6.",
    "notes": null,
    "unstable": null,
    "tags": [],
    "range_min": null,
    "range_max": null,
    "display_name": "Memory Bandwidth 6",
    "explicit_group": ""
  },
  "Latency 1": {
    "type": "benchmark",
    "description": "This is a test benchmark for Latency 1.",
    "notes": "A Latency test note!",
    "unstable": null,
    "tags": [],
    "range_min": null,
    "range_max": null,
    "display_name": "Latency 1",
    "explicit_group": ""
  },
  "Latency 2": {
    "type": "benchmark",
    "description": "This is a test benchmark for Latency 2.",
    "notes": "A Latency test note!",
    "unstable": null,
    "tags": [],
    "range_min": null,
    "range_max": null,
    "display_name": "Latency 2",
    "explicit_group": ""
  },
  "Latency 3": {
    "type": "benchmark",
    "description": "This is a test benchmark for Latency 3.",
    "notes": "A Latency test note!",
    "unstable": null,
    "tags": [],
    "range_min": null,
    "range_max": null,
    "display_name": "Latency 3",
    "explicit_group": ""
  },
  "Latency 4": {
    "type": "benchmark",
    "description": "This is a test benchmark for Latency 4.",
    "notes": "A Latency test note!",
    "unstable": null,
    "tags": [],
    "range_min": null,
    "range_max": null,
    "display_name": "Latency 4",
    "explicit_group": ""
  },
  "Latency 5": {
    "type": "benchmark",
    "description": "This is a test benchmark for Latency 5.",
    "notes": "A Latency test note!",
    "unstable": null,
    "tags": [],
    "range_min": null,
    "range_max": null,
    "display_name": "Latency 5",
    "explicit_group": ""
  },
  "Latency 6": {
    "type": "benchmark",
    "description": "This is a test benchmark for Latency 6.",
    "notes": "A Latency test note!",
    "unstable": null,
    "tags": [],
    "range_min": null,
    "range_max": null,
    "display_name": "Latency 6",
    "explicit_group": ""
  },
  "Throughput 1": {
    "type": "benchmark",
    "description": "This is a test benchmark for Throughput 1.",
    "notes": null,
    "unstable": null,
    "tags": [],
    "range_min": null,
    "range_max": null,
    "display_name": "Throughput 1",
    "explicit_group": ""
  },
  "Throughput 2": {
    "type": "benchmark",
    "description": "This is a test benchmark for Throughput 2.",
    "notes": null,
    "unstable": null,
    "tags": [],
    "range_min": null,
    "range_max": null,
    "display_name": "Throughput 2",
    "explicit_group": ""
  },
  "Throughput 3": {
    "type": "benchmark",
    "description": "This is a test benchmark for Throughput 3.",
    "notes": null,
    "unstable": null,
    "tags": [],
    "range_min": null,
    "range_max": null,
    "display_name": "Throughput 3",
    "explicit_group": ""
  },
  "Throughput 4": {
    "type": "benchmark",
    "description": "This is a test benchmark for Throughput 4.",
    "notes": null,
    "unstable": null,
    "tags": [],
    "range_min": null,
    "range_max": null,
    "display_name": "Throughput 4",
    "explicit_group": ""
  },
  "Throughput 5": {
    "type": "benchmark",
    "description": "This is a test benchmark for Throughput 5.",
    "notes": null,
    "unstable": null,
    "tags": [],
    "range_min": null,
    "range_max": null,
    "display_name": "Throughput 5",
    "explicit_group": ""
  },
  "Throughput 6": {
    "type": "benchmark",
    "description": "This is a test benchmark for Throughput 6.",
    "notes": null,
    "unstable": null,
    "tags": [],
    "range_min": null,
    "range_max": null,
    "display_name": "Throughput 6",
    "explicit_group": ""
  },
  "FLOPS 1": {
    "type": "benchmark",
    "description": "This is a test benchmark for FLOPS 1.",
    "notes": null,
    "unstable": "Unstable FLOPS test!",
    "tags": [],
    "range_min": null,
    "range_max": null,
    "display_name": "FLOPS 1",
    "explicit_group": ""
  },
  "FLOPS 2": {
    "type": "benchmark",
    "description": "This is a test benchmark for FLOPS 2.",
    "notes": null,
    "unstable": "Unstable FLOPS test!",
    "tags": [],
    "range_min": null,
    "range_max": null,
    "display_name": "FLOPS 2",
    "explicit_group": ""
  },
  "FLOPS 3": {
    "type": "benchmark",
    "description": "This is a test benchmark for FLOPS 3.",
    "notes": null,
    "unstable": "Unstable FLOPS test!",
    "tags": [],
    "range_min": null,
    "range_max": null,
    "display_name": "FLOPS 3",
    "explicit_group": ""
  },
  "FLOPS 4": {
    "type": "benchmark",
    "description": "This is a test benchmark for FLOPS 4.",
    "notes": null,
    "unstable": "Unstable FLOPS test!",
    "tags": [],
    "range_min": null,
    "range_max": null,
    "display_name": "FLOPS 4",
    "explicit_group": ""
  },
  "FLOPS 5": {
    "type": "benchmark",
    "description": "This is a test benchmark for FLOPS 5.",
    "notes": null,
    "unstable": "Unstable FLOPS test!",
    "tags": [],
    "range_min": null,
    "range_max": null,
    "display_name": "FLOPS 5",
    "explicit_group": ""
  },
  "FLOPS 6": {
    "type": "benchmark",
    "description": "This is a test benchmark for FLOPS 6.",
    "notes": null,
    "unstable": "Unstable FLOPS test!",
    "tags": [],
    "range_min": null,
    "range_max": null,
    "display_name": "FLOPS 6",
    "explicit_group": ""
  },
  "Cache Miss Rate 1": {
    "type": "benchmark",
    "description": "This is a test benchmark for Cache Miss Rate 1.",
    "notes": "Test Note",
    "unstable": "And another note!",
    "tags": [],
    "range_min": null,
    "range_max": null,
    "display_name": "Cache Miss Rate 1",
    "explicit_group": ""
  },
  "Cache Miss Rate 2": {
    "type": "benchmark",
    "description": "This is a test benchmark for Cache Miss Rate 2.",
    "notes": "Test Note",
    "unstable": "And another note!",
    "tags": [],
    "range_min": null,
    "range_max": null,
    "display_name": "Cache Miss Rate 2",
    "explicit_group": ""
  },
  "Cache Miss Rate 3": {
    "type": "benchmark",
    "description": "This is a test benchmark for Cache Miss Rate 3.",
    "notes": "Test Note",
    "unstable": "And another note!",
    "tags": [],
    "range_min": null,
    "range_max": null,
    "display_name": "Cache Miss Rate 3",
    "explicit_group": ""
  },
  "Cache Miss Rate 4": {
    "type": "benchmark",
    "description": "This is a test benchmark for Cache Miss Rate 4.",
    "notes": "Test Note",
    "unstable": "And another note!",
    "tags": [],
    "range_min": null,
    "range_max": null,
    "display_name": "Cache Miss Rate 4",
    "explicit_group": ""
  },
  "Cache Miss Rate 5": {
    "type": "benchmark",
    "description": "This is a test benchmark for Cache Miss Rate 5.",
    "notes": "Test Note",
    "unstable": "And another note!",
    "tags": [],
    "range_min": null,
    "range_max": null,
    "display_name": "Cache Miss Rate 5",
    "explicit_group": ""
  },
  "Cache Miss Rate 6": {
    "type": "benchmark",
    "description": "This is a test benchmark for Cache Miss Rate 6.",
    "notes": "Test Note",
    "unstable": "And another note!",
    "tags": [],
    "range_min": null,
    "range_max": null,
    "display_name": "Cache Miss Rate 6",
    "explicit_group": ""
  }
};

benchmarkTags = {
  "SYCL": {
    "name": "SYCL",
    "description": "Benchmark uses SYCL runtime"
  },
  "UR": {
    "name": "UR",
    "description": "Benchmark uses Unified Runtime API"
  },
  "L0": {
    "name": "L0",
    "description": "Benchmark uses Level Zero API directly"
  },
  "UMF": {
    "name": "UMF",
    "description": "Benchmark uses Unified Memory Framework directly"
  },
  "micro": {
    "name": "micro",
    "description": "Microbenchmark focusing on a specific functionality"
  },
  "application": {
    "name": "application",
    "description": "Real application-based performance test"
  },
  "proxy": {
    "name": "proxy",
    "description": "Benchmark that simulates real application use-cases"
  },
  "submit": {
    "name": "submit",
    "description": "Tests kernel submission performance"
  },
  "math": {
    "name": "math",
    "description": "Tests math computation performance"
  },
  "memory": {
    "name": "memory",
    "description": "Tests memory transfer or bandwidth performance"
  },
  "allocation": {
    "name": "allocation",
    "description": "Tests memory allocation performance"
  },
  "graph": {
    "name": "graph",
    "description": "Tests graph-based execution performance"
  },
  "latency": {
    "name": "latency",
    "description": "Measures operation latency"
  },
  "throughput": {
    "name": "throughput",
    "description": "Measures operation throughput"
  },
  "inference": {
    "name": "inference",
    "description": "Tests ML/AI inference performance"
  },
  "image": {
    "name": "image",
    "description": "Image processing benchmark"
  },
  "simulation": {
    "name": "simulation",
    "description": "Physics or scientific simulation benchmark"
  }
};

defaultCompareNames = [
  "This PR"
];
