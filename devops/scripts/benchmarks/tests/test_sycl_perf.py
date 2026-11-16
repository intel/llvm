# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Unit tests for benches/sycl_perf.py (no compiler or device needed)."""

import json
import os
import shutil
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest import mock

os.environ.setdefault("LLVM_BENCHMARKS_UNIT_TESTING", "1")
sys.path.append(f"{os.path.dirname(__file__)}/../")
from options import options
from benches import sycl_perf
from benches.sycl_perf import (
    RunLineError,
    SyclCompileBench,
    SyclPerfRuntimeBench,
    SyclPerfSuite,
    available_features,
    drop_if_segments,
    eval_feature_expr,
    parse_perf_results,
    read_perf_test,
)

SYCL = "/opt/sycl"


def perf_line(name, variant="", unit="ns", value=1.5, **extra):
    fields = dict(
        name=name,
        variant=variant,
        unit=unit,
        value=value,
        stddev=0.1,
        min=1.0,
        reps=10,
        lower_is_better=True,
    )
    fields.update(extra)
    return "PERF_RESULT " + json.dumps(fields)


class TempTests(unittest.TestCase):
    """Creates a tests dir and a workdir; restores the global options."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.tests = self.tmp / "PerformanceTests"
        self.tests.mkdir()
        self.saved = (
            options.workdir,
            options.sycl,
            getattr(options, "device_architecture", None),
            options.sycl_perf_tests_dir,
        )
        options.workdir = str(self.tmp / "wd")
        options.sycl = SYCL
        options.device_architecture = ""
        options.sycl_perf_tests_dir = str(self.tests)

    def tearDown(self):
        (
            options.workdir,
            options.sycl,
            options.device_architecture,
            options.sycl_perf_tests_dir,
        ) = self.saved
        shutil.rmtree(self.tmp, ignore_errors=True)

    def write(self, rel: str, text: str) -> Path:
        path = self.tests / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        text = textwrap.dedent(text)
        path.write_text(text[1:] if text.startswith("\n") else text)
        return path

    def read(self, rel: str, text: str):
        return read_perf_test(self.write(rel, text), rel[: -len(".cpp")])

    def assertRunLineError(self, rel: str, text: str, line: int, msg: str):
        with self.assertRaises(RunLineError) as cm:
            self.read(rel, text)
        self.assertIn(f"{self.tests / rel}:{line}:", str(cm.exception))
        self.assertIn(msg, str(cm.exception))


class TestRunLines(TempTests):
    def test_build_and_run(self):
        t = self.read(
            "a.cpp",
            """
            // RUN: %{build} -O3 -DX=1 -o %t.out
            // RUN: %{run} %t.out
            // PERF-DESCRIPTION: Something fast.
            int main() {}
            """,
        )
        self.assertEqual(t.description, "Something fast.")
        self.assertEqual(len(t.builds), 1)
        self.assertEqual(t.builds[0].sfx, "")
        self.assertEqual(len(t.runs), 1)
        self.assertIs(t.runs[0].build, t.builds[0])
        self.assertEqual(t.runs[0].variant, "")
        self.assertEqual(t.compiles, [])

    def test_build_without_flags(self):
        t = self.read("a.cpp", "// RUN: %{build} -o %t.out\n// RUN: %{run} %t.out\n")
        self.assertEqual(t.builds[0].command, "%{build} -o %t.out")

    def test_suffixes_bind_to_their_build(self):
        t = self.read(
            "dir/b.cpp",
            """
            // RUN: %{build} -DA -o %t1.out
            // RUN: %{run} %t1.out --variant="mode=a"
            // RUN: %{build} -DB -o %tb_2.x-y.out
            // RUN: %{run} %tb_2.x-y.out --variant="mode b" --n=3
            """,
        )
        self.assertEqual([b.sfx for b in t.builds], ["1", "b_2.x-y"])
        self.assertEqual([r.build.line for r in t.runs], [1, 3])
        self.assertEqual([r.variant for r in t.runs], ["mode=a", "mode b"])

    def test_continuation(self):
        t = self.read(
            "a.cpp",
            """
            // RUN: %{build} -O3 \\
            // RUN:   -DX=1 \\
            // RUN:   -o %t.out
            // RUN: %{run} %t.out \\
            // RUN:   --variant="v 1"
            """,
        )
        self.assertEqual(t.builds[0].line, 1)
        self.assertEqual(t.runs[0].line, 4)
        self.assertEqual(t.runs[0].variant, "v 1")
        expanded = SyclPerfSuite().substitutions(t).expand(t.builds[0].command, 1)
        self.assertEqual(expanded[-4:], ["-O3", "-DX=1", "-o", expanded[-1]])

    def test_dangling_continuation(self):
        self.assertRunLineError(
            "a.cpp", "// RUN: %{build} -o %t.out \\\nint x;\n", 1, "continuation"
        )
        self.assertRunLineError(
            "a.cpp", "// RUN: %{build} -o %t.out \\\n", 1, "continuation"
        )

    def test_if_segment_dropped(self):
        t = self.read(
            "a.cpp",
            """
            // RUN: %{build} -o %t.out
            // RUN: %if hip %{ env X=%{amd_arch} %} %{run} %t.out --pch-dir=%/t.dir
            // RUN: %if (cuda || hip) && !linux %{ %{build} -DCUDA -o %t.out %}
            """,
        )
        self.assertEqual(len(t.builds), 1)
        self.assertEqual(t.runs[0].args, "--pch-dir=%/t.dir")
        self.assertEqual(
            drop_if_segments('a %if x %{ b %{c} %} d "e  f"'), 'a  d "e  f"'
        )

    def test_if_else_rejected(self):
        self.assertRunLineError(
            "a.cpp",
            "// RUN: %if hip %{ a %} %else %{ b %} %{build} -o %t.out\n",
            1,
            "%else",
        )

    def test_compile_line(self):
        t = self.read(
            "CompileTime/c.cpp",
            """
            // RUN: python3 %S/../perf_compile.py --name "c.hpp, device syntax-only" -- %{build} -fsycl-device-only -fsyntax-only
            // RUN: python3 %S/../perf_compile.py --name "c.hpp, full build" --reps 3 --warmup=1 -- %{build} -o %t.out
            // RUN: python3 %S/perf_compile.py --name=plain -- %{build}
            """,
        )
        self.assertEqual(
            [c.name for c in t.compiles],
            ["c.hpp, device syntax-only", "c.hpp, full build", "plain"],
        )
        self.assertEqual(t.compiles[0].script, "../perf_compile.py")
        self.assertEqual(t.compiles[2].script, "perf_compile.py")
        self.assertEqual(t.runs, [])

    def test_rejected_forms(self):
        cases = [
            ('// RUN: echo "hi"\n', 1, "unsupported RUN line"),
            ("// RUN: %{build} -o %t.exe\n", 1, "unsupported RUN line"),
            ("// RUN: %{build} -o %t.out -O3\n", 1, "unsupported RUN line"),
            ("// RUN: %{run-aux} %t.out\n", 1, "unsupported RUN line"),
            ("// RUN: env X=1 %{run} %t.out\n", 1, "unsupported RUN line"),
            ("// RUN: %clangxx -fsycl %s -o %t.out\n", 1, "unsupported RUN line"),
            ("int x;\n// RUN: %{run} %t.out\n", 2, "not built"),
            (
                "// RUN: %{build} -o %t.out\n// RUN: %{build} -DX -o %t.out\n",
                2,
                "already built differently",
            ),
            ("// RUN: python3 %S/perf_compile.py -- %{build}\n", 1, "no --name"),
            (
                "// RUN: python3 %S/perf_compile.py --name x --bad 1 -- %{build}\n",
                1,
                "unsupported perf_compile.py option",
            ),
            (
                "// RUN: python3 %S/perf_compile.py --name x --reps many -- %{build}\n",
                1,
                "needs an integer",
            ),
            (
                '// RUN: python3 %S/perf_compile.py --name "a|b" -- %{build}\n',
                1,
                "invalid characters",
            ),
            (
                '// RUN: %{build} -o %t.out\n// RUN: %{run} %t.out --variant="a|b"\n',
                2,
                "invalid characters",
            ),
            ("// DEFINE: %{x}=y\n", 1, "not supported"),
            ("// REQUIRES-INTEL-DRIVER: lin: 1\n", 1, "not supported"),
            ("/* RUN: %{build} -o %t.out */\n", 1, "must start with"),
        ]
        for text, line, msg in cases:
            with self.subTest(text=text):
                self.assertRunLineError("bad.cpp", text, line, msg)

    def test_non_directive_comments_ignored(self):
        t = self.read(
            "a.cpp",
            """
            // RUN: %{build} -o %t.out
            // RUN: %{run} %t.out
            // NOTE: this is text, as is TODO: below
            // TODO: something
            // XFAIL: gpu-intel-gen12
            // XFAIL-TRACKER: https://example.com/1
            // UNSUPPORTED-TRACKER: https://example.com/2
            """,
        )
        self.assertEqual(t.xfail, ["gpu-intel-gen12"])


class TestSubstitutions(TempTests):
    def test_paths(self):
        path = self.write(
            "Dir/x.cpp",
            """
            // RUN: %{build} -I%S/inc -o %t.out
            // RUN: %{run} %t.out "--msg=%s" --pch-dir=%/t.dir --variant="a b"
            """,
        )
        t = read_perf_test(path, "Dir/x")
        subst = SyclPerfSuite().substitutions(t)
        src = str(path.resolve())
        tmp = Path(options.workdir).resolve() / "sycl-perf" / "Dir" / "x" / "t"
        self.assertEqual(
            subst.expand(t.builds[0].command, 1),
            [
                f"{SYCL}/bin/clang++",
                "-Werror",
                "-fsycl",
                "-fsycl-targets=spir64",
                src,
                f"-I{path.parent.resolve()}/inc",
                "-o",
                f"{tmp}.out",
            ],
        )
        self.assertEqual(
            subst.expand(t.runs[0].args, 2),
            [f"--msg={src}", f"--pch-dir={tmp}.dir", "--variant=a b"],
        )

    def test_unknown_substitution(self):
        path = self.write("x.cpp", "// RUN: %{build} -D%{foo} -o %t.out\n")
        t = read_perf_test(path, "x")
        with self.assertRaises(RunLineError) as cm:
            SyclPerfSuite().substitutions(t).expand(t.builds[0].command, 1)
        self.assertIn(f"{path}:1:", str(cm.exception))
        self.assertIn("%{foo}", str(cm.exception))

        with self.assertRaises(RunLineError):
            SyclPerfSuite().substitutions(t).expand("%p", 1)

    def test_unsafe_paths_rejected(self):
        path = self.write("x.cpp", "// RUN: %{build} -o %t.out\n")
        t = read_perf_test(path, "x")
        options.workdir = str(self.tmp / "with space")
        with self.assertRaises(ValueError):
            SyclPerfSuite().substitutions(t)


class TestFeatures(unittest.TestCase):
    def test_expressions(self):
        f = {"linux", "gpu", "level_zero"}
        cases = {
            "linux": True,
            "windows": False,
            "!windows": True,
            "linux && gpu": True,
            "linux && cuda": False,
            "cuda || level_zero": True,
            "!(cuda || hip) && gpu": True,
            "!!linux": True,
            "true": True,
            "false || !true": False,
            "gpu-intel-pvc || arch-intel_gpu_bmg_g21": False,
        }
        for expr, expected in cases.items():
            with self.subTest(expr=expr):
                self.assertEqual(eval_feature_expr(expr, f), expected)

    def test_malformed(self):
        for expr in ["", "linux &&", "(linux", "linux)", "a b", "{{.*}}", "&& a"]:
            with self.subTest(expr=expr):
                with self.assertRaises(ValueError):
                    eval_feature_expr(expr, set())

    def test_available_features(self):
        self.assertEqual(
            available_features(""),
            {"linux", "gpu", "level_zero", "enable-perf-tests"},
        )
        self.assertIn("gpu-intel-dg2", available_features("intel_gpu_acm_g10"))
        self.assertIn("gpu-intel-pvc", available_features("intel_gpu_pvc"))
        self.assertIn("arch-intel_gpu_bmg_g21", available_features("intel_gpu_bmg_g21"))


class TestRequiresUnsupported(TempTests):
    def skip(self, directives: str, arch: str = ""):
        t = self.read("t.cpp", directives + "// RUN: %{build} -o %t.out\n")
        return t.skip_reason(available_features(arch))

    def test_requires(self):
        self.assertIsNone(self.skip("// REQUIRES: linux, gpu\n"))
        self.assertIsNone(self.skip("// REQUIRES: linux\n// REQUIRES: level_zero\n"))
        self.assertIsNone(self.skip("// REQUIRES: enable-perf-tests && !cuda\n"))
        self.assertIn("REQUIRES", self.skip("// REQUIRES: linux, cuda\n"))
        self.assertIn("REQUIRES", self.skip("// REQUIRES: aspect-fp64\n"))

    def test_unsupported(self):
        self.assertIsNone(self.skip("// UNSUPPORTED: windows, target-native_cpu\n"))
        self.assertIn("UNSUPPORTED", self.skip("// UNSUPPORTED: windows, linux\n"))
        self.assertIsNone(self.skip("// UNSUPPORTED: gpu-intel-gen12\n"))
        self.assertIn(
            "UNSUPPORTED",
            self.skip("// UNSUPPORTED: gpu-intel-gen12\n", "intel_gpu_tgllp"),
        )

    def test_bad_expression_reports_line(self):
        t = self.read("t.cpp", "int x;\n// UNSUPPORTED: linux &&\n")
        with self.assertRaises(RunLineError) as cm:
            t.skip_reason(available_features())
        self.assertIn("t.cpp:2:", str(cm.exception))


class TestPerfResults(unittest.TestCase):
    def test_valid(self):
        out = "\n".join(
            [
                "some noise",
                perf_line("has(fp64)", value=12.5),
                "  PERF_RESULT not at line start is ignored",
                "PERF_RESULTS also ignored",
                perf_line("x, y", variant="rounding=force", unit="us", value=3),
            ]
        )
        res = parse_perf_results(out)
        self.assertEqual([r.name for r in res], ["has(fp64)", "x, y"])
        self.assertEqual(res[0].value, 12.5)
        self.assertEqual(res[0].stddev, 0.1)
        self.assertEqual(res[1].variant, "rounding=force")
        self.assertEqual(res[1].unit, "us")
        self.assertEqual(res[1].reps, 10)

    def test_none(self):
        with self.assertRaises(ValueError):
            parse_perf_results("nothing here\n")
        with self.assertRaises(ValueError):
            parse_perf_results("")

    def test_malformed(self):
        bad = [
            "PERF_RESULT {not json",
            "PERF_RESULT [1, 2]",
            perf_line("a", value="1"),
            perf_line("a", reps=1.5),
            perf_line("a", lower_is_better=1),
            perf_line("a|b"),
            perf_line(""),
            perf_line("a", variant='q"uote'),
            perf_line("a").replace(', "min": 1.0', ""),
        ]
        for line in bad:
            with self.subTest(line=line):
                with self.assertRaises(ValueError):
                    parse_perf_results(line)


class TestSuite(TempTests):
    def make_tests(self):
        self.write(
            "aspect_queries.cpp",
            """
            // RUN: %{build} -O3 -o %t.out
            // RUN: %{run} %t.out
            // PERF-DESCRIPTION: Aspect queries.
            """,
        )
        self.write(
            "ParallelFor/roundup.cpp",
            """
            // RUN: %{build} -fsycl-range-rounding=disable -o %t1.out
            // RUN: %{run} %t1.out --variant="rounding=disable"
            // RUN: %{build} -fsycl-range-rounding=force -o %t2.out
            // RUN: %{run} %t2.out --variant="rounding=force"
            """,
        )
        self.write(
            "KernelCompiler/pch.cpp",
            """
            // RUN: %{build} -o %t.out
            // RUN: %{run} %t.out --pch-dir=%/t.dir
            // UNSUPPORTED: target-native_cpu
            """,
        )
        self.write(
            "CompileTime/core_hpp.cpp",
            """
            // RUN: python3 %S/../perf_compile.py --name "core.hpp, device syntax-only" -- %{build} -fsycl-device-only -fsyntax-only
            // PERF-DESCRIPTION: Compile time of core.hpp.
            """,
        )
        self.write(
            "Skipped/cuda.cpp",
            "// REQUIRES: cuda\n// RUN: %{build} -o %t.out\n// RUN: %{run} %t.out\n",
        )
        self.write("perf_harness.hpp", "// RUN: not a test\n")

    def test_discovery_and_metadata(self):
        self.make_tests()
        suite = SyclPerfSuite()
        self.assertEqual(suite.name(), "SYCL Performance Tests")
        benches = {b.name(): b for b in suite.benchmarks()}
        self.assertEqual(
            sorted(benches),
            [
                "sycl_compile core.hpp, device syntax-only",
                "sycl_perf KernelCompiler/pch",
                "sycl_perf ParallelFor/roundup rounding=disable",
                "sycl_perf ParallelFor/roundup rounding=force",
                "sycl_perf aspect_queries",
            ],
        )
        rt = benches["sycl_perf aspect_queries"]
        self.assertIsInstance(rt, SyclPerfRuntimeBench)
        self.assertEqual(rt.explicit_group(), "aspect_queries")
        self.assertEqual(rt.get_tags(), ["SYCL", "micro", "latency"])
        self.assertEqual(rt.description(), "Aspect queries.")
        self.assertIsNone(rt.stddev_threshold())
        self.assertEqual(
            benches["sycl_perf KernelCompiler/pch"].stddev_threshold(), 0.05
        )

        cb = benches["sycl_compile core.hpp, device syntax-only"]
        self.assertIsInstance(cb, SyclCompileBench)
        self.assertEqual(cb.explicit_group(), "CompileTime/core_hpp")
        self.assertEqual(cb.get_tags(), ["SYCL", "compile"])
        self.assertEqual(cb.stddev_threshold(), 0.05)
        self.assertFalse(cb.traceable())
        self.assertEqual(cb.script(), (self.tests / "perf_compile.py").resolve())

        groups = suite.additional_metadata()
        self.assertEqual(
            sorted(groups),
            [
                "CompileTime/core_hpp",
                "KernelCompiler/pch",
                "ParallelFor/roundup",
                "aspect_queries",
            ],
        )
        self.assertTrue(all(g.type == "group" for g in groups.values()))
        self.assertEqual(groups["CompileTime/core_hpp"].tags, ["SYCL", "compile"])
        meta = rt.get_metadata()["sycl_perf aspect_queries"]
        self.assertEqual(meta.explicit_group, "aspect_queries")

    def test_parse_errors_fail_setup_only(self):
        self.make_tests()
        self.write("bad.cpp", 'int a;\nint b;\n// RUN: echo "x"\n')
        suite = SyclPerfSuite()
        self.assertEqual(len(suite.benchmarks()), 5)  # bad.cpp is left out
        with self.assertRaises(RuntimeError) as cm:
            suite.setup()
        self.assertIn("bad.cpp:3:", str(cm.exception))

    def test_duplicate_names(self):
        self.write(
            "a.cpp",
            "// RUN: %{build} -o %t.out\n// RUN: %{run} %t.out\n// RUN: %{run} %t.out --n=2\n",
        )
        suite = SyclPerfSuite()
        suite.benchmarks()
        self.assertTrue(any("duplicate benchmark name" in e for e in suite._errors))

    def test_missing_dir(self):
        options.sycl_perf_tests_dir = str(self.tmp / "missing")
        suite = SyclPerfSuite()
        self.assertEqual(suite.benchmarks(), [])
        self.assertEqual(suite.additional_metadata(), {})

    def test_default_dir(self):
        options.sycl_perf_tests_dir = None
        expected = Path(sycl_perf.__file__).resolve().parents[4]
        self.assertEqual(
            SyclPerfSuite().tests_dir,
            expected / "sycl" / "test-e2e" / "PerformanceTests",
        )
        self.assertTrue((expected / "devops" / "scripts" / "benchmarks").is_dir())

    def test_runtime_results(self):
        self.make_tests()
        suite = SyclPerfSuite()
        bench = next(
            b for b in suite.benchmarks() if b.name().endswith("rounding=force")
        )
        out = "\n".join(
            [
                "noise",
                perf_line("1000x1000", variant="rounding=force", value=7.0),
                perf_line("10x10", variant="rounding=force", value=2.0, unit="us"),
            ]
        )
        with mock.patch.object(
            SyclPerfRuntimeBench, "run_bench", return_value=out
        ) as rb:
            results = bench.run({"ZE_AFFINITY_MASK": "0"})
        command = rb.call_args[0][0]
        self.assertTrue(command[0].endswith("/sycl-perf/ParallelFor/roundup/t2.out"))
        self.assertEqual(command[1:], ["--variant=rounding=force"])
        self.assertEqual(
            [r.label for r in results],
            [
                "sycl_perf ParallelFor/roundup rounding=force 1000x1000",
                "sycl_perf ParallelFor/roundup rounding=force 10x10",
            ],
        )
        self.assertEqual([r.unit for r in results], ["ns", "us"])
        self.assertEqual(results[0].stddev, 0.1)
        self.assertEqual(results[0].git_url, "https://github.com/intel/llvm.git")
        self.assertEqual(results[0].env, {"ZE_AFFINITY_MASK": "0"})

        with mock.patch.object(SyclPerfRuntimeBench, "run_bench", return_value="noise"):
            with self.assertRaises(ValueError):
                bench.run({})

    def test_compile_results(self):
        self.make_tests()
        suite = SyclPerfSuite()
        bench = next(b for b in suite.benchmarks() if isinstance(b, SyclCompileBench))
        out = "\n".join(
            [
                perf_line("core.hpp, device syntax-only", unit="s", value=1.25),
                perf_line(
                    "core.hpp, device syntax-only, CPU time", unit="s", value=1.5
                ),
            ]
        )
        completed = mock.Mock()
        completed.stdout = out.encode()
        with mock.patch.dict(os.environ, {"CPATH": "/x", "SYCL_CACHE_DIR": "/y"}):
            with mock.patch.object(sycl_perf, "run", return_value=completed) as run:
                results = bench.run({"SYCL_UR_TRACE": "1", "ZE_AFFINITY_MASK": "0"})
        command = run.call_args[0][0]
        env_at = command.index("env")
        self.assertEqual(command[:env_at][0], "taskset")
        scrub = command[env_at:]
        for name in ["CPATH", "SYCL_CACHE_DIR", "SYCL_UR_TRACE"]:
            self.assertIn(name, scrub[: scrub.index("python3")])
        tail = scrub[scrub.index("python3") :]
        self.assertEqual(
            tail[1:9],
            [
                str((self.tests / "perf_compile.py").resolve()),
                "--name",
                "core.hpp, device syntax-only",
                "--warmup",
                "0",
                "--reps",
                "1",
                "--",
            ],
        )
        self.assertEqual(tail[9], f"{SYCL}/bin/clang++")
        self.assertEqual(tail[-2:], ["-fsycl-device-only", "-fsyntax-only"])
        self.assertEqual(
            [(r.label, r.value, r.unit) for r in results],
            [
                ("sycl_compile core.hpp, device syntax-only", 1.25, "s"),
                ("sycl_compile core.hpp, device syntax-only, CPU time", 1.5, "s"),
            ],
        )
        self.assertEqual(results[0].env, {"ZE_AFFINITY_MASK": "0"})

        completed.stdout = perf_line("core.hpp, device syntax-only", unit="s").encode()
        with mock.patch.object(sycl_perf, "run", return_value=completed):
            with self.assertRaises(ValueError):
                bench.run({})

    def test_build_once_and_up_to_date(self):
        self.make_tests()
        sycl = self.tmp / "sycl"
        (sycl / "lib").mkdir(parents=True)
        (sycl / "lib" / "libsycl.so").write_text("")
        options.sycl = str(sycl)
        suite = SyclPerfSuite()
        bench = next(
            b for b in suite.benchmarks() if b.name() == "sycl_perf aspect_queries"
        )
        binary = Path(f"{suite.tmp_prefix(bench.test)}.out")

        def fake_build(cmd, **kwargs):
            binary.parent.mkdir(parents=True, exist_ok=True)
            binary.write_text("bin")

        with mock.patch.object(sycl_perf, "run", side_effect=fake_build) as run:
            self.assertEqual(suite.build(bench.test, bench.step.build), binary)
            self.assertEqual(suite.build(bench.test, bench.step.build), binary)
        self.assertEqual(run.call_count, 1)
        self.assertEqual(run.call_args[0][0][0], "env")
        self.assertTrue(run.call_args[1]["add_sycl"])

        # A new suite object (next main.py run) reuses the up-to-date binary.
        os.utime(binary, (binary.stat().st_mtime + 10,) * 2)
        suite2 = SyclPerfSuite()
        with mock.patch.object(sycl_perf, "run", side_effect=fake_build) as run:
            suite2.build(bench.test, bench.step.build)
        self.assertEqual(run.call_count, 0)

        # A newer libsycl.so triggers a rebuild.
        os.utime(sycl / "lib" / "libsycl.so", (binary.stat().st_mtime + 10,) * 2)
        suite3 = SyclPerfSuite()
        with mock.patch.object(sycl_perf, "run", side_effect=fake_build) as run:
            suite3.build(bench.test, bench.step.build)
        self.assertEqual(run.call_count, 1)


if __name__ == "__main__":
    unittest.main()
