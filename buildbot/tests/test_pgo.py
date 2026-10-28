import argparse
import importlib.util
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

PGO_SCRIPT = Path(__file__).resolve().parents[1] / "pgo.py"
SPEC = importlib.util.spec_from_file_location("dpcpp_pgo", PGO_SCRIPT)
PGO = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PGO)


class PgoCommandTest(unittest.TestCase):
    def test_instrumented_configuration_uses_isolated_profile_directory(self):
        command = PGO._configure_command(
            Path("configure.py"),
            Path("src"),
            Path("out/instrumented"),
            "Ninja",
            [
                "LLVM_BUILD_INSTRUMENTED=IR",
                "LLVM_PROFILE_DATA_DIR=out/instrumented/profiles",
                f"CMAKE_C_FLAGS={PGO.INSTRUMENTED_PROFILE_FLAGS}",
                f"CMAKE_CXX_FLAGS={PGO.INSTRUMENTED_PROFILE_FLAGS}",
            ],
            ["--no-assertions"],
        )

        self.assertIn("--cmake-opt=-DLLVM_BUILD_INSTRUMENTED=IR", command)
        self.assertIn(
            "--cmake-opt=-DLLVM_PROFILE_DATA_DIR=out/instrumented/profiles",
            command,
        )
        self.assertIn(
            "--cmake-opt=-DCMAKE_CXX_FLAGS=" f"{PGO.INSTRUMENTED_PROFILE_FLAGS}",
            command,
        )
        self.assertEqual(command[-1], "--no-assertions")

    def test_optimized_configuration_consumes_merged_profile(self):
        command = PGO._configure_command(
            Path("configure.py"),
            Path("src"),
            Path("out/optimized"),
            "Ninja",
            ["LLVM_PROFDATA_FILE=out/dpcpp.profdata"],
            [],
        )

        self.assertIn("--cmake-opt=-DLLVM_PROFDATA_FILE=out/dpcpp.profdata", command)

    def test_profdata_tool_comes_from_host_clang_toolchain(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            toolchain = Path(temporary_directory)
            clang = toolchain / "clang"
            suffix = ".exe" if os.name == "nt" else ""
            profdata = toolchain / f"llvm-profdata{suffix}"
            clang.touch()
            profdata.touch()

            found = PGO._find_profdata_tool(
                str(clang), requested_tool=None, dry_run=False
            )

            self.assertEqual(found, profdata.resolve())

    def test_missing_host_profdata_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            clang = Path(temporary_directory) / "clang"
            clang.touch()

            with self.assertRaisesRegex(RuntimeError, "host llvm-profdata"):
                PGO._find_profdata_tool(str(clang), requested_tool=None, dry_run=False)

    def test_explicit_profdata_override_is_preserved(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            profdata = Path(temporary_directory) / "custom-profdata"
            profdata.touch()

            found = PGO._find_profdata_tool(
                "clang", requested_tool=str(profdata), dry_run=False
            )

            self.assertEqual(found, profdata.absolute())

    def test_dry_run_build_plan_covers_sycl_training(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            source_dir = Path(temporary_directory)
            (source_dir / "buildbot").mkdir()
            (source_dir / "buildbot" / "configure.py").touch()
            arguments = argparse.Namespace(
                src_dir=str(source_dir),
                out_dir=str(source_dir / "out"),
                cmake_gen="Ninja",
                host_clang="clang",
                host_clangxx="clang++",
                configure_arg=[],
                training_target=None,
                build_target="all",
                parallelism=2,
                llvm_profdata=None,
                profile_output=None,
                profile_only=False,
                dry_run=True,
            )

            PGO.run_pgo_build(arguments)

            self.assertIn("check-sycl", PGO.DEFAULT_TRAINING_TARGETS)
            self.assertFalse((source_dir / "out").exists())

    def test_dry_run_configures_training_build_with_instrumented_compilers(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            source_dir = Path(temporary_directory)
            (source_dir / "buildbot").mkdir()
            (source_dir / "buildbot" / "configure.py").touch()
            arguments = argparse.Namespace(
                src_dir=str(source_dir),
                out_dir=str(source_dir / "out"),
                cmake_gen="Ninja",
                host_clang="clang",
                host_clangxx="clang++",
                configure_arg=[],
                training_target=None,
                build_target="all",
                parallelism=2,
                llvm_profdata=None,
                profile_output=None,
                profile_only=False,
                dry_run=True,
            )

            with mock.patch("builtins.print") as print_mock:
                PGO.run_pgo_build(arguments)

            plan = "\n".join(
                call.args[0] for call in print_mock.call_args_list if call.args
            )
            resolved_source = source_dir.resolve()
            training_dir = resolved_source / "out" / "training"
            instrumented_bin = resolved_source / "out" / "instrumented" / "bin"
            suffix = ".exe" if os.name == "nt" else ""
            self.assertIn(str(training_dir), plan)
            self.assertIn(
                f"CMAKE_C_COMPILER={instrumented_bin / f'clang{suffix}'}",
                plan,
            )
            self.assertIn(
                f"CMAKE_CXX_COMPILER={instrumented_bin / f'clang++{suffix}'}",
                plan,
            )

    def test_dry_run_uses_explicit_host_compilers(self):
        clang, clangxx = PGO._find_host_compilers(
            "toolchain/bin/clang", "toolchain/bin/clang++", dry_run=True
        )

        self.assertEqual(clang, "toolchain/bin/clang")
        self.assertEqual(clangxx, "toolchain/bin/clang++")

    def test_host_compiler_paths_preserve_clangxx_driver_name(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            toolchain = Path(temporary_directory)
            clang = toolchain / "clang"
            clangxx = toolchain / "clang++"
            clang.touch()
            try:
                clangxx.symlink_to(clang)
            except OSError as error:
                self.skipTest(f"cannot create a compiler symlink: {error}")

            found_clang, found_clangxx = PGO._find_host_compilers(
                str(clang), str(clangxx), dry_run=False
            )

            self.assertEqual(found_clang, os.path.abspath(clang))
            self.assertEqual(found_clangxx, os.path.abspath(clangxx))
            self.assertTrue(found_clangxx.endswith("clang++"))

    def test_host_compiler_lookup_does_not_resolve_paths(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            toolchain = Path(temporary_directory)
            clang = toolchain / "clang"
            clangxx = toolchain / "clang++"
            clang.touch()
            clangxx.touch()

            with mock.patch.object(
                Path,
                "resolve",
                side_effect=AssertionError("compiler path was resolved"),
            ):
                PGO._find_host_compilers(str(clang), str(clangxx), dry_run=False)

    def test_training_target_failures_are_nonfatal(self):
        runner = mock.Mock()
        runner.run.side_effect = subprocess.CalledProcessError(1, ["cmake"])

        with mock.patch("builtins.print") as print_mock:
            PGO._run_training_targets(runner, Path("instrumented"), ["check-clang"], 2)

        print_mock.assert_called_once()
        self.assertIn("continuing", print_mock.call_args.args[0])

    def test_existing_instrumented_build_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory)
            instrumented_dir = output_dir / "instrumented"
            instrumented_dir.mkdir()

            with self.assertRaisesRegex(RuntimeError, "stale profile data"):
                PGO._validate_fresh_outputs(
                    instrumented_dir,
                    output_dir / "training",
                    output_dir / "optimized",
                    output_dir / "dpcpp.profdata",
                    profile_only=False,
                )


if __name__ == "__main__":
    unittest.main()
