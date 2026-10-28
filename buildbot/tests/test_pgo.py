import argparse
import importlib.util
from pathlib import Path
import tempfile
import unittest


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
            ],
            ["--no-assertions"],
        )

        self.assertIn("--cmake-opt=-DLLVM_BUILD_INSTRUMENTED=IR", command)
        self.assertIn(
            "--cmake-opt=-DLLVM_PROFILE_DATA_DIR=out/instrumented/profiles",
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

        self.assertIn(
            "--cmake-opt=-DLLVM_PROFDATA_FILE=out/dpcpp.profdata", command
        )

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

    def test_dry_run_uses_explicit_host_compilers(self):
        clang, clangxx = PGO._find_host_compilers(
            "toolchain/bin/clang", "toolchain/bin/clang++", dry_run=True
        )

        self.assertEqual(clang, "toolchain/bin/clang")
        self.assertEqual(clangxx, "toolchain/bin/clang++")

    def test_existing_instrumented_build_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory)
            instrumented_dir = output_dir / "instrumented"
            instrumented_dir.mkdir()

            with self.assertRaisesRegex(RuntimeError, "stale profile data"):
                PGO._validate_fresh_outputs(
                    instrumented_dir,
                    output_dir / "optimized",
                    output_dir / "dpcpp.profdata",
                    profile_only=False,
                )


if __name__ == "__main__":
    unittest.main()
