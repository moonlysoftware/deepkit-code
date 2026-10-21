#!/usr/bin/env python3
"""Exercise the musl release wrapper's compiler and linker contracts."""

import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


WRAPPER = Path(__file__).with_name("zig-musl-tool.sh")


class ZigMuslToolTests(unittest.TestCase):
    def setUp(self):
        self.scratch = tempfile.TemporaryDirectory(prefix="dkc-zig-tool-test-")
        self.addCleanup(self.scratch.cleanup)
        self.root = Path(self.scratch.name)
        fake_zig = self.root / "zig"
        fake_zig.write_text(
            "#!/usr/bin/env python3\n"
            "import json, os, sys\n"
            "with open(os.environ['DKC_ZIG_ARGS_PATH'], 'w') as out:\n"
            "    json.dump(sys.argv[1:], out)\n"
        )
        fake_zig.chmod(0o755)
        self.env = os.environ.copy()
        self.env["PATH"] = f"{self.root}:{self.env['PATH']}"
        self.env["DKC_ZIG_ARGS_PATH"] = str(self.root / "arguments.json")

    def invoke(self, name, *arguments):
        tool = self.root / name
        tool.symlink_to(WRAPPER)
        result = subprocess.run(
            [str(tool), *arguments],
            env=self.env,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            args = json.loads((self.root / "arguments.json").read_text())
        else:
            args = None
        return result, args

    def test_x86_compiler_translates_rust_target_and_avx512(self):
        result, args = self.invoke(
            "dkc-zig-musl-x86_64-cc",
            "--target=x86_64-unknown-linux-musl",
            "-mavx512f",
            "-c",
            "vector.c",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            args,
            ["cc", "-target", "x86_64-linux-musl", "--target=x86_64-linux-musl",
             "-mavx512f", "-c", "vector.c", "-mavx10.1-512"],
        )

    def test_arm_linker_uses_zig_crt_and_drops_unsupported_flag(self):
        result, args = self.invoke(
            "dkc-zig-musl-aarch64-linker",
            "--target=aarch64-unknown-linux-musl",
            "-Wl,--fix-cortex-a53-843419",
            "/sysroot/lib/self-contained/crt1.o",
            "main.o",
            "-lc",
            "-lgcc_s",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            args,
            ["c++", "-target", "aarch64-linux-musl", "main.o", "-lunwind"],
        )

    def test_unknown_command_name_is_rejected(self):
        result, args = self.invoke("dkc-zig-musl-invalid", "main.o")
        self.assertEqual(result.returncode, 2)
        self.assertIsNone(args)


if __name__ == "__main__":
    unittest.main()
