"""Generated document fixtures need a Git HEAD without tracking hostile files."""

from pathlib import Path
import runpy
import subprocess
import tempfile
import unittest


SCRIPT = Path(__file__).with_name("make-fixtures-docs.py")


class FixtureGitSeedTest(unittest.TestCase):
    def test_seed_is_idempotent_and_leaves_fixture_bytes_untracked(self) -> None:
        namespace = runpy.run_path(str(SCRIPT))
        seed = namespace["seed_git"]
        verify = namespace["verify_git"]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "README.md").write_text("generated fixture\n")
            first = seed(root)
            self.assertEqual(first, seed(root))
            self.assertEqual(first, verify(root))
            self.assertEqual(subprocess.check_output(
                ["git", "-C", str(root), "ls-tree", "--name-only", "HEAD"],
                text=True,
            ).strip(), "")
            self.assertEqual(subprocess.check_output(
                ["git", "-C", str(root), "ls-files"], text=True,
            ).strip(), "")
            (root / "README.md").write_text("changed fixture\n")
            self.assertEqual(first, verify(root))
            subprocess.run(["git", "-C", str(root), "add", "README.md"], check=True)
            with self.assertRaisesRegex(ValueError, "not the empty seed commit"):
                verify(root)
            subprocess.run(["git", "-C", str(root), "reset", "-q"], check=True)
            subprocess.run([
                "git", "-C", str(root), "-c", "user.name=Test",
                "-c", "user.email=test@example.invalid", "commit", "--allow-empty",
                "-qm", "unexpected new head",
            ], check=True)
            with self.assertRaisesRegex(ValueError, "not the empty seed commit"):
                verify(root)


if __name__ == "__main__":
    unittest.main()
