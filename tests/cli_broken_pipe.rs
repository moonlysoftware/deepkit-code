//! CLI stdout behavior when a consumer closes its pipe; see `v1.md` §16.

use std::{
    error::Error,
    process::{Command, Stdio},
};

#[test]
fn cli_output_survives_a_closed_stdout_pipe() -> Result<(), Box<dyn Error>> {
    let root = tempfile::tempdir()?;
    let binary = assert_cmd::cargo::cargo_bin!("dkc");

    for args in [["config", "dump"], ["status", "--json"]] {
        let normal = Command::new(binary)
            .args(args)
            .current_dir(root.path())
            .output()?;
        assert!(normal.status.success(), "normal {args:?}: {normal:?}");
        assert!(!normal.stdout.is_empty(), "empty normal {args:?} output");

        for _ in 0..4 {
            let mut child = Command::new(binary)
                .args(args)
                .current_dir(root.path())
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .spawn()?;
            drop(child.stdout.take());
            let output = child.wait_with_output()?;
            assert!(
                output.status.success(),
                "closed-pipe {args:?} failed: {}",
                String::from_utf8_lossy(&output.stderr)
            );
            assert!(
                !String::from_utf8_lossy(&output.stderr).contains("panicked"),
                "closed-pipe {args:?} panicked"
            );
        }
    }
    Ok(())
}
