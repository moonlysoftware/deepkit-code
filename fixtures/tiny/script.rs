pub fn should_retry(attempt: u32) -> bool {
    attempt < 3
}
