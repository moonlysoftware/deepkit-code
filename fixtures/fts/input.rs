// Configures retries.
pub struct RetryPolicy {
    maxRetries: u32,
}

pub fn getUserById(user_id: u64) -> User {
    let secret = "never body";
    // Returns a user.
    get_user_by_id(user_id)
}
