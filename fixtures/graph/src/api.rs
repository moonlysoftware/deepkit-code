/// Returns the incremented input for callers in this module.
pub fn duplicate(value: u32) -> u32 {
    value + 1
}

pub fn caller() -> u32 {
    duplicate(1)
}
