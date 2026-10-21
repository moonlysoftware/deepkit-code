//! Exact, bounded reuse of repeated chunk token counts; v1.md §6.1/§3.5.
use std::{cell::RefCell, collections::HashMap};

const MAX_BYTES: usize = 256 * 1024;
const MAX_ENTRIES: usize = 128;
const MAX_ENTRY_BYTES: usize = 64 * 1024;

#[derive(Default)]
struct Cache {
    counts: HashMap<String, usize>,
    bytes: usize,
}

impl Cache {
    fn count(&mut self, text: &str) -> usize {
        if let Some(&count) = self.counts.get(text) {
            return count;
        }
        let count = encode_count(text);
        if text.len() <= MAX_ENTRY_BYTES {
            if self.counts.len() >= MAX_ENTRIES || self.bytes + text.len() > MAX_BYTES {
                self.counts.clear();
                self.bytes = 0;
            }
            self.bytes += text.len();
            self.counts.insert(text.to_owned(), count);
        }
        count
    }
}

/// Shares exact counts between AST sizing and unit preparation on each worker.
/// Keys compare full text; memory is bounded independently of repository size.
pub(super) fn count(text: &str) -> usize {
    thread_local! {
        static CACHE: RefCell<Cache> = RefCell::new(Cache::default());
    }
    CACHE.with(|cache| cache.borrow_mut().count(text))
}

fn encode_count(text: &str) -> usize {
    static PREFIXED_SPECIALS: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
        tiktoken_rs::cl100k_base_singleton()
            .special_tokens()
            .iter()
            .all(|token| token.starts_with("<|"))
    });
    let tokenizer = tiktoken_rs::cl100k_base_singleton();
    // The vocabulary check keeps this exact even if a future tokenizer adds
    // differently spelled specials. Ordinary source avoids rebuilding the
    // special-token set and running its regex for every AST size probe.
    if *PREFIXED_SPECIALS && !text.contains("<|") {
        tokenizer.encode_ordinary(text).len()
    } else {
        tokenizer.encode_with_special_tokens(text).len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bounded_reuse_preserves_counts_across_eviction_and_large_inputs() {
        let tokenizer = tiktoken_rs::cl100k_base_singleton();
        let mut cache = Cache::default();
        let special = "λ 中文 <|endoftext|> tail";
        let expected = tokenizer.encode_with_special_tokens(special).len();
        for i in 0..MAX_ENTRIES * 2 {
            let text = format!("fn item_{i}() {{ /* {} */ }}", "value ".repeat(1500));
            assert_eq!(
                cache.count(&text),
                tokenizer.encode_with_special_tokens(&text).len()
            );
            assert_eq!(cache.count(special), expected);
            assert!(cache.bytes <= MAX_BYTES);
            assert!(cache.counts.len() <= MAX_ENTRIES);
            assert_eq!(
                cache.bytes,
                cache.counts.keys().map(String::len).sum::<usize>()
            );
        }
        let large = "large Unicode λ input
"
        .repeat(MAX_ENTRY_BYTES / 10);
        let retained = (cache.bytes, cache.counts.len());
        assert_eq!(
            cache.count(&large),
            tokenizer.encode_with_special_tokens(&large).len()
        );
        assert_eq!((cache.bytes, cache.counts.len()), retained);
        assert_eq!(cache.count(special), expected);
    }
}
