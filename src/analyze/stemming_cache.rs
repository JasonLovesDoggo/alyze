use ahash::AHashMap;

use crate::analyze::StemmingLanguage;

/// Keep a small cache of stemmed tokens to avoid repeated stemming of common tokens.
/// Can yield ~2x throughput improvement for stemmed analysis.
///
/// Keys are (lowercased) tokens of up to [`MAX_KEY_LEN`] bytes. Stems are stored as a diff
/// against the key — `key[..keep] ++ append` — since Snowball stemmers rewrite suffixes;
/// this keeps a (key, outcome) entry at 24 bytes. Tokens or stems that don't fit are simply
/// not cached.
///
/// The cache fills to capacity and then stops accepting entries (no eviction). Stems are
/// only valid for one language, so `prepare` clears the cache whenever a
/// differently-configured analyzer uses the same `ReusableBuffer`.
#[derive(Debug, Clone)]
pub struct StemmingCache {
    cache: AHashMap<CacheKey, StemOutcome>,
    /// Small direct-mapped cache in front of `cache`. The main map is ~1.5 MiB, so every
    /// probe of it is effectively an L2 access; the Zipf head of the token stream is hot
    /// enough that a 48 KiB front absorbs roughly half the lookups at L1 latency. Slots are
    /// clobbered on promotion, so hot tokens win them back immediately.
    front: Box<[(CacheKey, StemOutcome)]>,
    language: Option<StemmingLanguage>,
}

/// Tokens longer than this are not cached. Covers ~99.8% of word-like tokens in English
/// text (vs ~95.9% at 10).
const MAX_KEY_LEN: usize = 14;
const MAX_APPEND_LEN: usize = 6;

/// 2048 x 24-byte entries = 48 KiB.
const FRONT_SLOTS: usize = 2048;

/// Never equal to a real key (keys are non-empty), so vacant slots can't match.
const VACANT: (CacheKey, StemOutcome) = (
    ShortToken {
        valid_length: 0,
        buffer: [0; MAX_KEY_LEN],
    },
    StemOutcome::Unchanged,
);

/// Reconstructed stems are at most the kept key prefix plus the append suffix.
pub(crate) const MAX_STEM_LEN: usize = MAX_KEY_LEN + MAX_APPEND_LEN;

pub(crate) type CacheKey = ShortToken<MAX_KEY_LEN>;

impl StemmingCache {
    pub fn new_with_capacity(capacity: usize) -> Self {
        Self {
            cache: AHashMap::with_capacity(capacity),
            front: vec![VACANT; FRONT_SLOTS].into_boxed_slice(),
            language: None,
        }
    }

    /// Readies the cache for stemming in `language`, clearing it if the previous user
    /// stemmed a different language (cached stems are language-specific).
    pub(crate) fn prepare(&mut self, language: StemmingLanguage) {
        if self.language != Some(language) {
            self.cache.clear();
            self.front.fill(VACANT);
            self.language = Some(language);
        }
    }

    pub(crate) fn lookup(&mut self, key: &CacheKey) -> Option<&StemOutcome> {
        use std::hash::BuildHasher;
        let slot = self.cache.hasher().hash_one(key) as usize & (FRONT_SLOTS - 1);
        if self.front[slot].0 != *key {
            let outcome = *self.cache.get(key)?;
            self.front[slot] = (*key, outcome);
        }
        Some(&self.front[slot].1)
    }

    /// Inserts an outcome if there is spare capacity; the main map never evicts or grows.
    pub(crate) fn insert(&mut self, key: CacheKey, outcome: StemOutcome) {
        use std::hash::BuildHasher;
        if self.cache.len() < self.cache.capacity() {
            let clobbered = self.cache.insert(key, outcome);
            debug_assert!(clobbered.is_none(), "lookup misses should precede inserts");
            // Fresh stems are usually about to repeat (topical words), so front them too.
            let slot = self.cache.hasher().hash_one(&key) as usize & (FRONT_SLOTS - 1);
            self.front[slot] = (key, outcome);
        }
    }
}

/// The result of stemming one token, encoded relative to its cache key.
#[derive(Debug, Clone, Copy)]
pub(crate) enum StemOutcome {
    Unchanged,
    /// The stem is `key[..keep] ++ append[..append_len]`.
    Stemmed {
        keep: u8,
        append_len: u8,
        append: [u8; MAX_APPEND_LEN],
    },
}

impl StemOutcome {
    /// Encodes `stem` as a diff against `key`, or `None` if it doesn't fit (the token then
    /// just isn't cached).
    pub(crate) fn encode(key: &CacheKey, stem: &str) -> Option<Self> {
        let key_bytes = key.as_str().as_bytes();
        let stem_bytes = stem.as_bytes();
        let keep = key_bytes
            .iter()
            .zip(stem_bytes)
            .take_while(|(k, s)| k == s)
            .count();
        let tail = &stem_bytes[keep..];
        if tail.len() > MAX_APPEND_LEN {
            return None;
        }
        let mut append = [0; MAX_APPEND_LEN];
        append[..tail.len()].copy_from_slice(tail);
        Some(Self::Stemmed {
            keep: keep as u8,
            append_len: tail.len() as u8,
            append,
        })
    }

    /// Reconstructs the stem into `buf`; `None` means the token stems to itself.
    pub(crate) fn reconstruct<'b>(
        &self,
        key: &CacheKey,
        buf: &'b mut [u8; MAX_STEM_LEN],
    ) -> Option<&'b str> {
        match self {
            Self::Unchanged => None,
            Self::Stemmed {
                keep,
                append_len,
                append,
            } => {
                let (keep, append_len) = (*keep as usize, *append_len as usize);
                buf[..keep].copy_from_slice(&key.as_str().as_bytes()[..keep]);
                buf[keep..keep + append_len].copy_from_slice(&append[..append_len]);
                // SAFETY: `encode` guarantees these are exactly the bytes of the original
                // stem, which was a valid UTF-8 string.
                Some(unsafe { std::str::from_utf8_unchecked(&buf[..keep + append_len]) })
            }
        }
    }
}

// Non-empty, short token with a maximum size (N).
// Avoids heap allocs and minimizes memory footprint.
#[derive(Debug, Hash, Eq, PartialEq, Clone, Copy)]
pub(crate) struct ShortToken<const N: usize> {
    valid_length: u8,
    buffer: [u8; N],
}

impl<const N: usize> ShortToken<N> {
    const _ASSERT: () = assert!(N <= u8::MAX as usize, "N must be <= 255");

    pub(crate) fn new_from_str(s: &str) -> Option<Self> {
        let bytes = s.as_bytes();
        if bytes.is_empty() || bytes.len() > N {
            return None;
        }
        let mut buffer = [0; N];
        buffer[..bytes.len()].copy_from_slice(bytes);
        Some(Self {
            valid_length: bytes.len() as u8,
            buffer,
        })
    }

    pub(crate) fn as_str(&self) -> &str {
        // SAFETY: Buffer is always initialized with valid UTF-8 from the constructor,
        // and valid_length is guaranteed to be <= the length of the buffer.
        unsafe { std::str::from_utf8_unchecked(&self.buffer[..self.valid_length as usize]) }
    }
}

#[cfg(test)]
mod tests {
    use super::{CacheKey, MAX_STEM_LEN, StemOutcome};

    #[track_caller]
    fn roundtrip(key: &str, stem: &str) -> Option<String> {
        let key = CacheKey::new_from_str(key).unwrap();
        let outcome = StemOutcome::encode(&key, stem)?;
        let mut buf = [0; MAX_STEM_LEN];
        Some(outcome.reconstruct(&key, &mut buf).unwrap().to_string())
    }

    #[test]
    fn outcome_roundtrips() {
        // Snowball stems rewrite suffixes, so they share a prefix with the input.
        for (key, stem) in [
            ("running", "run"),
            ("happy", "happi"),
            ("ties", "tie"),
            ("studies", "studi"),
            ("générations", "génér"), // non-ASCII bytes in the kept prefix
            ("a", "a"),
        ] {
            assert_eq!(
                roundtrip(key, stem).as_deref(),
                Some(stem),
                "{key} -> {stem}"
            );
        }
    }

    #[test]
    fn oversized_diffs_are_not_cached() {
        assert_eq!(roundtrip("shorten", "completelydifferent"), None);
    }

    #[test]
    fn front_cache_serves_same_outcomes() {
        use super::StemmingCache;
        use crate::analyze::StemmingLanguage;

        let mut cache = StemmingCache::new_with_capacity(100);
        cache.prepare(StemmingLanguage::English);
        let key = CacheKey::new_from_str("running").unwrap();
        let outcome = StemOutcome::encode(&key, "run").unwrap();
        cache.insert(key, outcome);

        let mut buf = [0; MAX_STEM_LEN];
        // First lookup may promote from the main map; the second is served by the front.
        for _ in 0..2 {
            let got = cache.lookup(&key).unwrap().reconstruct(&key, &mut buf);
            assert_eq!(got, Some("run"));
        }

        // A language switch must clear the front as well as the main map.
        cache.prepare(StemmingLanguage::German);
        assert!(cache.lookup(&key).is_none());
    }

    #[test]
    fn entries_stay_compact() {
        // The point of the diff encoding: a (key, outcome) pair fits in 24 bytes.
        assert_eq!(
            std::mem::size_of::<(CacheKey, StemOutcome)>(),
            24,
            "cache entry grew; check the memory math in the module docs"
        );
    }
}
