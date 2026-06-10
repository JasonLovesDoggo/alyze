use crate::{
    analyze::stemming_cache::{CacheKey, StemOutcome, StemmingCache},
    uax29,
};

mod filters;
pub mod stemming_cache;
mod stopwords;
mod u17_to_lower;

#[derive(Clone, Copy, Debug)]
pub struct AnalysisOptions {
    pub tokenizer: TokenizerOptions,

    // Note: These are ordered in the sequence they are applied in
    pub maximum_token_length: Option<usize>,
    pub case_sensitive: bool,
    pub stopword_removal: Option<StopwordRemoval>,
    pub stemming: Option<StemmingLanguage>,
    pub ascii_folding: bool,
}

impl AnalysisOptions {
    pub fn valid(&self) -> bool {
        if self.stemming.is_some() && self.case_sensitive {
            return false; // stemming requires case insensitivity
        }
        if self.stopword_removal.is_some() && self.case_sensitive {
            return false; // stopword removal requires case insensitivity
        }
        true
    }
}

#[derive(Clone, Copy, Debug)]
pub enum TokenizerOptions {
    UAX29Word(uax29::word::Options),
}

#[derive(Copy, Clone, Debug)]
pub enum StopwordRemoval {
    ForLanguage(LanguageWithStopwords),
}

#[derive(Copy, Clone, Debug)]
pub enum LanguageWithStopwords {
    Danish,
    Dutch,
    English,
    Finnish,
    French,
    German,
    Hungarian,
    Italian,
    Norwegian,
    Portuguese,
    Russian,
    Spanish,
    Swedish,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum StemmingLanguage {
    Arabic,
    Danish,
    Dutch,
    English,
    Finnish,
    French,
    German,
    Greek,
    Hungarian,
    Italian,
    Norwegian,
    Portuguese,
    Romanian,
    Russian,
    Spanish,
    Swedish,
    Tamil,
    Turkish,
}

impl Into<rust_stemmers::Algorithm> for StemmingLanguage {
    fn into(self) -> rust_stemmers::Algorithm {
        match self {
            StemmingLanguage::Arabic => rust_stemmers::Algorithm::Arabic,
            StemmingLanguage::Danish => rust_stemmers::Algorithm::Danish,
            StemmingLanguage::Dutch => rust_stemmers::Algorithm::Dutch,
            StemmingLanguage::English => rust_stemmers::Algorithm::English,
            StemmingLanguage::Finnish => rust_stemmers::Algorithm::Finnish,
            StemmingLanguage::French => rust_stemmers::Algorithm::French,
            StemmingLanguage::German => rust_stemmers::Algorithm::German,
            StemmingLanguage::Greek => rust_stemmers::Algorithm::Greek,
            StemmingLanguage::Hungarian => rust_stemmers::Algorithm::Hungarian,
            StemmingLanguage::Italian => rust_stemmers::Algorithm::Italian,
            StemmingLanguage::Norwegian => rust_stemmers::Algorithm::Norwegian,
            StemmingLanguage::Portuguese => rust_stemmers::Algorithm::Portuguese,
            StemmingLanguage::Romanian => rust_stemmers::Algorithm::Romanian,
            StemmingLanguage::Russian => rust_stemmers::Algorithm::Russian,
            StemmingLanguage::Spanish => rust_stemmers::Algorithm::Spanish,
            StemmingLanguage::Swedish => rust_stemmers::Algorithm::Swedish,
            StemmingLanguage::Tamil => rust_stemmers::Algorithm::Tamil,
            StemmingLanguage::Turkish => rust_stemmers::Algorithm::Turkish,
        }
    }
}

/// A buffer that should be reused across multiple analyze() invocations
/// to avoid unnecessary allocations. Contents are opaque and internal to
/// the implementation.
#[derive(Debug, Clone)]
pub struct ReusableBuffer {
    a: String,
    b: String,
    stemming_cache: StemmingCache,
}

impl ReusableBuffer {
    pub fn new() -> Self {
        Self {
            a: String::new(),
            b: String::new(),
            stemming_cache: StemmingCache::new_with_capacity(32_000),
        }
    }

    pub fn stemming_cache(&mut self) -> &mut StemmingCache {
        &mut self.stemming_cache
    }

    pub fn reset_keep_stemming_cache(&mut self) {
        self.a.clear();
        self.b.clear();
    }
}

#[derive(Clone, Copy)]
pub struct Analyzer {
    options: AnalysisOptions,
}

impl Analyzer {
    pub fn new(options: AnalysisOptions) -> Self {
        assert!(options.valid(), "options are invalid");
        Self { options }
    }

    /// Analyzes a single input string, invoking the callback for each token.
    /// Returning false from the callback will stop analysis early.
    pub fn analyze<'a>(
        &self,
        input: &'a str,
        buffer: &mut ReusableBuffer,
        callback: impl FnMut(Token<'_>) -> bool,
    ) {
        self.analyze_inputs(std::iter::once(input), buffer, callback);
    }

    /// Analyzes a sequence of input strings, invoking the callback for each token.
    /// Returning false from the callback will stop analysis early.
    pub fn analyze_inputs<'a>(
        &self,
        inputs: impl Iterator<Item = &'a str>,
        buffer: &mut ReusableBuffer,
        callback: impl FnMut(Token<'_>) -> bool,
    ) {
        // Monomorphize on case sensitivity: the case-sensitive instantiation provably never
        // reads `TokenProperties::has_ascii_uppercase`, which lets the compiler strip that
        // bit's bookkeeping from its tokenizer fast path entirely.
        if self.options.case_sensitive {
            self.analyze_inputs_impl::<true>(inputs, buffer, callback)
        } else {
            self.analyze_inputs_impl::<false>(inputs, buffer, callback)
        }
    }

    fn analyze_inputs_impl<'a, const CASE_SENSITIVE: bool>(
        &self,
        inputs: impl Iterator<Item = &'a str>,
        buffer: &mut ReusableBuffer,
        mut callback: impl FnMut(Token<'_>) -> bool,
    ) {
        debug_assert_eq!(CASE_SENSITIVE, self.options.case_sensitive);
        let ReusableBuffer {
            a: buffer_a,
            b: buffer_b,
            stemming_cache,
        } = buffer;

        let stemmer = self.options.stemming.map(|stemming_language| {
            let algorithm = stemming_language.into();
            rust_stemmers::Stemmer::create(algorithm)
        });
        if let Some(language) = self.options.stemming {
            // Cached stems are language-specific; reusing this buffer with a different
            // stemming language must not serve stale entries.
            stemming_cache.prepare(language);
        }

        // Monotonic across all inputs. Every word-like token consumes
        // a position, even if a downstream filter (length, stopword) drops it,
        // which is important for phrase-distance accuracy.
        //
        // TODO configurable gap between inputs
        let mut next_position = 0;

        let TokenizerOptions::UAX29Word(tokenizer_opts) = self.options.tokenizer;

        for input in inputs {
            let mut prev = None;
            let input_as_bytes = input.as_bytes();
            uax29::word::tokenize(input, tokenizer_opts, |bp, props| {
                let Some(prev) = std::mem::replace(&mut prev, Some(bp)) else {
                    return true; // don't emit token on first breakpoint
                };
                if !props.is_word_like() {
                    return true; // skip non-word tokens
                }

                // Advance position after each word-like token.
                let position = next_position;
                next_position += 1;

                // SAFETY: tokenize guarentees that breakpoints are on valid UTF-8 boundaries,
                // thus slicing input by the breakpoint will always produce valid UTF-8.
                buffer_a.clear();
                let mut token_text = InputRefOrBuffered::InputRef {
                    input: unsafe { std::str::from_utf8_unchecked(&input_as_bytes[prev..bp]) },
                    buffer_if_needed: buffer_a,
                };

                // Token length
                if let Some(max_token_length) = self.options.maximum_token_length
                    && !filters::within_token_length_limit(token_text.as_str(), max_token_length)
                {
                    return true;
                }

                // Lowercasing
                if !CASE_SENSITIVE {
                    token_text.lowercase_in_place(props.is_ascii(), props.has_ascii_uppercase());
                }

                // Stopword removal
                if let Some(StopwordRemoval::ForLanguage(language)) = self.options.stopword_removal
                    && filters::is_stopword_in_language(language, token_text.as_str())
                {
                    return true;
                }

                // Stemming
                if let Some(stemmer) = &stemmer {
                    token_text.stem_in_place(stemmer, stemming_cache, buffer_b);
                }

                // ASCII folding
                // Note: Not needed if token is already ASCII
                if self.options.ascii_folding && !props.is_ascii() {
                    token_text.ascii_fold_in_place(buffer_b);

                    // ASCII folding can produce uppercase ASCII characters,
                    // so we'll lowercase again if case folding is enabled.
                    if !CASE_SENSITIVE {
                        let is_ascii = token_text.as_str().is_ascii();
                        token_text.lowercase_in_place(is_ascii, true);
                    }
                }

                let token = Token {
                    text: token_text.as_str(),
                    position,
                };
                callback(token)
            });
        }
    }
}

pub struct Token<'a> {
    /// Text of the token, either sliced from the input string or from the reused
    /// buffer. Only valid for the duration of the callback invocation.
    pub text: &'a str,

    /// Position of the token in the sequence of tokens. If `analyze_inputs` is used,
    /// token positions are threaded monotonically across all input strings. Every word-like
    /// token consumes one position, even if filtered out (e.g. by stopword removal, etc).
    pub position: usize,
}

enum InputRefOrBuffered<'input, 'buf> {
    InputRef {
        input: &'input str,
        buffer_if_needed: &'buf mut String,
    },
    Buffered(&'buf mut String),
}

impl InputRefOrBuffered<'_, '_> {
    fn as_str(&self) -> &str {
        match self {
            Self::InputRef { input, .. } => input,
            Self::Buffered(s) => s.as_str(),
        }
    }

    /// `may_have_upper` is a hint from the tokenizer: already-lowercase ASCII tokens (the
    /// overwhelmingly common case in prose) return immediately, without rescanning the
    /// bytes. Pass `true` when unknown.
    #[inline(always)]
    fn lowercase_in_place(&mut self, is_ascii: bool, may_have_upper: bool) {
        if is_ascii && !may_have_upper {
            debug_assert!(!self.as_str().bytes().any(|b| b.is_ascii_uppercase()));
            return;
        }
        self.lowercase_in_place_slow(is_ascii);
    }

    fn lowercase_in_place_slow(&mut self, is_ascii: bool) {
        debug_assert_eq!(
            is_ascii,
            self.as_str().is_ascii(),
            "caller must ensure is_ascii is correct"
        );

        if is_ascii && self.as_str().bytes().all(|b| !b.is_ascii_uppercase()) {
            return;
        }

        if let Self::InputRef {
            input,
            buffer_if_needed,
        } = self
        {
            debug_assert!(
                buffer_if_needed.is_empty(),
                "buffer must be empty when passed in for potential reuse"
            );
            buffer_if_needed.push_str(input);
            self.transition_to_buffered();
        }

        let Self::Buffered(s) = self else {
            unreachable!()
        };
        if is_ascii {
            s.make_ascii_lowercase();
        } else {
            filters::lowercase_chars_in_place(s);
        }
    }

    fn ascii_fold_in_place(&mut self, scratch: &mut String) {
        match self {
            Self::InputRef {
                input,
                buffer_if_needed,
            } => {
                debug_assert!(
                    buffer_if_needed.is_empty(),
                    "buffer must be empty when passed in for potential reuse"
                );
                filters::ascii_fold(input, buffer_if_needed);
                self.transition_to_buffered();
            }
            Self::Buffered(s) => {
                debug_assert!(
                    scratch.is_empty(),
                    "scratch buffer must be empty when passed in for potential reuse"
                );
                filters::ascii_fold(s, scratch);
                std::mem::swap(*s, scratch);
                scratch.clear();
            }
        }
    }

    fn stem_in_place(
        &mut self,
        stemmer: &rust_stemmers::Stemmer,
        cache: &mut StemmingCache,
        scratch: &mut String,
    ) {
        let cache_key = CacheKey::new_from_str(self.as_str());
        if let Some(key) = cache_key.as_ref()
            && let Some(outcome) = cache.lookup(key)
        {
            let mut stem_buf = [0; stemming_cache::MAX_STEM_LEN];
            if let Some(stem) = outcome.reconstruct(key, &mut stem_buf) {
                match self {
                    Self::InputRef {
                        buffer_if_needed, ..
                    } => {
                        debug_assert!(
                            buffer_if_needed.is_empty(),
                            "buffer must be empty when passed in for potential reuse"
                        );
                        buffer_if_needed.push_str(stem);
                        self.transition_to_buffered();
                    }
                    Self::Buffered(buf) => {
                        buf.clear();
                        buf.push_str(stem);
                    }
                }
            }
            return;
        }

        let outcome_to_insert = match self {
            Self::InputRef {
                input,
                buffer_if_needed,
            } => {
                let stemmed = stemmer.stem(input);
                if stemmed == *input {
                    Some(StemOutcome::Unchanged)
                } else {
                    debug_assert!(
                        buffer_if_needed.is_empty(),
                        "buffer must be empty when passed in for potential reuse"
                    );
                    let outcome = cache_key
                        .as_ref()
                        .and_then(|key| StemOutcome::encode(key, &stemmed));
                    buffer_if_needed.push_str(&stemmed);
                    self.transition_to_buffered();
                    outcome
                }
            }
            Self::Buffered(s) => {
                let stemmed = stemmer.stem(s.as_str());
                if stemmed == s.as_str() {
                    Some(StemOutcome::Unchanged)
                } else {
                    debug_assert!(
                        scratch.is_empty(),
                        "scratch buffer must be empty when passed in for potential reuse"
                    );
                    let outcome = cache_key
                        .as_ref()
                        .and_then(|key| StemOutcome::encode(key, &stemmed));
                    scratch.push_str(&stemmed);
                    std::mem::swap(*s, scratch);
                    scratch.clear(); // cleanup for caller's next use
                    outcome
                }
            }
        };

        if let Some(key) = cache_key
            && let Some(outcome) = outcome_to_insert
        {
            cache.insert(key, outcome);
        }
    }

    // Mutates self to transition from `InputRef` to `Buffered`. Caller is responsible
    // for populating `buffer_if_needed` with the appropriate contents before calling this.
    fn transition_to_buffered(&mut self) {
        // SAFETY: `InputRef` holds only `&mut` references (no owned data), so
        // dropping its bytes via overwrite is a no-op. We `ptr::read` self,
        // consume it to construct the new variant, then `ptr::write` back —
        // `*self` is never observed in an uninitialized state, and no value
        // is dropped twice.
        unsafe {
            let new = match std::ptr::read(self) {
                Self::InputRef {
                    buffer_if_needed, ..
                } => Self::Buffered(buffer_if_needed),
                Self::Buffered(_) => unreachable!(),
            };
            std::ptr::write(self, new);
        }
    }
}

// TODO this has extensive coverage in the turbopuffer repo, but not in the crate itself
// move some of the test suite in here

#[cfg(test)]
mod tests {
    use super::*;

    fn collect(
        analyzer: &Analyzer,
        buffer: &mut ReusableBuffer,
        input: &str,
    ) -> Vec<(String, usize)> {
        let mut out = Vec::new();
        analyzer.analyze(input, buffer, |token| {
            out.push((token.text.to_string(), token.position));
            true
        });
        out
    }

    fn full_english() -> AnalysisOptions {
        AnalysisOptions {
            tokenizer: TokenizerOptions::UAX29Word(crate::uax29::word::Options::default()),
            maximum_token_length: Some(40),
            case_sensitive: false,
            stopword_removal: Some(StopwordRemoval::ForLanguage(LanguageWithStopwords::English)),
            stemming: Some(StemmingLanguage::English),
            ascii_folding: true,
        }
    }

    #[test]
    fn full_pipeline_output() {
        let analyzer = Analyzer::new(full_english());
        let mut buffer = ReusableBuffer::new();
        // "The" and "are" are stopwords but still consume positions.
        assert_eq!(
            collect(&analyzer, &mut buffer, "The quick Foxes are running"),
            vec![
                ("quick".to_string(), 1),
                ("fox".to_string(), 2),
                ("run".to_string(), 4)
            ]
        );
    }

    /// The token cache must be transparent: a cold pass (all misses, running the real
    /// filter chain) and a warm pass (served from cache) must produce identical output.
    /// Inputs cover ASCII/Unicode case folding, folding-induced growth, skip outcomes,
    /// tokens too long to cache, and outputs too divergent to encode.
    #[test]
    fn cache_is_transparent() {
        let analyzer = Analyzer::new(full_english());
        let mut buffer = ReusableBuffer::new();
        let input = "The quick brown Foxes are running and THEIR happiness is \
                     internationalization! Wikipedia café Дом ПРИВЕТМИР İstanbul ﬃ \
                     e.g. can't 1,000 _connector_ a\u{0301} León supercalifragilistic";
        let cold = collect(&analyzer, &mut buffer, input);
        let warm = collect(&analyzer, &mut buffer, input);
        assert!(!cold.is_empty());
        assert_eq!(cold, warm);
    }

    /// Skip outcomes served from the cache must still consume token positions
    /// (phrase-distance accuracy depends on this).
    #[test]
    fn cached_skips_consume_positions() {
        let analyzer = Analyzer::new(AnalysisOptions {
            maximum_token_length: Some(5),
            ..full_english()
        });
        let mut buffer = ReusableBuffer::new();
        let input = "the extraordinary cat the extraordinary cat";
        let expected = vec![("cat".to_string(), 2), ("cat".to_string(), 5)];
        assert_eq!(collect(&analyzer, &mut buffer, input), expected);
        // Warm pass: every skip now comes from the cache.
        assert_eq!(collect(&analyzer, &mut buffer, input), expected);
    }

    /// Sharing a `ReusableBuffer` across differently-configured analyzers must not leak
    /// cached outcomes between configurations.
    #[test]
    fn cache_cleared_on_options_change() {
        let english = Analyzer::new(full_english());
        let german = Analyzer::new(AnalysisOptions {
            stopword_removal: Some(StopwordRemoval::ForLanguage(LanguageWithStopwords::German)),
            stemming: Some(StemmingLanguage::German),
            ..full_english()
        });
        let input = "connection connection";

        let mut shared = ReusableBuffer::new();
        let english_out = collect(&english, &mut shared, input);
        let german_via_shared = collect(&german, &mut shared, input);
        let german_fresh = collect(&german, &mut ReusableBuffer::new(), input);
        assert_eq!(german_via_shared, german_fresh);
        // Sanity: the two configurations actually disagree on this input.
        assert_ne!(english_out, german_via_shared);
    }
}
