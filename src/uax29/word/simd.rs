//! NEON block classifier for the word-break fast path.
//!
//! Classifies 64 input bytes at a time into the four character classes the fast path cares
//! about, returning one bitmask per class (bit i = byte i):
//!
//! - `word`: `[a-zA-Z0-9_]` — continues a word run (`alnum` is the same minus `_`).
//! - `space`: ASCII space — continues a whitespace run (WB3d).
//! - `solo`: ASCII bytes with `WordBreakProperty::Other` (`!`, `(`, `-`, tab, NUL, ...).
//!   These break unconditionally on both sides and form single-byte tokens.
//! - `x`: everything the fast path must hand to the DFA: non-ASCII bytes, CR/LF/VT/FF, and
//!   the mid-token candidates `, . : ; ' "` (MidNum / MidNumLet / MidLetter / quotes).
//!
//! Within a run of word/space/solo bytes, UAX #29 places boundaries exactly at class
//! transitions (plus one before every solo byte); `tests::classes_match_word_break_table`
//! pins this classification to `ASCII_WORD_BREAK_PROP`, and the differential test in the
//! parent module pins the resulting tokenization to the scalar implementation.
//!
//! The classifier is two `vqtbl1q` nibble lookups per 16 bytes (the simdjson technique):
//! each byte's class is `LO_LUT[b & 0xF] & HI_LUT[b >> 4]`, where each bit of the lookup
//! value is one (high-nibble set, low-nibble set) product term.

use std::arch::aarch64::{
    uint8x16_t, vandq_u8, vdupq_n_u8, vgetq_lane_u64, vld1q_u8, vorrq_u8, vpaddq_u8, vqtbl1q_u8,
    vreinterpretq_u64_u8, vshrq_n_u8, vtstq_u8,
};

// Product terms. Each is a (high nibbles, low nibbles) rectangle of the byte table.
const T_DIGIT: u8 = 0x01; //    hi 3,   lo 0-9: 0-9
const T_LETTER1: u8 = 0x02; //  hi 4|6, lo 1-F: A-O, a-o
const T_LETTER2: u8 = 0x04; //  hi 5|7, lo 0-A: P-Z, p-z
const T_UNDERSCORE: u8 = 0x08; // hi 5, lo F
const T_SPACE: u8 = 0x10; //    hi 2,   lo 0
const T_MIDQUOTE: u8 = 0x20; // hi 2,   lo 2,7,C,E: " ' , .
const T_MIDCOLON: u8 = 0x40; // hi 3,   lo A,B: : ;
const T_NEWLINE: u8 = 0x80; // hi 0,   lo A-D: LF VT FF CR

const WORD: u8 = T_DIGIT | T_LETTER1 | T_LETTER2 | T_UNDERSCORE;
const ALNUM: u8 = T_DIGIT | T_LETTER1 | T_LETTER2;
const NEEDS_DFA: u8 = T_MIDQUOTE | T_MIDCOLON | T_NEWLINE;

const LO_LUT: [u8; 16] = {
    let mut lut = [0u8; 16];
    let mut lo = 0;
    while lo < 16 {
        lut[lo] = (T_DIGIT * (lo <= 9) as u8)
            | (T_LETTER1 * (lo >= 1) as u8)
            | (T_LETTER2 * (lo <= 0xA) as u8)
            | (T_UNDERSCORE * (lo == 0xF) as u8)
            | (T_SPACE * (lo == 0) as u8)
            | (T_MIDQUOTE * matches!(lo, 0x2 | 0x7 | 0xC | 0xE) as u8)
            | (T_MIDCOLON * matches!(lo, 0xA | 0xB) as u8)
            | (T_NEWLINE * (0xA <= lo && lo <= 0xD) as u8);
        lo += 1;
    }
    lut
};

const HI_LUT: [u8; 16] = {
    let mut lut = [0u8; 16];
    lut[0x0] = T_NEWLINE;
    lut[0x2] = T_SPACE | T_MIDQUOTE;
    lut[0x3] = T_DIGIT | T_MIDCOLON;
    lut[0x4] = T_LETTER1;
    lut[0x5] = T_LETTER2 | T_UNDERSCORE;
    lut[0x6] = T_LETTER1;
    lut[0x7] = T_LETTER2;
    // 0x8-0xF (non-ASCII) stay 0; those bytes land in `x` via their high bit.
    lut
};

/// Per-class bitmasks for one 64-byte block; bit i corresponds to byte `base + i`.
#[derive(Clone, Copy)]
pub(super) struct BlockMasks {
    pub(super) word: u64,
    pub(super) space: u64,
    pub(super) solo: u64,
    pub(super) alnum: u64,
    pub(super) x: u64,
}

impl BlockMasks {
    pub(super) const EMPTY: Self = Self {
        word: 0,
        space: 0,
        solo: 0,
        alnum: 0,
        x: !0,
    };
}

/// Classifies the (up to) 64 bytes starting at `base`. Bytes past the end of `text` are
/// classified as `x`, so the engine naturally stops at the end of input.
pub(super) fn classify(text: &[u8], base: usize) -> BlockMasks {
    let chunk = &text[base..];
    if let Ok(chunk) = <&[u8; 64]>::try_from(&chunk[..chunk.len().min(64)]) {
        classify64(chunk)
    } else {
        let mut buf = [0xFF_u8; 64]; // 0xFF classifies as `x`
        buf[..chunk.len()].copy_from_slice(chunk);
        classify64(&buf)
    }
}

fn classify64(chunk: &[u8; 64]) -> BlockMasks {
    // SAFETY: NEON is baseline on aarch64 (this module only compiles there), and all loads
    // are within the 64-byte `chunk`.
    unsafe {
        let lo_lut = vld1q_u8(LO_LUT.as_ptr());
        let hi_lut = vld1q_u8(HI_LUT.as_ptr());
        let mut raw = [vdupq_n_u8(0); 4];
        let mut cls = [vdupq_n_u8(0); 4];
        for i in 0..4 {
            let v = vld1q_u8(chunk.as_ptr().add(16 * i));
            raw[i] = v;
            cls[i] = vandq_u8(
                vqtbl1q_u8(lo_lut, vandq_u8(v, vdupq_n_u8(0x0F))),
                vqtbl1q_u8(hi_lut, vshrq_n_u8(v, 4)),
            );
        }
        let word = movemask(cls.map(|c| vtstq_u8(c, vdupq_n_u8(WORD))));
        let alnum = movemask(cls.map(|c| vtstq_u8(c, vdupq_n_u8(ALNUM))));
        let space = movemask(cls.map(|c| vtstq_u8(c, vdupq_n_u8(T_SPACE))));
        let x = movemask(std::array::from_fn(|i| {
            vorrq_u8(
                vtstq_u8(cls[i], vdupq_n_u8(NEEDS_DFA)),
                vtstq_u8(raw[i], vdupq_n_u8(0x80)), // non-ASCII
            )
        }));
        BlockMasks {
            word,
            space,
            solo: !(word | space | x),
            alnum,
            x,
        }
    }
}

/// Packs four vectors of per-byte 0xFF/0x00 lane masks into a 64-bit mask (bit i = byte i).
fn movemask(v: [uint8x16_t; 4]) -> u64 {
    // SAFETY: NEON is baseline on aarch64.
    unsafe {
        const BIT: [u8; 16] = [1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128];
        let bits = vld1q_u8(BIT.as_ptr());
        // Each pairwise add halves the byte count; after three rounds, byte k of the low
        // half holds the 8-bit mask of input bytes 8k..8k+8 (lanes hold distinct bits, so
        // addition acts as OR and cannot carry).
        let sum0 = vpaddq_u8(vandq_u8(v[0], bits), vandq_u8(v[1], bits));
        let sum1 = vpaddq_u8(vandq_u8(v[2], bits), vandq_u8(v[3], bits));
        let sum = vpaddq_u8(sum0, sum1);
        let sum = vpaddq_u8(sum, sum);
        vgetq_lane_u64::<0>(vreinterpretq_u64_u8(sum))
    }
}

#[cfg(test)]
mod tests {
    use super::{BlockMasks, classify};
    use crate::uax29::word::properties::{ASCII_WORD_BREAK_PROP, WordBreakProperty};

    /// The classification the engine relies on, derived from the authoritative property
    /// table: word/space/solo cover exactly the ASCII bytes whose `WordBreakProperty` is
    /// ALetter/Numeric/ExtendNumLet, WSegSpace, and Other respectively; everything else
    /// (including all non-ASCII) must go to the DFA.
    fn reference(b: u8) -> (bool, bool, bool, bool, bool) {
        if b >= 0x80 {
            return (false, false, false, false, true);
        }
        match ASCII_WORD_BREAK_PROP[b as usize] {
            WordBreakProperty::ALetter | WordBreakProperty::Numeric => {
                (true, false, false, true, false)
            }
            WordBreakProperty::ExtendNumLet => (true, false, false, false, false),
            WordBreakProperty::WSegSpace => (false, true, false, false, false),
            WordBreakProperty::Other => (false, false, true, false, false),
            _ => (false, false, false, false, true),
        }
    }

    /// Every byte value, in every lane position.
    #[test]
    fn classes_match_word_break_table() {
        for b in 0..=255u8 {
            let (word, space, solo, alnum, x) = reference(b);
            for lane in 0..64 {
                let mut block = [b'a'; 64];
                block[lane] = b;
                let masks = classify(&block, 0);
                let bit = |m: u64| m >> lane & 1 != 0;
                let got = (
                    bit(masks.word),
                    bit(masks.space),
                    bit(masks.solo),
                    bit(masks.alnum),
                    bit(masks.x),
                );
                assert_eq!(
                    got,
                    (word, space, solo, alnum, x),
                    "byte {b:#04x} lane {lane}"
                );
            }
        }
    }

    /// Bytes past the end of the input must classify as `x` so the engine stops there.
    #[test]
    fn padding_is_x() {
        for len in 0..=64 {
            let block = vec![b'a'; len];
            let masks = classify(&block, 0);
            let expect_x = if len == 64 { 0 } else { !((1u64 << len) - 1) };
            assert_eq!(masks.x, expect_x, "len {len}");
            assert_eq!(masks.word, !expect_x, "len {len}");
        }
    }

    #[test]
    fn empty_masks_are_inert() {
        assert_eq!(BlockMasks::EMPTY.x, !0);
    }
}
