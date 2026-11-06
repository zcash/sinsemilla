use alloc::vec::Vec;

use group::Group;
use pasta_curves::pallas;

use super::{C, HashDomain, K, MessageWords, lebs2ip_k, tests::bits};

#[test]
fn message_words_match_padded_bits() {
    let msg = bits(0x53494e53454d494c, K * C);
    for len in 0..=K * C {
        let mut padded = msg[..len].to_vec();
        padded.resize(len.div_ceil(K) * K, false);
        let expected: Vec<_> = padded
            .as_chunks::<K>()
            .0
            .iter()
            .map(|chunk| lebs2ip_k(*chunk))
            .collect();
        let actual: Vec<_> = MessageWords::new(msg[..len].iter().copied())
            .map(u32::from)
            .collect();
        assert_eq!(actual, expected, "{len} bits");
    }
}

#[test]
fn message_words_do_not_poll_after_end() {
    for len in [0, 1, K, K + 1] {
        let mut calls = 0;
        let msg = core::iter::from_fn(|| {
            calls += 1;
            assert!(calls <= len + 1);
            (calls <= len).then_some(true)
        });
        let mut words = MessageWords::new(msg);
        assert_eq!(words.by_ref().count(), len.div_ceil(K));
        assert_eq!(words.next(), None);
        assert_eq!(words.next(), None);
    }
}

#[test]
#[should_panic(expected = "longer than K * C bits")]
fn streaming_checks_length_after_bottom() {
    let domain = HashDomain::from_Q(pallas::Point::identity());
    let _ = domain.hash_to_point(core::iter::repeat_n(false, K * C + 1));
}
