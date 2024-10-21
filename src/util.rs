/// Take two key value pairs and return them sorted by their key
/// 
/// Note: We only use `PartialEq` here. For f64 in particular, this implies an
/// unstable sort order if one of the keys is NaN.
pub(crate) fn minmax_pair<K: PartialOrd + Copy, V: Copy>(a: &(K, V), b: &(K, V)) -> ((K, V), (K, V)) {
    if a.0 < b.0 { (*a,*b) } else { (*b,*a) }
}

/// Take two key value pairs and return the one with the smaller key
/// 
/// Note: We only use `PartialEq` here. For f64 in particular, this implies an
/// unstable sort order if one of the keys is NaN.
pub(crate) fn min_pair<K: PartialOrd + Copy, V: Copy>(a: &(K, V), b: &(K, V)) -> (K, V) {
    if a.0 < b.0 { *a } else { *b }
}

/// Take two key value pairs and return the one with the larger key
/// 
/// Note: We only use `PartialEq` here. For f64 in particular, this implies an
/// unstable sort order if one of the keys is NaN.
pub(crate) fn max_pair<K: PartialOrd + Copy, V: Copy>(a: &(K, V), b: &(K, V)) -> (K, V) {
    if a.0 < b.0 { *b } else { *a }
}

#[macro_export]
/// Macro for reporting lost rays, if the corresponding feature is enabled
/// (otherwise does the same as the ? operator)
macro_rules! unwrap_lost_ray {
    ($e: expr, $msg: literal) => {
        if cfg!(feature = "report-lost-rays") {
            match $e {
                Some(x) => x,
                None => {
                    eprintln!("LOST_RAY: {}", $msg);
                    return None;
                },
            }
        }
        else {
            $e?
        }
    };
}