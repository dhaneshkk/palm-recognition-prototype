//! Port of `preprocess/preprocessor.py`, step for step:
//! gray (prototype's RGB2GRAY-on-BGR) → CLAHE(2.0, 5×5) → medianBlur(k=1, a no-op) → threshold > 80 →
//! outer contours → centroid of the largest (by area) → 276×276 crop (shifted at left/top edges,
//! truncated at right/bottom edges, exactly as the Python) → INTER_LINEAR to 138×138 → /255.

use crate::contours::{area, moments, outer_contours};
use crate::imgproc::{clahe, resize_linear, GrayImage, RgbImage};
use serde::Serialize;

pub const CLAHE_CLIP: f64 = 2.0;
pub const CLAHE_TILES: usize = 5;
pub const ROI: usize = 276;
pub const TARGET: usize = 138;
pub const THRESHOLD: u8 = 80;

/// How the palm region is cut: the prototype's centroid crop (bit-exact with preprocessor.py), or
/// aligned to the finger valleys (`align.rs`), falling back to the centroid crop when the valleys
/// can't be found. A model is trained and run with the same mode (stored in its model card).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, serde::Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum RoiMode {
    #[default]
    Centroid,
    Valley,
}

impl std::str::FromStr for RoiMode {
    type Err = anyhow::Error;
    fn from_str(s: &str) -> anyhow::Result<Self> {
        match s.trim().to_lowercase().as_str() {
            "centroid" => Ok(Self::Centroid),
            "valley" => Ok(Self::Valley),
            o => anyhow::bail!("roi mode must be centroid or valley, not {o:?}"),
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct Trace {
    pub n_contours: usize,
    pub largest_area: f64,
    pub cx: i64,
    pub cy: i64,
    pub roi_w: usize,
    pub roi_h: usize,
    /// "centroid", "valley", or "centroid (valleys not found)"
    pub roi_mode: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub valleys: Option<crate::align::Valleys>,
}

pub struct Stages {
    pub gray: GrayImage,
    pub enhanced: GrayImage,
    pub binary: GrayImage,
    pub roi: GrayImage,
    pub resized: GrayImage,
    /// 138×138, values in [0, 1]
    pub normalized: Vec<f32>,
    pub trace: Trace,
}

/// Returns `None` where the Python returns None (no contours, zero-area largest contour, empty ROI).
pub fn preprocess(img: &RgbImage) -> Option<Stages> {
    preprocess_mode(img, RoiMode::Centroid)
}

/// As `preprocess`, cutting the palm region the given way.
pub fn preprocess_mode(img: &RgbImage, mode: RoiMode) -> Option<Stages> {
    let gray = img.to_gray_prototype();
    let enhanced = clahe(&gray, CLAHE_CLIP, CLAHE_TILES, CLAHE_TILES); // medianBlur(k=1) leaves it unchanged
    let binary = GrayImage { w: enhanced.w, h: enhanced.h, data: enhanced.data.iter().map(|&v| if v > THRESHOLD { 255 } else { 0 }).collect() };
    let contours = outer_contours(&binary);
    // Python's max(..., key=contourArea) keeps the FIRST maximum in contour order
    let mut best: Option<(usize, f64)> = None;
    for (i, c) in contours.iter().enumerate() {
        let a = area(c);
        if best.map_or(true, |(_, b)| a > b) {
            best = Some((i, a));
        }
    }
    let (bi, barea) = best?;
    let m = moments(&contours[bi]);
    if m.m00 == 0.0 {
        return None;
    }
    let (cx, cy) = ((m.m10 / m.m00) as i64, (m.m01 / m.m00) as i64); // int() truncates toward zero
    let half = (ROI / 2) as i64;
    let x = (cx - half).max(0) as usize;
    let y = (cy - half).max(0) as usize;
    let x_end = (x + ROI).min(enhanced.w);
    let y_end = (y + ROI).min(enhanced.h);
    if x >= x_end || y >= y_end {
        return None;
    }
    let roi = enhanced.crop(x, y, x_end, y_end);
    let valleys = if mode == RoiMode::Valley { crate::align::find_valleys(&contours[bi]) } else { None };
    let (resized, roi_mode) = match &valleys {
        Some(v) => (crate::align::aligned_roi(&enhanced, v, TARGET), "valley".to_string()),
        None if mode == RoiMode::Valley => (resize_linear(&roi, TARGET, TARGET), "centroid (valleys not found)".to_string()),
        None => (resize_linear(&roi, TARGET, TARGET), "centroid".to_string()),
    };
    let normalized = resized.data.iter().map(|&v| v as f32 / 255.0).collect();
    let trace = Trace { n_contours: contours.len(), largest_area: barea, cx, cy, roi_w: roi.w, roi_h: roi.h, roi_mode, valleys };
    Some(Stages { gray, enhanced, binary, roi, resized, normalized, trace })
}

/// Decode like `cv2.imread(path)` (alpha dropped, 3 channels).
pub fn load(path: &std::path::Path) -> anyhow::Result<RgbImage> {
    RgbImage::load(path)
}

pub fn load_bytes(bytes: &[u8]) -> anyhow::Result<RgbImage> {
    Ok(RgbImage::from_dynamic(&image::load_from_memory(bytes)?))
}
