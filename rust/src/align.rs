//! Finger-valley ROI alignment (the classic contact-free palmprint ROI: Zhang et al. 2003, as adapted
//! for contactless captures). On the largest hand contour of the preprocessing:
//!
//! 1. distance of each contour point from the hand's centroid, sampled at 400 points and smoothed;
//! 2. fingertips = local maxima standing at least 0.25·r above their surroundings (r = equivalent
//!    radius of the hand, √(area/π)); the four consecutive tips reaching farthest are index…little
//!    (the thumb tip is closer to the palm centre);
//! 3. the three valleys between them = the lowest contour points between consecutive tips;
//! 4. the outer two valleys v₁, v₃ define the palm frame: x along v₁→v₃, y toward the centroid, scale
//!    L = |v₁v₃|; the ROI is a square of side 1.4·L whose top edge sits 0.25·L below the valley line;
//! 5. sampled bilinearly (2×2 supersampling) straight to 138×138.
//!
//! This makes the crop independent of hand position, in-plane rotation and scale. Training uses the
//! same code: `palmprint rois` writes the crops that `training/prepare.py --roi valley` trains on.

use crate::contours::{area, moments, Contour};
use crate::imgproc::GrayImage;
use serde::Serialize;

pub const SAMPLES: usize = 400;
/// how far a fingertip must stand out, as a fraction of r
pub const DEPTH: f64 = 0.25;
pub const SIDE: f64 = 1.4;
pub const GAP: f64 = 0.25;

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct Valleys {
    pub v1: (f64, f64),
    pub v2: (f64, f64),
    pub v3: (f64, f64),
    /// rotation of the palm frame, degrees (0 = valley line horizontal, fingers up)
    pub angle: f64,
    /// |v1 v3| in pixels
    pub scale: f64,
    /// ROI centre in the image
    pub center: (f64, f64),
}

/// Finger valleys and the palm frame, or `None` when three convincing valleys aren't found.
pub fn find_valleys(c: &Contour) -> Option<Valleys> {
    let n = c.points.len();
    if n < SAMPLES {
        return None;
    }
    let m = moments(c);
    if m.m00 <= 0.0 {
        return None;
    }
    let (cx, cy) = (m.m10 / m.m00, m.m01 / m.m00);
    let r = (area(c) / std::f64::consts::PI).sqrt();
    let pts: Vec<(f64, f64)> = (0..SAMPLES).map(|k| { let p = c.points[k * n / SAMPLES]; (p.0 as f64, p.1 as f64) }).collect();
    let raw: Vec<f64> = pts.iter().map(|p| ((p.0 - cx).powi(2) + (p.1 - cy).powi(2)).sqrt()).collect();
    let at = |v: &[f64], i: isize| v[i.rem_euclid(SAMPLES as isize) as usize];
    let d: Vec<f64> = (0..SAMPLES as isize).map(|i| (-2..=2).map(|k| at(&raw, i + k)).sum::<f64>() / 5.0).collect();
    // fingertips: local maxima (within ±10 samples) standing at least 0.25·r above the lowest point
    // within ±60 samples on each side
    let mut tips: Vec<(usize, f64)> = vec![];
    for i in 0..SAMPLES as isize {
        let v = at(&d, i);
        if !(-10..=10).all(|k| k == 0 || at(&d, i + k) < v || (at(&d, i + k) == v && k > 0)) {
            continue;
        }
        let left = (1..=60).map(|k| at(&d, i - k)).fold(f64::MAX, f64::min);
        let right = (1..=60).map(|k| at(&d, i + k)).fold(f64::MAX, f64::min);
        if v - left.max(right) >= DEPTH * r {
            tips.push((i as usize, v));
        }
    }
    if tips.len() < 4 {
        return None;
    }
    // index, middle, ring, little: the four consecutive tips reaching farthest from the palm centre
    // (the thumb tip is closer to it than the fingers)
    let k = tips.len();
    let s0 = (0..k).max_by(|&a, &b| {
        let sum = |s: usize| (0..4).map(|q| tips[(s + q) % k].1).sum::<f64>();
        sum(a).total_cmp(&sum(b))
    })?;
    let four: Vec<usize> = (0..4).map(|q| tips[(s0 + q) % k].0).collect();
    // valley between consecutive tips: the lowest point of d between them
    let valley = |a: usize, b: usize| -> (f64, f64) {
        let len = (b + SAMPLES - a) % SAMPLES;
        let i = (0..=len).map(|t| (a + t) % SAMPLES).min_by(|&x, &y| d[x].total_cmp(&d[y])).unwrap();
        pts[i]
    };
    let dist = |a: (f64, f64), b: (f64, f64)| ((a.0 - b.0).powi(2) + (a.1 - b.1).powi(2)).sqrt();
    let best = Some((valley(four[0], four[1]), valley(four[1], four[2]), valley(four[2], four[3])));
    let (v1, v2, v3) = best?;
    let scale = dist(v1, v3);
    // sanity: plausible size, middle valley between the outer two
    if scale < 0.3 * r || scale > 2.5 * r || dist(v1, v2) + dist(v2, v3) > 1.6 * scale {
        return None;
    }
    let mid = ((v1.0 + v3.0) / 2.0, (v1.1 + v3.1) / 2.0);
    let (ux, uy) = ((v3.0 - v1.0) / scale, (v3.1 - v1.1) / scale);
    // normal pointing from the valley line toward the palm centre
    let (mut nx, mut ny) = (-uy, ux);
    if (cx - mid.0) * nx + (cy - mid.1) * ny < 0.0 {
        nx = -nx;
        ny = -ny;
    }
    let off = (GAP + SIDE / 2.0) * scale;
    let center = (mid.0 + nx * off, mid.1 + ny * off);
    // frame x-axis = normal rotated by +90° (right-handed with y = normal, image coordinates)
    let angle = (-nx).atan2(ny).to_degrees();
    Some(Valleys { v1, v2, v3, angle, scale, center })
}

/// The aligned square ROI, `out`×`out` pixels, bilinear with 2×2 supersampling, zero outside the image.
pub fn aligned_roi(img: &GrayImage, v: &Valleys, out: usize) -> GrayImage {
    let side = SIDE * v.scale;
    let (s, c) = v.angle.to_radians().sin_cos();
    // frame axes: x = (cos, sin), y = (-sin, cos)
    let sample = |x: f64, y: f64| -> f64 {
        let (x0, y0) = (x.floor(), y.floor());
        let (fx, fy) = (x - x0, y - y0);
        let px = |xx: f64, yy: f64| -> f64 {
            if xx < 0.0 || yy < 0.0 || xx >= img.w as f64 || yy >= img.h as f64 { 0.0 } else { img.data[yy as usize * img.w + xx as usize] as f64 }
        };
        px(x0, y0) * (1.0 - fx) * (1.0 - fy) + px(x0 + 1.0, y0) * fx * (1.0 - fy) + px(x0, y0 + 1.0) * (1.0 - fx) * fy + px(x0 + 1.0, y0 + 1.0) * fx * fy
    };
    let mut o = GrayImage::new(out, out, 0);
    for j in 0..out {
        for i in 0..out {
            let mut acc = 0.0;
            for (dy, dx) in [(0.25, 0.25), (0.25, 0.75), (0.75, 0.25), (0.75, 0.75)] {
                let u = ((i as f64 + dx) / out as f64 - 0.5) * side;
                let w = ((j as f64 + dy) / out as f64 - 0.5) * side;
                acc += sample(v.center.0 + u * c - w * s, v.center.1 + u * s + w * c);
            }
            o.data[j * out + i] = (acc / 4.0).round().clamp(0.0, 255.0) as u8;
        }
    }
    o
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::contours::outer_contours;

    /// A synthetic hand mask: palm ellipse + four fingers (+ a thumb), rotated by `deg`.
    fn hand(deg: f64, w: usize, h: usize) -> GrayImage {
        let (cx, cy) = (w as f64 / 2.0, h as f64 / 2.0 + 40.0);
        let (s, c) = deg.to_radians().sin_cos();
        let mut g = GrayImage::new(w, h, 0);
        for y in 0..h {
            for x in 0..w {
                // back-rotate into the upright hand frame
                let (dx, dy) = (x as f64 - cx, y as f64 - cy);
                let (u, v) = (dx * c + dy * s, -dx * s + dy * c);
                let palm = (u / 150.0).powi(2) + (v / 175.0).powi(2) <= 1.0;
                let finger = (0..4).any(|k| { let fx = -105.0 + 70.0 * k as f64; (u - fx).abs() <= 22.0 && (-330.0..=-120.0).contains(&v) });
                let thumb = { let (tu, tv) = (u + 150.0, v - 10.0); let (a, b) = (tu * 0.8 + tv * 0.6, -tu * 0.6 + tv * 0.8); (-150.0..=0.0).contains(&a) && b.abs() <= 25.0 };
                if palm || finger || thumb {
                    g.data[y * w + x] = 255;
                }
            }
        }
        g
    }

    #[test]
    fn valleys_follow_rotation() {
        for deg in [-20.0, -8.0, 0.0, 12.0, 25.0] {
            let g = hand(deg, 800, 800);
            let cs = outer_contours(&g);
            let big = cs.iter().max_by(|a, b| area(a).total_cmp(&area(b))).unwrap();
            let v = find_valleys(big).unwrap_or_else(|| panic!("no valleys at {deg}°"));
            assert!((v.angle - deg).abs() < 3.0, "angle {:.1} vs {deg}", v.angle);
            assert!((v.scale - 140.0).abs() < 12.0, "scale {:.1}", v.scale);
        }
    }

    #[test]
    fn roi_is_rotation_invariant() {
        // the same hand at two rotations gives (nearly) the same ROI
        let rois: Vec<GrayImage> = [0.0, 20.0].iter().map(|&deg| {
            let g = hand(deg, 800, 800);
            let cs = outer_contours(&g);
            let big = cs.iter().max_by(|a, b| area(a).total_cmp(&area(b))).unwrap();
            aligned_roi(&g, &find_valleys(big).unwrap(), 138)
        }).collect();
        let diff = rois[0].data.iter().zip(&rois[1].data).filter(|(a, b)| a.abs_diff(**b) > 64).count();
        assert!(diff < 138 * 138 / 50, "{diff} pixels differ");
    }

    #[test]
    fn no_fingers_no_valleys() {
        let mut g = GrayImage::new(400, 400, 0);
        for y in 0..400 { for x in 0..400 { if ((x as f64 - 200.0) / 150.0).powi(2) + ((y as f64 - 200.0) / 120.0).powi(2) <= 1.0 { g.data[y * 400 + x] = 255; } } }
        let cs = outer_contours(&g);
        assert!(find_valleys(&cs[0]).is_none());
    }
}
