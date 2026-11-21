/-
Palmprint matcher (`find_best_matches`) — formal properties. Lean 4.31, kernel only (no Mathlib).
Scores are fixed-point naturals (cosine × 10⁴ shifted to ≥ 0); the threshold uses the same scale.
-/

/-- "Match found": some returned score reaches the threshold. -/
def matchFound (l : List Nat) (t : Nat) : Bool := l.any (fun s => decide (t ≤ s))

/-- The returned list: the top-k scores at or above the threshold, or all top-k if none are. -/
def chosen (l : List Nat) (t : Nat) : List Nat :=
  if (l.filter (fun s => decide (t ≤ s))).isEmpty then l else l.filter (fun s => decide (t ≤ s))

/-- The UI always has something to show when the database is non-empty. -/
theorem chosen_nonempty (l : List Nat) (t : Nat) (h : l ≠ []) : chosen l t ≠ [] := by
  unfold chosen
  split
  · exact h
  · rename_i hne; intro hnil; simp_all

/-- The fallback never manufactures a match: the verdict on the returned list equals the verdict
    on the top-k itself. -/
theorem chosen_found_iff (l : List Nat) (t : Nat) : matchFound (chosen l t) t = matchFound l t := by
  unfold chosen matchFound
  split
  · rfl
  · rename_i hne
    simp only [List.isEmpty_iff] at hne
    have : (List.filter (fun s => decide (t ≤ s)) l).any (fun s => decide (t ≤ s)) = true := by
      obtain ⟨x, hx⟩ := List.exists_mem_of_ne_nil _ hne
      exact List.any_eq_true.mpr ⟨x, hx, (List.mem_filter.mp hx).2⟩
    rw [this]
    obtain ⟨x, hx⟩ := List.exists_mem_of_ne_nil _ hne
    exact (List.any_eq_true.mpr ⟨x, (List.mem_filter.mp hx).1, (List.mem_filter.mp hx).2⟩).symm

/-- Raising the threshold never turns "no match" into "match". -/
theorem found_antitone (l : List Nat) (t t' : Nat) (ht : t ≤ t') (h : matchFound l t' = true) :
    matchFound l t = true := by
  unfold matchFound at *
  obtain ⟨x, hx, hxt⟩ := List.any_eq_true.mp h
  exact List.any_eq_true.mpr ⟨x, hx, by simp at hxt ⊢; omega⟩

/-- When a match is found, every returned score reaches the threshold (no below-threshold padding). -/
theorem found_all_above (l : List Nat) (t : Nat) (h : matchFound l t = true) :
    ∀ s ∈ chosen l t, t ≤ s := by
  unfold chosen
  split
  · rename_i hemp
    unfold matchFound at h
    obtain ⟨x, hx, hxt⟩ := List.any_eq_true.mp h
    simp only [List.isEmpty_iff, List.filter_eq_nil_iff] at hemp
    exact absurd hxt (by simpa using hemp x hx)
  · intro s hs; simpa using (List.mem_filter.mp hs).2

/-! ## Threshold calibration (training/metrics.py `far_threshold`)

Impostor scores sorted in descending order; the threshold is chosen strictly above the score at
index k = ⌊α·N⌋. Then at most k impostor pairs are accepted (score ≥ threshold), so the calibration
FAR is at most α. Scores are the same fixed-point naturals as above. -/

theorem filter_ge_nil (l : List Nat) (t : Nat) (h : ∀ x ∈ l, x < t) :
    l.filter (fun s => decide (t ≤ s)) = [] := by
  rw [List.filter_eq_nil_iff]
  intro x hx
  have := h x hx
  simp
  omega

/-- At most k scores reach a threshold placed above the k-th (0-based) largest score. -/
theorem far_threshold_bound (l : List Nat) (hs : l.Pairwise (· ≥ ·)) (k : Nat) (hk : k < l.length)
    (t : Nat) (ht : l[k] < t) : (l.filter (fun s => decide (t ≤ s))).length ≤ k := by
  induction l generalizing k with
  | nil => simp at hk
  | cons a l ih =>
    rw [List.pairwise_cons] at hs
    cases k with
    | zero =>
      simp at ht
      have : (a :: l).filter (fun s => decide (t ≤ s)) = [] := by
        apply filter_ge_nil
        intro x hx
        simp at hx
        rcases hx with rfl | hx
        · exact ht
        · have := hs.1 x hx
          omega
      simp [this]
    | succ k =>
      simp at ht hk
      have := ih hs.2 k (by omega) ht
      simp only [List.filter_cons]
      split <;> simp <;> omega

/-- With k = ⌊p·N/q⌋ accepted impostors out of N, the empirical FAR c/N is at most α = p/q. -/
theorem far_le_alpha (c N p q : Nat) (h : c ≤ p * N / q) : c * q ≤ p * N :=
  Nat.le_trans (Nat.mul_le_mul_right q h) (Nat.div_mul_le_self (p * N) q)

/-! ## CNN + palm-ridge fusion (src/ridges.rs `fuse`, training/calibrate_fusion.py)

Fixed point: cosine c and ridge similarity R − d (distance d ≤ R) on the scale R, weight w out of
W. fused = w·c + (W − w)·(R − d). A higher cosine or a lower ridge distance never lowers the fused
score, so neither feature can push a better-matching image below a worse one. -/

def fuse (W w R c d : Nat) : Nat := w * c + (W - w) * (R - d)

theorem fuse_mono (W w R c₁ c₂ d₁ d₂ : Nat) (hc : c₁ ≤ c₂) (hd : d₂ ≤ d₁) :
    fuse W w R c₁ d₁ ≤ fuse W w R c₂ d₂ := by
  unfold fuse
  exact Nat.add_le_add (Nat.mul_le_mul_left w hc) (Nat.mul_le_mul_left (W - w) (by omega))

/-- w = W is the CNN alone; w = 0 is the ridges alone (up to the scale W). -/
theorem fuse_ends (W R c d : Nat) : fuse W W R c d = W * c ∧ fuse W 0 R c d = W * (R - d) := by
  unfold fuse
  simp

/-! ## ArcFace margin warm-up (training/train.py, models.SubCenterArcFace.set_margin)

m_e = m · min(1, e / W) for epoch e and warm-up length W (fixed point: ⌊m·e/W⌋ before W). The margin
never exceeds m, never decreases from one epoch to the next, starts at 0 and is m from epoch W on. -/

def warm (m W e : Nat) : Nat := if W ≤ e then m else m * e / W

theorem warm_le (m W e : Nat) : warm m W e ≤ m := by
  unfold warm
  split
  · exact Nat.le_refl m
  · rename_i h
    exact Nat.le_trans (Nat.div_le_div_right (Nat.mul_le_mul_left m (Nat.le_of_lt (Nat.lt_of_not_le h)))) (by
      rcases Nat.eq_zero_or_pos W with hW | hW
      · simp [hW]
      · exact Nat.le_of_eq (Nat.mul_div_cancel m hW))

theorem warm_mono (m W e : Nat) : warm m W e ≤ warm m W (e + 1) := by
  unfold warm
  by_cases h1 : W ≤ e
  · have : W ≤ e + 1 := by omega
    simp [h1, this]
  · by_cases h2 : W ≤ e + 1
    · simp only [h1, h2, if_false, if_true]
      have := warm_le m W e
      unfold warm at this
      simpa [h1] using this
    · simp only [h1, h2, if_false]
      exact Nat.div_le_div_right (Nat.mul_le_mul_left m (by omega))

theorem warm_start (m W : Nat) (hW : 0 < W) : warm m W 0 = 0 := by
  unfold warm
  rw [if_neg (by omega)]
  simp

theorem warm_full (m W e : Nat) (h : W ≤ e) : warm m W e = m := by
  simp [warm, h]
