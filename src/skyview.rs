use ndarray::{s, Array2, Array3, ArrayView2, Zip};
use numpy::{IntoPyArray, PyArray2, PyArray3, PyReadonlyArray2};
use pyo3::prelude::*;

// Constants
const SHADOW_FLAG_INTERMEDIATE: f64 = 1.0; // Intermediate value indicating shadow
const SUNLIT_FLAG_INTERMEDIATE: f64 = 0.0; // Intermediate value indicating sunlit
const FINAL_SUNLIT_VALUE: f64 = 1.0; // Final output value representing sunlit

#[pyclass]
/// Holds the results of the Sky View Factor calculation.
pub struct SvfResult {
    #[pyo3(get)]
    pub svf: Py<PyArray2<f64>>,
    #[pyo3(get)]
    pub svf_east: Py<PyArray2<f64>>,
    #[pyo3(get)]
    pub svf_south: Py<PyArray2<f64>>,
    #[pyo3(get)]
    pub svf_west: Py<PyArray2<f64>>,
    #[pyo3(get)]
    pub svf_north: Py<PyArray2<f64>>,
    #[pyo3(get)]
    pub svf_veg: Py<PyArray2<f64>>,
    #[pyo3(get)]
    pub svf_veg_east: Py<PyArray2<f64>>,
    #[pyo3(get)]
    pub svf_veg_south: Py<PyArray2<f64>>,
    #[pyo3(get)]
    pub svf_veg_west: Py<PyArray2<f64>>,
    #[pyo3(get)]
    pub svf_veg_north: Py<PyArray2<f64>>,
    #[pyo3(get)]
    pub svf_aniso_veg: Py<PyArray2<f64>>, // svfaveg in Python
    #[pyo3(get)]
    pub svf_aniso_veg_east: Py<PyArray2<f64>>, // svfEaveg
    #[pyo3(get)]
    pub svf_aniso_veg_south: Py<PyArray2<f64>>, // svfSaveg
    #[pyo3(get)]
    pub svf_aniso_veg_west: Py<PyArray2<f64>>, // svfWaveg
    #[pyo3(get)]
    pub svf_aniso_veg_north: Py<PyArray2<f64>>, // svfNaveg
    #[pyo3(get)]
    /// Shadow matrix for buildings (rows, cols, patches)
    pub shadow_matrix: Py<PyArray3<f64>>,
    #[pyo3(get)]
    /// Shadow matrix for vegetation (rows, cols, patches)
    pub veg_shadow_matrix: Py<PyArray3<f64>>,
    #[pyo3(get)]
    /// Vegetation Blocking Building Shadow matrix (rows, cols, patches)
    pub vbshvegsh_matrix: Py<PyArray3<f64>>,
}

/// Calculates the weight for a given sky annulus segment.
/// Corresponds to annulus_weight in Python.
fn calculate_annulus_weight(altitude_deg: f64, patches_in_annulus: f64) -> f64 {
    let n = 90.0; // Number of altitude steps (hardcoded in Python)
    let step_rad = (360.0 / patches_in_annulus).to_radians();
    let annulus_index = 91.0 - altitude_deg; // 1-based index from zenith
    let w = (1.0 / (2.0 * std::f64::consts::PI))
        * (std::f64::consts::PI / (2.0 * n)).sin()
        * (std::f64::consts::PI * (2.0 * annulus_index - 1.0) / (2.0 * n)).sin();
    step_rad * w
}

/// Generates sky patch definitions for the 153-patch scheme.
/// Corresponds to create_patches(patch_option=2) in Python.
fn create_patches_153() -> (
    Vec<f64>, // skyvaultalt (center altitude degrees) - Not directly used in loop logic below
    Vec<f64>, // skyvaultazi (center azimuth degrees) - Not directly used
    Vec<f64>, // annulino (lower altitude bound index)
    Vec<f64>, // skyvaultaltint (altitude bands degrees)
    Vec<f64>, // aziinterval (number of patches per altitude band)
    Vec<f64>, // skyvaultaziint (azimuth degrees per patch in band) - Calculated below
    Vec<f64>, // azistart (starting azimuth for band)
) {
    // Based on common 153 patch distribution (Lindberg et al., 2008 or similar)
    // 8 altitude bands, 1 zenith patch (handled implicitly by loops)
    let skyvaultaltint = vec![6., 18., 30., 42., 54., 66., 78., 90.]; // Upper bounds of altitude bands
    let aziinterval = vec![8., 12., 16., 20., 24., 24., 24., 24.]; // Patches per band
    let azistart = vec![22.5, 7.5, 11.25, 9., 7.5, 7.5, 7.5, 7.5]; // Start azimuth for each band

    // annulino seems to relate altitude bands to a finer scale (0-90).
    // Let's assume it corresponds to the lower bound of the altitude band.
    let mut annulino = vec![0.0; skyvaultaltint.len() + 1]; // Need one extra for the last loop range
    for i in 0..skyvaultaltint.len() {
        annulino[i + 1] = skyvaultaltint[i]; // Use upper bound of previous as lower bound of current
    }
    // Manually adjust based on Python logic needs (k ranges from annulino[i]+1 to annulino[i+1])
    // The Python code uses k as altitude degrees directly in annulus_weight.
    // Let's refine annulino to be the lower bound degree for each band for clarity.
    let mut annulino_refined = vec![0.0; skyvaultaltint.len() + 1];
    annulino_refined[0] = 0.0; // Zenith patch lower bound (implicitly)
    for i in 0..skyvaultaltint.len() {
        annulino_refined[i + 1] = skyvaultaltint[i]; // Upper bound of band i is lower bound for weight loop k
    }

    // skyvaultalt and skyvaultazi are not strictly needed for the loop logic if we calculate on the fly
    let skyvaultalt = vec![]; // Placeholder
    let skyvaultazi = vec![]; // Placeholder

    (
        skyvaultalt,
        skyvaultazi,
        annulino_refined, // Use the refined version
        skyvaultaltint,
        aziinterval,
        vec![], // skyvaultaziint calculated later
        azistart,
    )
}

// Placeholder for a simplified shadow function tailored for SVF
// This needs to be implemented based on shadowing.rs logic but returning only needed maps.
// For now, it will just return zeros. A proper implementation is crucial.
fn calculate_svf_shadows(
    dsm: ArrayView2<f64>,
    veg_canopy_dsm: ArrayView2<f64>,
    veg_trunk_dsm: ArrayView2<f64>,
    azimuth_deg: f64,
    altitude_deg: f64,
    scale: f64,
    amaxvalue: f64,
    bush: ArrayView2<f64>,
    usevegdem: bool,
) -> (Array2<f64>, Array2<f64>, Array2<f64>) {
    let shape = dsm.dim();
    // TODO: Implement actual shadow calculation based on shadowing.rs
    // This implementation should be parallelized internally like shadowingfunction_wallheight_25

    // --- TEMPORARY PLACEHOLDER ---
    // Simulating the structure of the output needed, but with dummy data.
    // In a real implementation, call a modified version of the shadow casting loop.

    let sh = Array2::<f64>::ones(shape); // Building shadow (1.0 = sunlit)
    let vegsh = Array2::<f64>::ones(shape); // Vegetation shadow (1.0 = sunlit)
    let vbshvegsh = Array2::<f64>::ones(shape); // Veg blocking building shadow (1.0 = potential building shadow blocked by veg)

    // --- END PLACEHOLDER ---

    // Return inverted flags (0 = shadow, 1 = sunlit) as expected by SVF logic
    (
        sh.mapv(|v| FINAL_SUNLIT_VALUE - v), // 1.0 - sh (0=shadow, 1=sun)
        vegsh.mapv(|v| FINAL_SUNLIT_VALUE - v), // 1.0 - vegsh
        vbshvegsh.mapv(|v| FINAL_SUNLIT_VALUE - v), // 1.0 - vbshvegsh
    )
}

#[pyfunction]
/// Calculates Sky View Factor (SVF) using the 153-patch method.
///
/// Corresponds to svfForProcessing153 in Python.
///
/// # Arguments
/// * `dsm` - Digital Surface Model (buildings, ground)
/// * `veg_canopy_dsm` - Vegetation canopy height DSM
/// * `veg_trunk_dsm` - Vegetation trunk height DSM (defines bottom of canopy)
/// * `scale` - Pixel size (meters)
/// * `usevegdem` - Boolean flag indicating whether to include vegetation in calculation.
///
/// # Returns
/// * `SvfResult` struct containing various SVF maps.
#[allow(clippy::too_many_arguments, clippy::too_many_lines)] // Match Python function signature
pub fn calculate_svf_153(
    py: Python,
    dsm_py: PyReadonlyArray2<f64>,
    veg_canopy_dsm_py: PyReadonlyArray2<f64>, // vegdem in Python
    veg_trunk_dsm_py: PyReadonlyArray2<f64>,  // vegdem2 in Python
    scale: f64,
    usevegdem: bool,
) -> PyResult<PyObject> {
    let dsm = dsm_py.as_array();
    let veg_canopy_dsm_in = veg_canopy_dsm_py.as_array();
    let veg_trunk_dsm_in = veg_trunk_dsm_py.as_array();
    let shape = dsm.dim();
    let (rows, cols) = (shape.0, shape.1);

    // --- Initialization ---
    let mut svf = Array2::<f64>::zeros(shape);
    let mut svf_e = Array2::<f64>::zeros(shape);
    let mut svf_s = Array2::<f64>::zeros(shape);
    let mut svf_w = Array2::<f64>::zeros(shape);
    let mut svf_n = Array2::<f64>::zeros(shape);
    let mut svf_veg = Array2::<f64>::zeros(shape);
    let mut svf_veg_e = Array2::<f64>::zeros(shape);
    let mut svf_veg_s = Array2::<f64>::zeros(shape);
    let mut svf_veg_w = Array2::<f64>::zeros(shape);
    let mut svf_veg_n = Array2::<f64>::zeros(shape);
    let mut svf_aniso_veg = Array2::<f64>::zeros(shape); // svfaveg
    let mut svf_aniso_veg_e = Array2::<f64>::zeros(shape);
    let mut svf_aniso_veg_s = Array2::<f64>::zeros(shape);
    let mut svf_aniso_veg_w = Array2::<f64>::zeros(shape);
    let mut svf_aniso_veg_n = Array2::<f64>::zeros(shape);

    // Calculate amaxvalue (approximate percentile like NumPy)
    // For simplicity, using max. A true percentile might need external crate or sampling.
    let vegmax = veg_canopy_dsm_in
        .iter()
        .fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let dsm_max = dsm.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b)); // Simple max for now
    let amaxvalue = dsm_max.max(vegmax); // Use max as proxy for percentile

    // Adjust vegetation DEMs relative to DSM ground height
    let mut veg_canopy_dsm = Array2::<f64>::zeros(shape);
    let mut veg_trunk_dsm = Array2::<f64>::zeros(shape);
    Zip::from(&mut veg_canopy_dsm)
        .and(&mut veg_trunk_dsm)
        .and(&dsm)
        .and(&veg_canopy_dsm_in)
        .and(&veg_trunk_dsm_in)
        .par_for_each(|vc, vt, &d, &vc_in, &vt_in| {
            let vc_abs = vc_in + d;
            let vt_abs = vt_in + d;
            *vc = if vc_abs == d { 0.0 } else { vc_abs }; // Set to 0 if only ground height
            *vt = if vt_abs == d { 0.0 } else { vt_abs };
        });

    // Bush separation (low vegetation) - Assuming trunk DSM represents ground under canopy
    // Python: bush = np.logical_not((vegdem2 * vegdem)) * vegdem
    // This translates to: bush = (veg_trunk_dsm == 0) * veg_canopy_dsm
    // Let's refine based on the likely intent: identify areas with canopy but no trunk height defined *above ground*
    // A simpler interpretation: bush layer is where veg_canopy > 0 and veg_trunk == 0 (relative to ground)
    let mut bush = Array2::<f64>::zeros(shape);
    Zip::from(&mut bush)
        .and(&veg_canopy_dsm) // Use adjusted canopy height
        .and(&veg_trunk_dsm) // Use adjusted trunk height
        .par_for_each(|b, &vc, &vt| {
            // If canopy exists but trunk is at ground level (0 after adjustment)
            *b = if vc > 0.0 && vt == 0.0 { vc } else { 0.0 };
        });

    // --- Patch Creation ---
    let (
        _skyvaultalt,
        _skyvaultazi,
        annulino,
        skyvaultaltint,
        aziinterval,
        _skyvaultaziint_calc,
        azistart,
    ) = create_patches_153();

    let skyvaultaziint: Vec<f64> = aziinterval.iter().map(|&patches| 360.0 / patches).collect();
    let total_patches = aziinterval.iter().map(|&x| x as usize).sum();

    // Calculate all azimuth angles beforehand
    let mut azimuth_angles = Vec::with_capacity(total_patches);
    for j in 0..skyvaultaltint.len() {
        for k in 0..aziinterval[j] as i32 {
            let mut azi = (k as f64) * skyvaultaziint[j] + azistart[j];
            if azi >= 360.0 {
                azi -= 360.0;
            }
            azimuth_angles.push(azi);
        }
    }

    // Shadow matrices (rows, cols, patches)
    let mut shadow_matrix = Array3::<f64>::zeros((rows, cols, total_patches));
    let mut veg_shadow_matrix = Array3::<f64>::zeros((rows, cols, total_patches));
    let mut vbshvegsh_matrix = Array3::<f64>::zeros((rows, cols, total_patches));

    // --- Main Loop ---
    let mut patch_index = 0;
    // Iterate through altitude bands
    for i in 0..skyvaultaltint.len() {
        let altitude_band_upper_deg = skyvaultaltint[i]; // This is the upper edge altitude
                                                         // Calculate representative altitude for the band (e.g., midpoint) for shadow casting
        let altitude_band_lower_deg = if i == 0 { 0.0 } else { skyvaultaltint[i - 1] };
        let altitude_deg_for_shadow = (altitude_band_lower_deg + altitude_band_upper_deg) / 2.0;

        let num_azi_intervals = aziinterval[i] as usize;
        let patches_in_annulus_aniso = (aziinterval[i] / 2.0).ceil(); // For anisotropic SVF

        // Iterate through azimuth angles within the current altitude band
        for _j in 0..num_azi_intervals {
            let azimuth_deg = azimuth_angles[patch_index];

            // --- Shadow Calculation ---
            // TODO: Replace placeholder with actual shadow calculation call
            let (sh, vegsh, vbshvegsh) = calculate_svf_shadows(
                dsm,
                veg_canopy_dsm.view(),
                veg_trunk_dsm.view(),
                azimuth_deg,
                altitude_deg_for_shadow, // Use representative altitude
                scale,
                amaxvalue,
                bush.view(),
                usevegdem,
            );

            // Store shadow results (0=shadow, 1=sunlit)
            shadow_matrix.slice_mut(s![.., .., patch_index]).assign(&sh);
            if usevegdem {
                veg_shadow_matrix
                    .slice_mut(s![.., .., patch_index])
                    .assign(&vegsh);
                vbshvegsh_matrix
                    .slice_mut(s![.., .., patch_index])
                    .assign(&vbshvegsh);
            }

            // --- SVF Accumulation ---
            // Iterate through the finer altitude steps (k) within the current band for weighting
            let alt_lower_bound = annulino[i]; // Lower altitude degree for this band's weight calculation
            let alt_upper_bound = annulino[i + 1]; // Upper altitude degree

            for k_alt_deg in (alt_lower_bound as i32 + 1)..=(alt_upper_bound as i32) {
                let k_alt_deg_f64 = k_alt_deg as f64;

                // Calculate weights
                let weight = calculate_annulus_weight(k_alt_deg_f64, aziinterval[i]);
                let weight_aniso =
                    calculate_annulus_weight(k_alt_deg_f64, patches_in_annulus_aniso);

                // Accumulate isotropic SVF (buildings only)
                // svf = svf + weight * sh (where sh is 0 for shadow, 1 for sun)
                // Equivalent to: svf = svf + weight * (1 - shadow_flag)
                // Since our sh is already 0 for shadow, 1 for sun:
                Zip::from(&mut svf)
                    .and(&sh)
                    .par_for_each(|svf_val, &sh_val| *svf_val += weight * sh_val);

                // Accumulate anisotropic SVF (buildings only)
                let azi_rad = azimuth_deg.to_radians();
                // East (0 <= azi < 180)
                if azi_rad >= 0.0 && azi_rad < std::f64::consts::PI {
                    Zip::from(&mut svf_e)
                        .and(&sh)
                        .par_for_each(|svf_val, &sh_val| *svf_val += weight_aniso * sh_val);
                }
                // South (90 <= azi < 270)
                if azi_rad >= std::f64::consts::FRAC_PI_2
                    && azi_rad < 3.0 * std::f64::consts::FRAC_PI_2
                {
                    Zip::from(&mut svf_s)
                        .and(&sh)
                        .par_for_each(|svf_val, &sh_val| *svf_val += weight_aniso * sh_val);
                }
                // West (180 <= azi < 360)
                if azi_rad >= std::f64::consts::PI && azi_rad < std::f64::consts::TAU {
                    Zip::from(&mut svf_w)
                        .and(&sh)
                        .par_for_each(|svf_val, &sh_val| *svf_val += weight_aniso * sh_val);
                }
                // North (270 <= azi < 360 or 0 <= azi < 90)
                if azi_rad >= 3.0 * std::f64::consts::FRAC_PI_2
                    || azi_rad < std::f64::consts::FRAC_PI_2
                {
                    Zip::from(&mut svf_n)
                        .and(&sh)
                        .par_for_each(|svf_val, &sh_val| *svf_val += weight_aniso * sh_val);
                }

                // Accumulate vegetation SVFs if enabled
                if usevegdem {
                    // svfveg = svfveg + weight * vegsh
                    Zip::from(&mut svf_veg)
                        .and(&vegsh)
                        .par_for_each(|svf_val, &vegsh_val| *svf_val += weight * vegsh_val);
                    // svfaveg = svfaveg + weight * vbshvegsh
                    Zip::from(&mut svf_aniso_veg)
                        .and(&vbshvegsh)
                        .par_for_each(|svf_val, &vbsh_val| *svf_val += weight * vbsh_val);

                    // Anisotropic vegetation SVFs
                    // East
                    if azi_rad >= 0.0 && azi_rad < std::f64::consts::PI {
                        Zip::from(&mut svf_veg_e).and(&vegsh).par_for_each(
                            |svf_val, &vegsh_val| *svf_val += weight_aniso * vegsh_val,
                        );
                        Zip::from(&mut svf_aniso_veg_e)
                            .and(&vbshvegsh)
                            .par_for_each(|svf_val, &vbsh_val| *svf_val += weight_aniso * vbsh_val);
                    }
                    // South
                    if azi_rad >= std::f64::consts::FRAC_PI_2
                        && azi_rad < 3.0 * std::f64::consts::FRAC_PI_2
                    {
                        Zip::from(&mut svf_veg_s).and(&vegsh).par_for_each(
                            |svf_val, &vegsh_val| *svf_val += weight_aniso * vegsh_val,
                        );
                        Zip::from(&mut svf_aniso_veg_s)
                            .and(&vbshvegsh)
                            .par_for_each(|svf_val, &vbsh_val| *svf_val += weight_aniso * vbsh_val);
                    }
                    // West
                    if azi_rad >= std::f64::consts::PI && azi_rad < std::f64::consts::TAU {
                        Zip::from(&mut svf_veg_w).and(&vegsh).par_for_each(
                            |svf_val, &vegsh_val| *svf_val += weight_aniso * vegsh_val,
                        );
                        Zip::from(&mut svf_aniso_veg_w)
                            .and(&vbshvegsh)
                            .par_for_each(|svf_val, &vbsh_val| *svf_val += weight_aniso * vbsh_val);
                    }
                    // North
                    if azi_rad >= 3.0 * std::f64::consts::FRAC_PI_2
                        || azi_rad < std::f64::consts::FRAC_PI_2
                    {
                        Zip::from(&mut svf_veg_n).and(&vegsh).par_for_each(
                            |svf_val, &vegsh_val| *svf_val += weight_aniso * vegsh_val,
                        );
                        Zip::from(&mut svf_aniso_veg_n)
                            .and(&vbshvegsh)
                            .par_for_each(|svf_val, &vbsh_val| *svf_val += weight_aniso * vbsh_val);
                    }
                } // end if usevegdem
            } // end loop k (finer altitude steps)

            patch_index += 1;
        } // end loop j (azimuth angles)
    } // end loop i (altitude bands)

    // --- Post-Loop Adjustments ---
    let small_const = 3.0459e-004; // From Python code
    svf_s.par_mapv_inplace(|v| v + small_const);
    svf_w.par_mapv_inplace(|v| v + small_const);

    // Clamp SVF values to max 1.0
    svf.par_mapv_inplace(|v| v.min(1.0));
    svf_e.par_mapv_inplace(|v| v.min(1.0));
    svf_s.par_mapv_inplace(|v| v.min(1.0));
    svf_w.par_mapv_inplace(|v| v.min(1.0));
    svf_n.par_mapv_inplace(|v| v.min(1.0));

    if usevegdem {
        // Python: last[(vegdem2 == 0.0)] = 3.0459e-004
        // This means add small_const where trunk height is zero (after adjustment)
        let mut last = Array2::<f64>::zeros(shape);
        Zip::from(&mut last)
            .and(&veg_trunk_dsm) // Use adjusted trunk height
            .par_for_each(|l, &vt| {
                if vt == 0.0 {
                    *l = small_const;
                }
            });

        Zip::from(&mut svf_veg_s)
            .and(&last)
            .par_for_each(|svf_val, &l| *svf_val += l);
        Zip::from(&mut svf_veg_w)
            .and(&last)
            .par_for_each(|svf_val, &l| *svf_val += l);
        Zip::from(&mut svf_aniso_veg_s)
            .and(&last)
            .par_for_each(|svf_val, &l| *svf_val += l);
        Zip::from(&mut svf_aniso_veg_w)
            .and(&last)
            .par_for_each(|svf_val, &l| *svf_val += l);

        // Clamp vegetation SVF values
        svf_veg.par_mapv_inplace(|v| v.min(1.0));
        svf_veg_e.par_mapv_inplace(|v| v.min(1.0));
        svf_veg_s.par_mapv_inplace(|v| v.min(1.0));
        svf_veg_w.par_mapv_inplace(|v| v.min(1.0));
        svf_veg_n.par_mapv_inplace(|v| v.min(1.0));
        svf_aniso_veg.par_mapv_inplace(|v| v.min(1.0));
        svf_aniso_veg_e.par_mapv_inplace(|v| v.min(1.0));
        svf_aniso_veg_s.par_mapv_inplace(|v| v.min(1.0));
        svf_aniso_veg_w.par_mapv_inplace(|v| v.min(1.0));
        svf_aniso_veg_n.par_mapv_inplace(|v| v.min(1.0));
    }

    // --- Prepare and Return Results ---
    let result = SvfResult {
        svf: svf.into_pyarray(py).to_owned().into(),
        svf_east: svf_e.into_pyarray(py).to_owned().into(),
        svf_south: svf_s.into_pyarray(py).to_owned().into(),
        svf_west: svf_w.into_pyarray(py).to_owned().into(),
        svf_north: svf_n.into_pyarray(py).to_owned().into(),
        svf_veg: svf_veg.into_pyarray(py).to_owned().into(),
        svf_veg_east: svf_veg_e.into_pyarray(py).to_owned().into(),
        svf_veg_south: svf_veg_s.into_pyarray(py).to_owned().into(),
        svf_veg_west: svf_veg_w.into_pyarray(py).to_owned().into(),
        svf_veg_north: svf_veg_n.into_pyarray(py).to_owned().into(),
        svf_aniso_veg: svf_aniso_veg.into_pyarray(py).to_owned().into(),
        svf_aniso_veg_east: svf_aniso_veg_e.into_pyarray(py).to_owned().into(),
        svf_aniso_veg_south: svf_aniso_veg_s.into_pyarray(py).to_owned().into(),
        svf_aniso_veg_west: svf_aniso_veg_w.into_pyarray(py).to_owned().into(),
        svf_aniso_veg_north: svf_aniso_veg_n.into_pyarray(py).to_owned().into(),
        shadow_matrix: shadow_matrix.into_pyarray(py).to_owned().into(),
        veg_shadow_matrix: veg_shadow_matrix.into_pyarray(py).to_owned().into(),
        vbshvegsh_matrix: vbshvegsh_matrix.into_pyarray(py).to_owned().into(),
    };

    result
        .into_pyobject(py)
        .map(|bound| bound.unbind().into()) // Convert Py<SvfResult> to PyObject
        .map_err(|e| e.into()) // Convert PyO3 error
}
