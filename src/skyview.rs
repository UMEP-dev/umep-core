use ndarray::{s, Array2, Array3, ArrayView2, Zip};
use numpy::{IntoPyArray, PyArray2, PyArray3, PyReadonlyArray2};
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::shadowing::{calculate_shadows_rust, ShadowingResultRust}; // Import the internal Rust function and the Rust result struct

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
    // Get ArrayView references early
    let dsm_view = dsm_py.as_array();
    let veg_canopy_dsm_in_view = veg_canopy_dsm_py.as_array();
    let veg_trunk_dsm_in_view = veg_trunk_dsm_py.as_array();
    let shape = dsm_view.dim();
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

    // Calculate amaxvalue
    let vegmax = veg_canopy_dsm_in_view
        .iter()
        .fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let dsm_max = dsm_view.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let amaxvalue = dsm_max.max(vegmax);

    // Adjust vegetation DEMs relative to DSM ground height
    let mut veg_canopy_dsm = Array2::<f64>::zeros(shape);
    let mut veg_trunk_dsm = Array2::<f64>::zeros(shape);
    Zip::from(&mut veg_canopy_dsm)
        .and(&mut veg_trunk_dsm)
        .and(&dsm_view) // Use view
        .and(&veg_canopy_dsm_in_view) // Use view
        .and(&veg_trunk_dsm_in_view) // Use view
        .par_for_each(|vc, vt, &d, &vc_in, &vt_in| {
            let vc_abs = vc_in + d;
            let vt_abs = vt_in + d;
            *vc = if vc_abs == d { 0.0 } else { vc_abs };
            *vt = if vt_abs == d { 0.0 } else { vt_abs };
        });

    // Bush separation
    let mut bush = Array2::<f64>::zeros(shape);
    Zip::from(&mut bush)
        .and(&veg_canopy_dsm)
        .and(&veg_trunk_dsm)
        .par_for_each(|b, &vc, &vt| {
            *b = if vc > 0.0 && vt == 0.0 { vc } else { 0.0 };
        });
    let bush_view = bush.view(); // Get view for passing to shadow function

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

    // Precompute patch index offsets for each altitude band for fast lookup
    let mut patch_offsets = Vec::with_capacity(skyvaultaltint.len() + 1);
    patch_offsets.push(0);
    for i in 0..skyvaultaltint.len() {
        patch_offsets.push(patch_offsets[i] + aziinterval[i] as usize);
    }

    // --- Precompute Annulus Weights ---
    let num_alt_bands = skyvaultaltint.len();
    let max_alt_deg: i32 = 90; // Maximum possible altitude degree as i32
    let mut weights = vec![vec![0.0; (max_alt_deg + 1) as usize]; num_alt_bands];
    let mut weights_aniso = vec![vec![0.0; (max_alt_deg + 1) as usize]; num_alt_bands];

    for i in 0..num_alt_bands {
        let patches_in_annulus = aziinterval[i];
        let patches_in_annulus_aniso = (patches_in_annulus / 2.0).ceil();
        let alt_lower_bound = annulino[i];
        let alt_upper_bound = annulino[i + 1];
        for k_alt_deg in (alt_lower_bound as i32 + 1)..=(alt_upper_bound as i32) {
            if k_alt_deg > 0 && k_alt_deg <= max_alt_deg {
                let k_alt_deg_f64 = k_alt_deg as f64;
                let k_idx = k_alt_deg as usize;
                weights[i][k_idx] = calculate_annulus_weight(k_alt_deg_f64, patches_in_annulus);
                weights_aniso[i][k_idx] =
                    calculate_annulus_weight(k_alt_deg_f64, patches_in_annulus_aniso);
            }
        }
    }

    // Shadow matrices (rows, cols, patches)
    let mut shadow_matrix = Array3::<f64>::zeros((rows, cols, total_patches));
    let mut veg_shadow_matrix = Array3::<f64>::zeros((rows, cols, total_patches));
    let mut vbshvegsh_matrix = Array3::<f64>::zeros((rows, cols, total_patches));

    // --- Main Loop (Parallelized) ---
    let patch_results: Vec<_> = (0..skyvaultaltint.len())
        .into_par_iter()
        .flat_map_iter(|i| {
            let altitude_deg_for_shadow = (annulino[i] + annulino[i + 1]) / 2.0;
            let num_azi_intervals = aziinterval[i] as usize;
            let patch_offset = patch_offsets[i];
            let skyvaultaziint = 360.0 / aziinterval[i];
            let azi_start = azistart[i];
            // Precompute weights for this band to avoid repeated indexing in the inner loop
            let weights_band = &weights[i];
            let weights_aniso_band = &weights_aniso[i];
            let alt_lower_bound = annulino[i];
            let alt_upper_bound = annulino[i + 1];
            // Precompute weights only once per band, store in tuple for all azimuths in this band
            let (total_weight, total_weight_aniso) = {
                let mut tw = 0.0;
                let mut twa = 0.0;
                // This loop is efficient as it runs only once per altitude band (outer parallel task).
                for k_alt_deg in (alt_lower_bound as i32 + 1)..=(alt_upper_bound as i32) {
                    if k_alt_deg > 0 && k_alt_deg <= 90 {
                        let k_idx = k_alt_deg as usize;
                        tw += weights_band[k_idx];
                        twa += weights_aniso_band[k_idx];
                    }
                }
                (tw, twa)
            };
            // This sequential map generates the parameters for each azimuth within the current altitude band.
            // It runs sequentially within each parallel task of the outer loop.
            (0..num_azi_intervals).map(move |j| {
                let patch_index = patch_offset + j;
                let mut azimuth_deg = (j as f64) * skyvaultaziint + azi_start;
                if azimuth_deg >= 360.0 {
                    azimuth_deg -= 360.0;
                }
                let azi_rad = azimuth_deg.to_radians();

                (
                    i,
                    patch_index,
                    azimuth_deg,
                    azi_rad,
                    altitude_deg_for_shadow,
                    total_weight,       // Pass precomputed weights
                    total_weight_aniso, // Pass precomputed weights
                )
            })
        })
        // The .map() below processes each generated patch tuple in parallel.
        // The main workload (calculate_shadows_rust) happens here.
        .map(
            |(
                _i, // i is no longer needed here as weights are precomputed and passed
                patch_index,
                azimuth_deg,
                azi_rad,
                altitude_deg_for_shadow,
                total_weight,
                total_weight_aniso,
            )| {
                let shadow_result: ShadowingResultRust = calculate_shadows_rust(
                    dsm_view,
                    veg_canopy_dsm.view(),
                    veg_trunk_dsm.view(),
                    azimuth_deg,
                    altitude_deg_for_shadow,
                    scale,
                    amaxvalue,
                    bush_view,
                    None,
                    None,
                    None,
                    None,
                );
                let sh = shadow_result.bldg_shadow_map;
                let vegsh = shadow_result.veg_shadow_map;
                let vbshvegsh = shadow_result.vbshvegsh;

                // Use precomputed total_weight and total_weight_aniso
                let mut svf_patch = sh.mapv(|v| total_weight * v);
                let mut svf_e_patch = Array2::<f64>::zeros(shape);
                let mut svf_s_patch = Array2::<f64>::zeros(shape);
                let mut svf_w_patch = Array2::<f64>::zeros(shape);
                let mut svf_n_patch = Array2::<f64>::zeros(shape);

                if azi_rad >= 0.0 && azi_rad < std::f64::consts::PI {
                    svf_e_patch = sh.mapv(|v| total_weight_aniso * v);
                }
                if azi_rad >= std::f64::consts::FRAC_PI_2
                    && azi_rad < 3.0 * std::f64::consts::FRAC_PI_2
                {
                    svf_s_patch = sh.mapv(|v| total_weight_aniso * v);
                }
                if azi_rad >= std::f64::consts::PI && azi_rad < std::f64::consts::TAU {
                    svf_w_patch = sh.mapv(|v| total_weight_aniso * v);
                }
                if azi_rad >= 3.0 * std::f64::consts::FRAC_PI_2
                    || azi_rad < std::f64::consts::FRAC_PI_2
                {
                    svf_n_patch = sh.mapv(|v| total_weight_aniso * v);
                }

                let mut svf_veg_patch = Array2::<f64>::zeros(shape);
                let mut svf_aniso_veg_patch = Array2::<f64>::zeros(shape);
                let mut svf_veg_e_patch = Array2::<f64>::zeros(shape);
                let mut svf_veg_s_patch = Array2::<f64>::zeros(shape);
                let mut svf_veg_w_patch = Array2::<f64>::zeros(shape);
                let mut svf_veg_n_patch = Array2::<f64>::zeros(shape);
                let mut svf_aniso_veg_e_patch = Array2::<f64>::zeros(shape);
                let mut svf_aniso_veg_s_patch = Array2::<f64>::zeros(shape);
                let mut svf_aniso_veg_w_patch = Array2::<f64>::zeros(shape);
                let mut svf_aniso_veg_n_patch = Array2::<f64>::zeros(shape);

                if usevegdem {
                    svf_veg_patch = vegsh.mapv(|v| total_weight * v);
                    svf_aniso_veg_patch = vbshvegsh.mapv(|v| total_weight * v);

                    if azi_rad >= 0.0 && azi_rad < std::f64::consts::PI {
                        svf_veg_e_patch = vegsh.mapv(|v| total_weight_aniso * v);
                        svf_aniso_veg_e_patch = vbshvegsh.mapv(|v| total_weight_aniso * v);
                    }
                    if azi_rad >= std::f64::consts::FRAC_PI_2
                        && azi_rad < 3.0 * std::f64::consts::FRAC_PI_2
                    {
                        svf_veg_s_patch = vegsh.mapv(|v| total_weight_aniso * v);
                        svf_aniso_veg_s_patch = vbshvegsh.mapv(|v| total_weight_aniso * v);
                    }
                    if azi_rad >= std::f64::consts::PI && azi_rad < std::f64::consts::TAU {
                        svf_veg_w_patch = vegsh.mapv(|v| total_weight_aniso * v);
                        svf_aniso_veg_w_patch = vbshvegsh.mapv(|v| total_weight_aniso * v);
                    }
                    if azi_rad >= 3.0 * std::f64::consts::FRAC_PI_2
                        || azi_rad < std::f64::consts::FRAC_PI_2
                    {
                        svf_veg_n_patch = vegsh.mapv(|v| total_weight_aniso * v);
                        svf_aniso_veg_n_patch = vbshvegsh.mapv(|v| total_weight_aniso * v);
                    }
                }

                (
                    patch_index,
                    sh,
                    vegsh,
                    vbshvegsh,
                    svf_patch,
                    svf_e_patch,
                    svf_s_patch,
                    svf_w_patch,
                    svf_n_patch,
                    svf_veg_patch,
                    svf_veg_e_patch,
                    svf_veg_s_patch,
                    svf_veg_w_patch,
                    svf_veg_n_patch,
                    svf_aniso_veg_patch,
                    svf_aniso_veg_e_patch,
                    svf_aniso_veg_s_patch,
                    svf_aniso_veg_w_patch,
                    svf_aniso_veg_n_patch,
                )
            },
        )
        .collect();

    // --- Aggregate results ---
    for (
        patch_index,
        sh,
        vegsh,
        vbshvegsh,
        svf_patch,
        svf_e_patch,
        svf_s_patch,
        svf_w_patch,
        svf_n_patch,
        svf_veg_patch,
        svf_veg_e_patch,
        svf_veg_s_patch,
        svf_veg_w_patch,
        svf_veg_n_patch,
        svf_aniso_veg_patch,
        svf_aniso_veg_e_patch,
        svf_aniso_veg_s_patch,
        svf_aniso_veg_w_patch,
        svf_aniso_veg_n_patch,
    ) in patch_results
    {
        svf += &svf_patch;
        svf_e += &svf_e_patch;
        svf_s += &svf_s_patch;
        svf_w += &svf_w_patch;
        svf_n += &svf_n_patch;
        shadow_matrix.slice_mut(s![.., .., patch_index]).assign(&sh);
        if usevegdem {
            svf_veg += &svf_veg_patch;
            svf_veg_e += &svf_veg_e_patch;
            svf_veg_s += &svf_veg_s_patch;
            svf_veg_w += &svf_veg_w_patch;
            svf_veg_n += &svf_veg_n_patch;
            svf_aniso_veg += &svf_aniso_veg_patch;
            svf_aniso_veg_e += &svf_aniso_veg_e_patch;
            svf_aniso_veg_s += &svf_aniso_veg_s_patch;
            svf_aniso_veg_w += &svf_aniso_veg_w_patch;
            svf_aniso_veg_n += &svf_aniso_veg_n_patch;
            veg_shadow_matrix
                .slice_mut(s![.., .., patch_index])
                .assign(&vegsh);
            vbshvegsh_matrix
                .slice_mut(s![.., .., patch_index])
                .assign(&vbshvegsh);
        }
    }

    let small_const = 3.0459e-004;
    svf_s.par_mapv_inplace(|v| v + small_const);
    svf_w.par_mapv_inplace(|v| v + small_const);
    svf.par_mapv_inplace(|v| v.min(1.0));
    svf_e.par_mapv_inplace(|v| v.min(1.0));
    svf_s.par_mapv_inplace(|v| v.min(1.0));
    svf_w.par_mapv_inplace(|v| v.min(1.0));
    svf_n.par_mapv_inplace(|v| v.min(1.0));
    if usevegdem {
        let mut last = Array2::<f64>::zeros(shape);
        Zip::from(&mut last)
            .and(&veg_trunk_dsm)
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
        .map(|bound| bound.unbind().into())
        .map_err(|e| e.into())
}
