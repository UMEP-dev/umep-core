use ndarray::{Array, Array2, ArrayView2, Zip};
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;
use rayon::prelude::*;
use std::f32::consts::PI;

// Import the correct result struct from shadowing
use crate::shadowing::{calculate_shadows_rust, ShadowingResultRust};

// --- Constants ---
// Sky patch definitions (153 patches)
const SKYVAULTALT_153: [f32; 8] = [6.0, 18.0, 30.0, 42.0, 54.0, 66.0, 78.0, 90.0];
const AZIINTERVAL_153: [f32; 7] = [8.0, 12.0, 16.0, 20.0, 24.0, 28.0, 32.0];
const ANNULINO_153: [i32; 8] = [0, 8, 20, 36, 56, 80, 108, 140];
const SKYVAULTALT_WEIGHTS: [f32; 9] = [0.0, 6.0, 18.0, 30.0, 42.0, 54.0, 66.0, 78.0, 90.0];

// Correction factor applied in finalize step
const LAST_ANNULUS_CORRECTION: f32 = 3.0459e-4;
// Percentile for amaxvalue calculation
const AMAXVALUE_PERCENTILE: f32 = 0.995;

// --- Structs ---

// Structure to hold SVF results for Python
#[pyclass]
pub struct SvfResult {
    #[pyo3(get)]
    pub svf: Py<PyArray2<f32>>,
    #[pyo3(get)]
    pub svf_north: Py<PyArray2<f32>>,
    #[pyo3(get)]
    pub svf_east: Py<PyArray2<f32>>,
    #[pyo3(get)]
    pub svf_south: Py<PyArray2<f32>>,
    #[pyo3(get)]
    pub svf_west: Py<PyArray2<f32>>,
    #[pyo3(get)]
    pub svf_veg: Py<PyArray2<f32>>,
    #[pyo3(get)]
    pub svf_veg_north: Py<PyArray2<f32>>,
    #[pyo3(get)]
    pub svf_veg_east: Py<PyArray2<f32>>,
    #[pyo3(get)]
    pub svf_veg_south: Py<PyArray2<f32>>,
    #[pyo3(get)]
    pub svf_veg_west: Py<PyArray2<f32>>,
    #[pyo3(get)]
    pub svf_aniso_veg: Py<PyArray2<f32>>,
    #[pyo3(get)]
    pub svf_aniso_veg_north: Py<PyArray2<f32>>,
    #[pyo3(get)]
    pub svf_aniso_veg_east: Py<PyArray2<f32>>,
    #[pyo3(get)]
    pub svf_aniso_veg_south: Py<PyArray2<f32>>,
    #[pyo3(get)]
    pub svf_aniso_veg_west: Py<PyArray2<f32>>,
}

// Internal structure for accumulating contributions during parallel processing
#[derive(Clone)]
struct PatchContribution {
    rows: usize,
    cols: usize,
    svf: Array2<f32>,
    svf_n: Array2<f32>,
    svf_e: Array2<f32>,
    svf_s: Array2<f32>,
    svf_w: Array2<f32>,
    svf_veg: Array2<f32>,
    svf_veg_n: Array2<f32>,
    svf_veg_e: Array2<f32>,
    svf_veg_s: Array2<f32>,
    svf_veg_w: Array2<f32>,
    svf_aniso_veg: Array2<f32>,
    svf_aniso_veg_n: Array2<f32>,
    svf_aniso_veg_e: Array2<f32>,
    svf_aniso_veg_s: Array2<f32>,
    svf_aniso_veg_w: Array2<f32>,
}

impl PatchContribution {
    // Create a new contribution object initialized with zeros
    // Always initialize all arrays, regardless of usevegdem
    fn zeros(rows: usize, cols: usize) -> Self {
        let zero_array = || Array2::zeros((rows, cols));
        Self {
            rows,
            cols,
            svf: zero_array(),
            svf_n: zero_array(),
            svf_e: zero_array(),
            svf_s: zero_array(),
            svf_w: zero_array(),
            svf_veg: zero_array(),
            svf_veg_n: zero_array(),
            svf_veg_e: zero_array(),
            svf_veg_s: zero_array(),
            svf_veg_w: zero_array(),
            svf_aniso_veg: zero_array(),
            svf_aniso_veg_n: zero_array(),
            svf_aniso_veg_e: zero_array(),
            svf_aniso_veg_s: zero_array(),
            svf_aniso_veg_w: zero_array(),
        }
    }

    // Combine two contributions (used in reduce step)
    fn combine(mut self, other: Self) -> Self {
        self.svf += &other.svf;
        self.svf_n += &other.svf_n;
        self.svf_e += &other.svf_e;
        self.svf_s += &other.svf_s;
        self.svf_w += &other.svf_w;
        // Always combine veg arrays as they are always initialized
        self.svf_veg += &other.svf_veg;
        self.svf_veg_n += &other.svf_veg_n;
        self.svf_veg_e += &other.svf_veg_e;
        self.svf_veg_s += &other.svf_veg_s;
        self.svf_veg_w += &other.svf_veg_w;
        self.svf_aniso_veg += &other.svf_aniso_veg;
        self.svf_aniso_veg_n += &other.svf_aniso_veg_n;
        self.svf_aniso_veg_e += &other.svf_aniso_veg_e;
        self.svf_aniso_veg_s += &other.svf_aniso_veg_s;
        self.svf_aniso_veg_w += &other.svf_aniso_veg_w;
        self
    }

    // Finalize the results and convert to Python objects
    fn finalize(
        mut self,
        py: Python,
        usevegdem: bool,
        vegdem2: ArrayView2<f32>,
    ) -> PyResult<Py<SvfResult>> {
        // Apply correction factors (matching Python code)
        self.svf_s += LAST_ANNULUS_CORRECTION;
        self.svf_w += LAST_ANNULUS_CORRECTION;

        // Ensure no values exceed 1.0
        self.svf.mapv_inplace(|x| x.min(1.0));
        self.svf_n.mapv_inplace(|x| x.min(1.0));
        self.svf_e.mapv_inplace(|x| x.min(1.0));
        self.svf_s.mapv_inplace(|x| x.min(1.0));
        self.svf_w.mapv_inplace(|x| x.min(1.0));

        // Process vegetation arrays if needed
        if usevegdem {
            // Create correction array for vegetation components
            let last_veg = Array::from_shape_fn((self.rows, self.cols), |(r, c)| {
                if vegdem2[[r, c]] == 0.0 {
                    LAST_ANNULUS_CORRECTION
                } else {
                    0.0
                }
            });

            // Apply corrections
            self.svf_veg_s += &last_veg;
            self.svf_veg_w += &last_veg;
            self.svf_aniso_veg_s += &last_veg;
            self.svf_aniso_veg_w += &last_veg;

            // Ensure no values exceed 1.0
            self.svf_veg.mapv_inplace(|x| x.min(1.0));
            self.svf_veg_n.mapv_inplace(|x| x.min(1.0));
            self.svf_veg_e.mapv_inplace(|x| x.min(1.0));
            self.svf_veg_s.mapv_inplace(|x| x.min(1.0));
            self.svf_veg_w.mapv_inplace(|x| x.min(1.0));
            self.svf_aniso_veg.mapv_inplace(|x| x.min(1.0));
            self.svf_aniso_veg_n.mapv_inplace(|x| x.min(1.0));
            self.svf_aniso_veg_e.mapv_inplace(|x| x.min(1.0));
            self.svf_aniso_veg_s.mapv_inplace(|x| x.min(1.0));
            self.svf_aniso_veg_w.mapv_inplace(|x| x.min(1.0));
        }
        // No need for an 'else' block to create zero arrays, they are already zero

        // Create Python objects - use .unbind() instead of .to_owned()
        Py::new(
            py,
            SvfResult {
                svf: self.svf.into_pyarray(py).unbind(),
                svf_north: self.svf_n.into_pyarray(py).unbind(),
                svf_east: self.svf_e.into_pyarray(py).unbind(),
                svf_south: self.svf_s.into_pyarray(py).unbind(),
                svf_west: self.svf_w.into_pyarray(py).unbind(),
                svf_veg: self.svf_veg.into_pyarray(py).unbind(),
                svf_veg_north: self.svf_veg_n.into_pyarray(py).unbind(),
                svf_veg_east: self.svf_veg_e.into_pyarray(py).unbind(),
                svf_veg_south: self.svf_veg_s.into_pyarray(py).unbind(),
                svf_veg_west: self.svf_veg_w.into_pyarray(py).unbind(),
                svf_aniso_veg: self.svf_aniso_veg.into_pyarray(py).unbind(),
                svf_aniso_veg_north: self.svf_aniso_veg_n.into_pyarray(py).unbind(),
                svf_aniso_veg_east: self.svf_aniso_veg_e.into_pyarray(py).unbind(),
                svf_aniso_veg_south: self.svf_aniso_veg_s.into_pyarray(py).unbind(),
                svf_aniso_veg_west: self.svf_aniso_veg_w.into_pyarray(py).unbind(),
            },
        )
    }
}

// --- Helper Functions ---

/// Calculate the maximum height value used for shadow calculations.
/// Considers DSM percentile and maximum vegetation height if applicable.
fn calculate_amaxvalue(dsm: ArrayView2<f32>, vegdem: ArrayView2<f32>, usevegdem: bool) -> f32 {
    // Find percentile of DSM heights
    let mut dsm_flat = dsm.iter().cloned().collect::<Vec<f32>>();
    dsm_flat.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)); // Handle potential NaNs
    let percentile_idx = (dsm_flat.len() as f32 * AMAXVALUE_PERCENTILE)
        .floor()
        .min(dsm_flat.len() as f32 - 1.0) as usize; // Ensure index is within bounds
    let percentile_val = dsm_flat.get(percentile_idx).cloned().unwrap_or(f32::MAX);

    // Include vegetation height if using vegetation
    let max_val_f32 = if usevegdem {
        let vegmax = vegdem
            .iter()
            .fold(f32::NEG_INFINITY, |max, &val| max.max(val));
        percentile_val.max(vegmax)
    } else {
        percentile_val
    };

    max_val_f32 as f32 // Convert final value to f32
}

/// Prepare vegetation-related input arrays (bush mask, adjusted canopy, adjusted trunk) for shadow calculation.
/// Converts inputs to f32.
fn prepare_vegetation_inputs(
    dsm: ArrayView2<f32>,
    vegdem: ArrayView2<f32>,
    vegdem2: ArrayView2<f32>,
) -> (Array2<f32>, Array2<f32>, Array2<f32>) {
    // Add DSM heights to vegetation heights
    let mut vegdem_adj = vegdem.to_owned();
    let mut vegdem2_adj = vegdem2.to_owned();

    Zip::from(&mut vegdem_adj)
        .and(&mut vegdem2_adj)
        .and(dsm)
        .par_for_each(|v1, v2, &d| {
            *v1 += d;
            if *v1 == d {
                *v1 = 0.0; // Set to 0 if only DSM height (no veg)
            }
            *v2 += d;
            if *v2 == d {
                *v2 = 0.0; // Set to 0 if only DSM height (no veg)
            }
        });

    // Calculate bush areas (vegetation without trunks)
    let bush_areas = Zip::from(&vegdem_adj)
        .and(&vegdem2_adj)
        .par_map_collect(|&v1, &v2| if v1 > 0.0 && v2 == 0.0 { v1 } else { 0.0 });

    (bush_areas, vegdem_adj, vegdem2_adj)
}

// --- Main Calculation Function ---

// Calculate SVF with 153 patches (equivalent to Python's svfForProcessing153)
#[pyfunction]
pub fn calculate_svf_153(
    py: Python,
    dsm_py: PyReadonlyArray2<f32>,
    vegdem_py: PyReadonlyArray2<f32>,
    vegdem2_py: PyReadonlyArray2<f32>,
    scale: f32,
    usevegdem: bool,
) -> PyResult<Py<SvfResult>> {
    // Get array views from Python arrays (still f32 initially)
    let dsm_f32 = dsm_py.as_array();
    let vegdem_f32 = vegdem_py.as_array();
    let vegdem2_f32 = vegdem2_py.as_array(); // Keep f32 version for finalize step

    let rows = dsm_f32.nrows();
    let cols = dsm_f32.ncols();

    // Calculate maximum height for shadow calculations (using f32 arrays is fine here)
    let amaxvalue = calculate_amaxvalue(dsm_f32, vegdem_f32, usevegdem);

    // Convert core inputs to f32 for calculate_shadows_rust
    let dsm_f32 = dsm_f32.mapv(|x| x as f32);
    let scale_f32 = scale as f32;

    // Prepare adjusted vegetation arrays (f32) if needed
    let (bush_f32, vegdem_adj_f32, vegdem2_adj_f32) = if usevegdem {
        let vegdem_f32 = vegdem_f32.mapv(|x| x as f32);
        let vegdem2_f32 = vegdem2_f32.mapv(|x| x as f32);
        let (bush, vegdem_adj, vegdem2_adj) =
            prepare_vegetation_inputs(dsm_f32.view(), vegdem_f32.view(), vegdem2_f32.view());
        (Some(bush), Some(vegdem_adj), Some(vegdem2_adj))
    } else {
        (None, None, None)
    };

    // Create sky patches (153 patches = option 2)
    let (skyvaultalt, aziinterval, _annulino) = create_patches_153(); // annulino not used directly

    // Calculate anisotropic azimuth intervals (ceil(interval/2))
    let aziintervalaniso: Vec<f32> = aziinterval.iter().map(|&azi| (azi / 2.0).ceil()).collect();

    // Calculate azimuth steps and starting points
    let skyvaultaziint: Vec<f32> = aziinterval.iter().map(|&patches| 360.0 / patches).collect();
    let azistart: Vec<f32> = aziinterval
        .iter()
        .map(|&patches| (360.0 / patches) / 2.0)
        .collect();

    // Generate list of all patches for parallel processing
    let mut patches_info = Vec::new();
    // Iterate based on the length of aziinterval (7) instead of skyvaultalt (8)
    for i in 0..aziinterval.len() {
        let altitude = skyvaultalt[i]; // Accessing skyvaultalt[0..6] is safe
        let num_azi_steps = aziinterval[i] as usize;
        let azi_step = skyvaultaziint[i];
        let start_azi = azistart[i];

        for j in 0..num_azi_steps {
            let mut azimuth = (j as f32 * azi_step) + start_azi;
            if azimuth >= 360.0 {
                azimuth -= 360.0;
            }

            patches_info.push((
                altitude,            // f32
                azimuth,             // f32
                i,                   // annulus index (usize)
                aziinterval[i],      // azi interval for weight (f32)
                aziintervalaniso[i], // azi interval for anisotropic weight (f32)
            ));
        }
    }

    // Process all patches in parallel using Rayon
    let result = patches_info
        .par_iter()
        .map(
            |&(altitude_f32, azimuth_f32, annulus_idx, azi_interval, azi_interval_aniso)| {
                // --- Prepare inputs for shadowing ---
                let altitude_f32 = altitude_f32 as f32;
                let azimuth_f32 = azimuth_f32 as f32;

                // Get views for shadowing function call (use f32 versions)
                let dsm_view = dsm_f32.view();
                // Use Option::as_ref().map() to get Option<ArrayView>
                let vegdem_adj_view = vegdem_adj_f32.as_ref().map(|v| v.view());
                let vegdem2_adj_view = vegdem2_adj_f32.as_ref().map(|v| v.view());
                let bush_view = bush_f32.as_ref().map(|b| b.view());

                // Provide default empty views if needed (shadowing expects ArrayView, not Option)
                // Note: This part is slightly awkward. Ideally, shadowing would handle Options.
                let default_empty_f32 = Array2::<f32>::zeros((0, 0));
                let bush_view_for_calc = bush_view.unwrap_or_else(|| default_empty_f32.view());
                let vegdem_adj_view_for_calc =
                    vegdem_adj_view.unwrap_or_else(|| default_empty_f32.view());
                let vegdem2_adj_view_for_calc =
                    vegdem2_adj_view.unwrap_or_else(|| default_empty_f32.view());

                // --- Call shadowing function ---
                let shadow_result: ShadowingResultRust = calculate_shadows_rust(
                    dsm_view,
                    vegdem_adj_view_for_calc, // Use canopy DSM (vegdem adjusted)
                    vegdem2_adj_view_for_calc, // Use trunk DSM (vegdem2 adjusted)
                    azimuth_f32,              // f32
                    altitude_f32,             // f32
                    scale_f32,                // f32
                    amaxvalue,                // f32
                    bush_view_for_calc,       // ArrayView2<f32>
                    None,                     // walls_view_opt
                    None,                     // aspect_view_opt
                    None,                     // walls_scheme_view_opt
                    None,                     // aspect_scheme_view_opt
                );

                // --- Accumulate results ---
                let mut contribution = PatchContribution::zeros(rows, cols);

                // Shadow results are already f32
                let sh_view = shadow_result.bldg_shadow_map.view(); // Directly use the view

                // Calculate weights ONCE for this patch
                let weight = annulus_weight(annulus_idx as i32, azi_interval);
                let weight_aniso = annulus_weight(annulus_idx as i32, azi_interval_aniso);

                // Accumulate building SVF
                contribution.svf.scaled_add(weight, &sh_view);

                // Accumulate directional building SVF (use f32 azimuth)
                if (0.0..180.0).contains(&azimuth_f32) {
                    // East
                    contribution.svf_e.scaled_add(weight_aniso, &sh_view);
                }
                if (90.0..270.0).contains(&azimuth_f32) {
                    // South
                    contribution.svf_s.scaled_add(weight_aniso, &sh_view);
                }
                if (180.0..360.0).contains(&azimuth_f32) {
                    // West
                    contribution.svf_w.scaled_add(weight_aniso, &sh_view);
                }
                if azimuth_f32 >= 270.0 || azimuth_f32 < 90.0 {
                    // North
                    contribution.svf_n.scaled_add(weight_aniso, &sh_view);
                }

                // Accumulate vegetation SVF if enabled
                if usevegdem {
                    // Veg shadow results are already f32
                    let vegsh_view = shadow_result.veg_shadow_map.view(); // Directly use the view
                    let vbshvegsh_view = shadow_result.vbshvegsh.view(); // Directly use the view

                    contribution.svf_veg.scaled_add(weight, &vegsh_view);
                    contribution
                        .svf_aniso_veg
                        .scaled_add(weight, &vbshvegsh_view);

                    // Accumulate directional vegetation SVF (use f32 azimuth)
                    if (0.0..180.0).contains(&azimuth_f32) {
                        // East
                        contribution.svf_veg_e.scaled_add(weight_aniso, &vegsh_view);
                        contribution
                            .svf_aniso_veg_e
                            .scaled_add(weight_aniso, &vbshvegsh_view);
                    }
                    if (90.0..270.0).contains(&azimuth_f32) {
                        // South
                        contribution.svf_veg_s.scaled_add(weight_aniso, &vegsh_view);
                        contribution
                            .svf_aniso_veg_s
                            .scaled_add(weight_aniso, &vbshvegsh_view);
                    }
                    if (180.0..360.0).contains(&azimuth_f32) {
                        // West
                        contribution.svf_veg_w.scaled_add(weight_aniso, &vegsh_view);
                        contribution
                            .svf_aniso_veg_w
                            .scaled_add(weight_aniso, &vbshvegsh_view);
                    }
                    if azimuth_f32 >= 270.0 || azimuth_f32 < 90.0 {
                        // North
                        contribution.svf_veg_n.scaled_add(weight_aniso, &vegsh_view);
                        contribution
                            .svf_aniso_veg_n
                            .scaled_add(weight_aniso, &vbshvegsh_view);
                    }
                }
                // Note: If not usevegdem, veg arrays remain zero

                contribution // Return contribution for this patch
            },
        )
        .reduce(
            || PatchContribution::zeros(rows, cols), // Initial value for reduce
            |a, b| a.combine(b),                     // Combine function
        );

    // Finalize and return results - use the original f32 vegdem2 view
    result.finalize(py, usevegdem, vegdem2_f32)
}

// --- Patch Creation & Weighting ---

// Create 153 sky patches (option 2 in Python)
fn create_patches_153() -> (Vec<f32>, Vec<f32>, Vec<i32>) {
    // Use constants defined at the top
    (
        SKYVAULTALT_153.to_vec(),
        AZIINTERVAL_153.to_vec(),
        ANNULINO_153.to_vec(),
    )
}

// Calculate weight for an annulus
fn annulus_weight(altitude_band_idx: i32, num_azi_intervals: f32) -> f32 {
    // Use constant defined at the top
    let alt_low = (SKYVAULTALT_WEIGHTS[altitude_band_idx as usize] * PI / 180.0).sin();
    let alt_high = (SKYVAULTALT_WEIGHTS[(altitude_band_idx + 1) as usize] * PI / 180.0).sin();

    (alt_high.powi(2) - alt_low.powi(2)) / num_azi_intervals
}
