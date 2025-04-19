use ndarray::{Array, Array2, ArrayView2, Zip};
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;
use rayon::prelude::*;
use std::f32::consts::PI;

// Import the correct result struct from shadowing
use crate::shadowing::{calculate_shadows_rust, ShadowingResultRust};

// Correction factor applied in finalize step
const LAST_ANNULUS_CORRECTION: f32 = 3.0459e-4;
// Percentile for amaxvalue calculation
const AMAXVALUE_PERCENTILE: f32 = 0.995;

// Struct to hold patch configurations

pub struct PatchInfo {
    pub altitude: f32,
    pub azimuth: f32,
    pub azimuth_interval: f32,
    pub azimuth_interval_aniso: f32,
    pub annulino_start: i32,
    pub annulino_end: i32,
}

fn create_patches(option: u8) -> Vec<PatchInfo> {
    let (annulino, altitudes, azi_starts, patches_in_band) = match option {
        1 => (
            vec![0, 12, 24, 36, 48, 60, 72, 84, 90],
            vec![6, 18, 30, 42, 54, 66, 78, 90],
            vec![0, 4, 2, 5, 8, 0, 10, 0],
            vec![30, 30, 24, 24, 18, 12, 6, 1],
        ),
        2 => (
            vec![0, 12, 24, 36, 48, 60, 72, 84, 90],
            vec![6, 18, 30, 42, 54, 66, 78, 90],
            vec![0, 4, 2, 5, 8, 0, 10, 0],
            vec![31, 30, 28, 24, 19, 13, 7, 1],
        ),
        3 => (
            vec![0, 12, 24, 36, 48, 60, 72, 84, 90],
            vec![6, 18, 30, 42, 54, 66, 78, 90],
            vec![0, 4, 2, 5, 8, 0, 10, 0],
            vec![62, 60, 56, 48, 38, 26, 14, 2],
        ),
        4 => (
            vec![0, 4, 9, 15, 21, 27, 33, 39, 45, 51, 57, 63, 69, 75, 81, 90],
            vec![3, 9, 15, 21, 27, 33, 39, 45, 51, 57, 63, 69, 75, 81, 90],
            vec![0, 0, 4, 4, 2, 2, 5, 5, 8, 8, 0, 0, 10, 10, 0],
            vec![62, 62, 60, 60, 56, 56, 48, 48, 38, 38, 26, 26, 14, 14, 2],
        ),
        _ => panic!("Unsupported patch option: {}", option),
    };
    // Iterate over the patch configurations and create PatchInfo instances
    let mut patches: Vec<PatchInfo> = Vec::new();
    for i in 0..altitudes.len() {
        let azimuth_interval = 360.0 / patches_in_band[i] as f32;
        for j in 0..patches_in_band[i] as usize {
            patches.push(PatchInfo {
                altitude: altitudes[i] as f32,
                // Calculate azimuth based on the start and interval
                // Use rem_euclid to ensure azimuth is within [0, 360)
                azimuth: (azi_starts[i] as f32 + j as f32 * azimuth_interval as f32)
                    .rem_euclid(360.0),
                azimuth_interval,
                // Calculate anisotropic azimuth intervals (ceil(interval/2))
                azimuth_interval_aniso: (azimuth_interval / 2.0).ceil(),
                annulino_start: annulino[i],
                annulino_end: annulino[i + 1],
            });
        }
    }
    patches
}

// Calculate weight for an annulus
fn annulus_weight(altitude: i32, aziinterval: f32) -> f32 {
    let n = 90.0;
    let steprad = (360.0 / aziinterval) * (PI / 180.0);
    let annulus = 91.0 - altitude as f32;
    let w = (1.0 / (2.0 * PI))
        * (PI / (2.0 * n)).sin()
        * ((PI * (2.0 * annulus - 1.0)) / (2.0 * n)).sin();
    steprad * w
}

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
    pub svf_vbssh_veg: Py<PyArray2<f32>>,
    #[pyo3(get)]
    pub svf_vbssh_veg_north: Py<PyArray2<f32>>,
    #[pyo3(get)]
    pub svf_vbssh_veg_east: Py<PyArray2<f32>>,
    #[pyo3(get)]
    pub svf_vbssh_veg_south: Py<PyArray2<f32>>,
    #[pyo3(get)]
    pub svf_vbssh_veg_west: Py<PyArray2<f32>>,
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
    svf_vbssh_veg: Array2<f32>,
    svf_vbssh_veg_n: Array2<f32>,
    svf_vbssh_veg_e: Array2<f32>,
    svf_vbssh_veg_s: Array2<f32>,
    svf_vbssh_veg_w: Array2<f32>,
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
            svf_vbssh_veg: zero_array(),
            svf_vbssh_veg_n: zero_array(),
            svf_vbssh_veg_e: zero_array(),
            svf_vbssh_veg_s: zero_array(),
            svf_vbssh_veg_w: zero_array(),
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
        self.svf_vbssh_veg += &other.svf_vbssh_veg;
        self.svf_vbssh_veg_n += &other.svf_vbssh_veg_n;
        self.svf_vbssh_veg_e += &other.svf_vbssh_veg_e;
        self.svf_vbssh_veg_s += &other.svf_vbssh_veg_s;
        self.svf_vbssh_veg_w += &other.svf_vbssh_veg_w;
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
            self.svf_vbssh_veg_s += &last_veg;
            self.svf_vbssh_veg_w += &last_veg;

            // Ensure no values exceed 1.0
            self.svf_veg.mapv_inplace(|x| x.min(1.0));
            self.svf_veg_n.mapv_inplace(|x| x.min(1.0));
            self.svf_veg_e.mapv_inplace(|x| x.min(1.0));
            self.svf_veg_s.mapv_inplace(|x| x.min(1.0));
            self.svf_veg_w.mapv_inplace(|x| x.min(1.0));
            self.svf_vbssh_veg.mapv_inplace(|x| x.min(1.0));
            self.svf_vbssh_veg_n.mapv_inplace(|x| x.min(1.0));
            self.svf_vbssh_veg_e.mapv_inplace(|x| x.min(1.0));
            self.svf_vbssh_veg_s.mapv_inplace(|x| x.min(1.0));
            self.svf_vbssh_veg_w.mapv_inplace(|x| x.min(1.0));
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
                svf_vbssh_veg: self.svf_vbssh_veg.into_pyarray(py).unbind(),
                svf_vbssh_veg_north: self.svf_vbssh_veg_n.into_pyarray(py).unbind(),
                svf_vbssh_veg_east: self.svf_vbssh_veg_e.into_pyarray(py).unbind(),
                svf_vbssh_veg_south: self.svf_vbssh_veg_s.into_pyarray(py).unbind(),
                svf_vbssh_veg_west: self.svf_vbssh_veg_w.into_pyarray(py).unbind(),
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
    patch_option: Option<u8>, // New argument for patch option
) -> PyResult<Py<SvfResult>> {
    let patch_option = patch_option.unwrap_or(2); // Default to 2 if not provided

    // Get array views from Python arrays
    let dsm_f32 = dsm_py.as_array();
    let vegdem_f32 = vegdem_py.as_array();
    let vegdem2_f32 = vegdem2_py.as_array(); // Keep f32 version for finalize step

    let rows = dsm_f32.nrows();
    let cols = dsm_f32.ncols();

    // Calculate maximum height for shadow calculations
    let amaxvalue = calculate_amaxvalue(dsm_f32, vegdem_f32, usevegdem);

    // Prepare adjusted vegetation arrays (f32) if needed
    let (bush_f32, vegdem_adj_f32, vegdem2_adj_f32) = if usevegdem {
        let (bush, vegdem_adj, vegdem2_adj) =
            prepare_vegetation_inputs(dsm_f32.view(), vegdem_f32.view(), vegdem2_f32.view());
        (Some(bush), Some(vegdem_adj), Some(vegdem2_adj))
    } else {
        (None, None, None)
    };

    // Create sky patches (use patch_option argument)
    let patches = create_patches(patch_option);

    // Process all patches in parallel using Rayon
    let result = patches
        .par_iter()
        .map(|patch| {
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
                patch.azimuth,            // f32
                patch.altitude,           // f32
                scale,                    // f32
                amaxvalue,                // f32
                bush_view_for_calc,       // ArrayView2<f32>
                None,                     // walls_view_opt
                None,                     // aspect_view_opt
                None,                     // walls_scheme_view_opt
                None,                     // aspect_scheme_view_opt
            );
            // Get views from shadow_result
            let sh_view = shadow_result.bldg_shadow_map.view();
            let vegsh_view = shadow_result.veg_shadow_map.view();
            let vbshvegsh_view = shadow_result.vbshvegsh.view();

            // --- Accumulate results ---
            let mut contribution = PatchContribution::zeros(rows, cols);

            for altitude in patch.annulino_start..=patch.annulino_end {
                // Accumulate building SVF
                let weight = annulus_weight(altitude, patch.azimuth_interval);
                contribution.svf.scaled_add(weight, &sh_view);
                // Accumulate directional building SVF
                let weight_aniso = annulus_weight(altitude, patch.azimuth_interval_aniso);
                if patch.azimuth >= 0.0 && patch.azimuth < 180.0 {
                    contribution.svf_e.scaled_add(weight_aniso, &sh_view);
                }
                if patch.azimuth >= 90.0 && patch.azimuth < 270.0 {
                    contribution.svf_s.scaled_add(weight_aniso, &sh_view);
                }
                if patch.azimuth >= 180.0 && patch.azimuth < 360.0 {
                    contribution.svf_w.scaled_add(weight_aniso, &sh_view);
                }
                if patch.azimuth >= 270.0 || patch.azimuth < 90.0 {
                    contribution.svf_n.scaled_add(weight_aniso, &sh_view);
                }

                // Accumulate vegetation SVF if enabled
                if usevegdem {
                    contribution.svf_veg.scaled_add(weight, &vegsh_view);
                    contribution
                        .svf_vbssh_veg
                        .scaled_add(weight, &vbshvegsh_view);

                    // Accumulate directional vegetation SVF
                    if patch.azimuth >= 0.0 && patch.azimuth < 180.0 {
                        contribution.svf_veg_e.scaled_add(weight_aniso, &vegsh_view);
                        contribution
                            .svf_vbssh_veg_e
                            .scaled_add(weight_aniso, &vbshvegsh_view);
                    }
                    if patch.azimuth >= 90.0 && patch.azimuth < 270.0 {
                        contribution.svf_veg_s.scaled_add(weight_aniso, &vegsh_view);
                        contribution
                            .svf_vbssh_veg_s
                            .scaled_add(weight_aniso, &vbshvegsh_view);
                    }
                    if patch.azimuth >= 180.0 && patch.azimuth < 360.0 {
                        contribution.svf_veg_w.scaled_add(weight_aniso, &vegsh_view);
                        contribution
                            .svf_vbssh_veg_w
                            .scaled_add(weight_aniso, &vbshvegsh_view);
                    }
                    if patch.azimuth >= 270.0 || patch.azimuth < 90.0 {
                        contribution.svf_veg_n.scaled_add(weight_aniso, &vegsh_view);
                        contribution
                            .svf_vbssh_veg_n
                            .scaled_add(weight_aniso, &vbshvegsh_view);
                    }
                }
            }
            // Note: If not usevegdem, veg arrays remain zero

            contribution // Return contribution for this patch
        })
        .reduce(
            || PatchContribution::zeros(rows, cols), // Initial value for reduce
            |a, b| a.combine(b),                     // Combine function
        );

    // Finalize and return results - use the original f32 vegdem2 view
    result.finalize(py, usevegdem, vegdem2_f32)
}
