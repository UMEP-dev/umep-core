use ndarray::{Array, Array2, ArrayView2, Zip};
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;
use rayon::prelude::*;
use std::f32::consts::PI;

// Import the correct result struct from shadowing
use crate::shadowing::{calculate_shadows_rust, ShadowingResultRust};

// Correction factor applied in finalize step
const LAST_ANNULUS_CORRECTION: f32 = 3.0459e-4;

// Struct to hold patch configurations

pub struct PatchInfo {
    pub altitude: f32,
    pub azimuth: f32,
    pub azimuth_patches: f32,
    pub azimuth_patches_aniso: f32,
    pub annulino_start: i32,
    pub annulino_end: i32,
}

fn create_patches(option: u8) -> Vec<PatchInfo> {
    let (annulino, altitudes, azi_starts, azimuth_patches) = match option {
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
        let azimuth_interval = 360.0 / azimuth_patches[i] as f32;
        for j in 0..azimuth_patches[i] as usize {
            // Calculate azimuth based on the start and interval
            // Use rem_euclid to ensure azimuth is within [0, 360)
            let azimuth =
                (azi_starts[i] as f32 + j as f32 * azimuth_interval as f32).rem_euclid(360.0);
            patches.push(PatchInfo {
                altitude: altitudes[i] as f32,
                azimuth,
                azimuth_patches: azimuth_patches[i] as f32,
                // Calculate anisotropic azimuth patches (ceil(interval/2))
                azimuth_patches_aniso: (azimuth_patches[i] as f32 / 2.0).ceil(),
                annulino_start: annulino[i],
                annulino_end: annulino[i + 1],
            });
        }
    }
    patches
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
fn calculate_amaxvalue(dsm: ArrayView2<f32>, vegdem: ArrayView2<f32>, usevegdem: bool) -> f32 {
    let dsm_max = dsm.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    if usevegdem {
        let vegmax = vegdem.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        dsm_max.max(vegmax)
    } else {
        dsm_max
    }
}

fn prepare_bushes(vegdem: ArrayView2<f32>, vegdem2: ArrayView2<f32>) -> (Array2<f32>) {
    let vegdem_adj = vegdem.to_owned();
    let vegdem2_adj = vegdem2.to_owned();
    // Calculate bush areas (vegetation without trunks)
    let bush_areas = Zip::from(&vegdem_adj)
        .and(&vegdem2_adj)
        .par_map_collect(|&v1, &v2| if v2 > 0.0 { 0.0 } else { v1 });
    bush_areas
}

// --- Main Calculation Function ---
// Calculate SVF with 153 patches (equivalent to Python's svfForProcessing153)
#[pyfunction]
pub fn calculate_svf(
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

    // Prepare bushes
    let bush_f32 = prepare_bushes(vegdem_f32.view(), vegdem2_f32.view());

    // Create sky patches (use patch_option argument)
    let patches = create_patches(patch_option);

    // Process all patches in parallel using Rayon
    let result = patches
        .par_iter()
        .map(|patch| {
            // Get views for shadowing function call (use f32 versions)
            let dsm_view = dsm_f32.view();
            // Use Option::as_ref().map() to get Option<ArrayView>
            let vegdem_adj_view = vegdem_f32.view();
            let vegdem2_adj_view = vegdem2_f32.view();
            let bush_view = bush_f32.view();

            // --- Call shadowing function ---
            let shadow_result: ShadowingResultRust = calculate_shadows_rust(
                dsm_view,
                vegdem_adj_view,  // Use canopy DSM (vegdem adjusted)
                vegdem2_adj_view, // Use trunk DSM (vegdem2 adjusted)
                patch.azimuth,    // f32
                patch.altitude,   // f32
                scale,            // f32
                amaxvalue,        // f32
                bush_view,        // ArrayView2<f32>
                None,             // walls_view_opt
                None,             // aspect_view_opt
                None,             // walls_scheme_view_opt
                None,             // aspect_scheme_view_opt
            );

            // --- Accumulate results ---
            let mut contribution = PatchContribution::zeros(rows, cols);

            // Shadow results are already f32
            let sh_view = shadow_result.bldg_shadow_map.view(); // Directly use the view

            // --- Pre-calculate altitude-dependent part of weights ---
            let n = 90.0; // Constant from annulus_weight
            let common_w_factor = (1.0 / (2.0 * PI)) * (PI / (2.0 * n)).sin(); // Constant part of w

            // Pre-calculate step radians for this patch
            let steprad_iso = (360.0 / patch.azimuth_patches) * (PI / 180.0);
            let steprad_aniso = (360.0 / patch.azimuth_patches_aniso) * (PI / 180.0);

            // --- Loop through altitudes for this patch ---
            for altitude in patch.annulino_start..=patch.annulino_end {
                // Calculate altitude-specific part of weight calculation
                let annulus = 91.0 - altitude as f32;
                let sin_term = ((PI * (2.0 * annulus - 1.0)) / (2.0 * n)).sin();
                let common_w_part = common_w_factor * sin_term; // Full w without steprad

                // Calculate final weights for this altitude
                let weight_iso = steprad_iso * common_w_part; // Isotropic weight
                let weight_aniso = steprad_aniso * common_w_part; // Anisotropic weight

                // Accumulate building SVF (Isotropic)
                contribution.svf.scaled_add(weight_iso, &sh_view);

                // Accumulate directional building SVF (Anisotropic)
                if patch.azimuth >= 0.0 && patch.azimuth < 180.0 {
                    // East
                    contribution.svf_e.scaled_add(weight_aniso, &sh_view);
                }
                if patch.azimuth >= 90.0 && patch.azimuth < 270.0 {
                    // South
                    contribution.svf_s.scaled_add(weight_aniso, &sh_view);
                }
                if patch.azimuth >= 180.0 && patch.azimuth < 360.0 {
                    // West
                    contribution.svf_w.scaled_add(weight_aniso, &sh_view);
                }
                if patch.azimuth >= 270.0 || patch.azimuth < 90.0 {
                    // North
                    contribution.svf_n.scaled_add(weight_aniso, &sh_view);
                }

                // Accumulate vegetation SVF if enabled
                if usevegdem {
                    // Veg shadow results are already f32
                    let vegsh_view = shadow_result.veg_shadow_map.view();
                    let vbshvegsh_view = shadow_result.vbshvegsh.view();

                    // Accumulate isotropic vegetation SVF (Matches Python)
                    contribution.svf_veg.scaled_add(weight_iso, &vegsh_view);
                    // Accumulate anisotropic vegetation SVF (Matches Python svfaveg)
                    contribution
                        .svf_vbssh_veg
                        .scaled_add(weight_iso, &vbshvegsh_view);

                    // Accumulate directional vegetation SVF (Anisotropic - Matches Python)
                    if patch.azimuth >= 0.0 && patch.azimuth < 180.0 {
                        // East
                        contribution.svf_veg_e.scaled_add(weight_aniso, &vegsh_view);
                        contribution
                            .svf_vbssh_veg_e
                            .scaled_add(weight_aniso, &vbshvegsh_view);
                    }
                    if patch.azimuth >= 90.0 && patch.azimuth < 270.0 {
                        // South
                        contribution.svf_veg_s.scaled_add(weight_aniso, &vegsh_view);
                        contribution
                            .svf_vbssh_veg_s
                            .scaled_add(weight_aniso, &vbshvegsh_view);
                    }
                    if patch.azimuth >= 180.0 && patch.azimuth < 360.0 {
                        // West
                        contribution.svf_veg_w.scaled_add(weight_aniso, &vegsh_view);
                        contribution
                            .svf_vbssh_veg_w
                            .scaled_add(weight_aniso, &vbshvegsh_view);
                    }
                    if patch.azimuth >= 270.0 || patch.azimuth < 90.0 {
                        // North
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
