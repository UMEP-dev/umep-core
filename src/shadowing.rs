use ndarray::{s, Array2, ArrayView2, Zip};
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::IntoPyObject;

// Constants for clarity
const SHADOW_FLAG_INTERMEDIATE: f64 = 1.0; // Value indicating shadow during calculation
const SUNLIT_FLAG_INTERMEDIATE: f64 = 0.0; // Value indicating sunlit during calculation
const FINAL_SUNLIT_VALUE: f64 = 1.0; // Final value representing sunlit in output maps
const PERGOLA_SOLID_THRESHOLD: f64 = 4.0; // Threshold for pergola logic (sum of 4 conditions)
const PI_OVER_4: f64 = std::f64::consts::FRAC_PI_4;
const THREE_PI_OVER_4: f64 = 3.0 * PI_OVER_4;
const FIVE_PI_OVER_4: f64 = 5.0 * PI_OVER_4;
const SEVEN_PI_OVER_4: f64 = 7.0 * PI_OVER_4;
const TAU: f64 = std::f64::consts::TAU; // 2 * PI
const EPSILON: f64 = 1e-8; // Small constant for floating-point equality checks

#[pyclass]
/// Result of the shadowing function, containing all output shadow maps.
pub struct ShadowingResult {
    #[pyo3(get)]
    pub veg_shadow_map: Py<PyArray2<f64>>, // Renamed from vegsh
    #[pyo3(get)]
    pub bldg_shadow_map: Py<PyArray2<f64>>, // Renamed from sh
    #[pyo3(get)]
    pub vbshvegsh: Py<PyArray2<f64>>, // Kept as is per request
    #[pyo3(get)]
    pub wallsh: Py<PyArray2<f64>>,
    #[pyo3(get)]
    pub wallsun: Py<PyArray2<f64>>,
    #[pyo3(get)]
    pub wallshve: Py<PyArray2<f64>>,
    #[pyo3(get)]
    pub facesh: Py<PyArray2<f64>>,
    #[pyo3(get)]
    pub facesun: Py<PyArray2<f64>>,
    #[pyo3(get)]
    pub shade_on_wall: Option<Py<PyArray2<f64>>>,
}

// Function to calculate shadows on walls based on azimuth and aspect
fn shade_on_walls(
    azimuth: f64,
    aspect: ArrayView2<f64>,
    walls: ArrayView2<f64>,
    dsm: ArrayView2<f64>,
    propagated_bldg_shadow_height: ArrayView2<f64>, // Renamed from f
    propagated_veg_shadow_height: ArrayView2<f64>,  // Renamed from shvoveg
) -> (
    Array2<f64>,
    Array2<f64>,
    Array2<f64>,
    Array2<f64>,
    Array2<f64>,
) {
    let shape = walls.dim();
    // wallbol: Boolean array indicating wall presence (1.0 if wall > 0, else 0.0)
    let mut wallbol = Array2::<f64>::zeros(shape);
    Zip::from(&mut wallbol)
        .and(&walls)
        .par_for_each(|w, &v| *w = if v > 0.0 { 1.0 } else { 0.0 });

    // Calculate wall face shadow (facesh) based on azimuth and wall aspect
    // Determines if a wall face is oriented away from the sun (in self-shadow)
    let azilow = azimuth - std::f64::consts::FRAC_PI_2; // 90 degrees counter-clockwise from sun
    let azihigh = azimuth + std::f64::consts::FRAC_PI_2; // 90 degrees clockwise from sun
    let mut facesh = Array2::<f64>::zeros(shape);
    // Handle different azimuth ranges to correctly wrap around 0/360 degrees (TAU)
    if azilow >= 0.0 && azihigh < TAU {
        Zip::from(&mut facesh)
            .and(aspect)
            .and(&wallbol)
            .par_for_each(|f, &asp, &wb| {
                // Wall faces shadow if aspect is outside the sun-facing range [azilow, azihigh)
                // Original MATLAB logic: (asp < azilow || asp >= azihigh) - wb + 1
                *f = if asp < azilow || asp >= azihigh {
                    SHADOW_FLAG_INTERMEDIATE // Shadow
                } else {
                    SUNLIT_FLAG_INTERMEDIATE // Sunlit
                } - wb
                    + 1.0;
            });
    } else if azilow < 0.0 && azihigh <= TAU {
        let azilow_wrapped = azilow + TAU; // Wrap azilow
        Zip::from(&mut facesh).and(aspect).par_for_each(|f, &asp| {
            // Wall faces shadow if aspect is within the wrapped range (azilow_wrapped, azihigh]
            // Original MATLAB logic: (asp > azilow_wrapped || asp <= azihigh) * -1 + 1
            *f = if asp > azilow_wrapped || asp <= azihigh {
                -1.0 // Shadow indicator (original MATLAB logic)
            } else {
                0.0
            } + 1.0;
        });
    } else if azilow > 0.0 && azihigh >= TAU {
        let azihigh_wrapped = azihigh - TAU; // Wrap azihigh
        Zip::from(&mut facesh).and(aspect).par_for_each(|f, &asp| {
            // Wall faces shadow if aspect is within the wrapped range (azilow, azihigh_wrapped]
            // Original MATLAB logic: (asp > azilow || asp <= azihigh_wrapped) * -1 + 1
            *f = if asp > azilow || asp <= azihigh_wrapped {
                -1.0 // Shadow indicator (original MATLAB logic)
            } else {
                0.0
            } + 1.0;
        });
    }

    // Calculate building shadow volume height (shvo = propagated_bldg_shadow_height - dsm)
    let mut shvo = Array2::<f64>::zeros(shape);
    Zip::from(&mut shvo)
        .and(&propagated_bldg_shadow_height)
        .and(&dsm)
        .par_for_each(|s, &fv, &dv| *s = fv - dv);

    // Calculate sunlit wall faces (facesun)
    // Wall is sunlit if it exists (walls > 0) and is not in self-shadow (facesh == 0)
    // Original MATLAB logic: ((facesh + (walls > 0)) == 1) & (walls > 0)
    let mut facesun = Array2::<f64>::zeros(shape);
    Zip::from(&mut facesun)
        .and(&facesh)
        .and(walls)
        .par_for_each(|fs, &fh, &w| {
            let wall_exists = w > 0.0;
            let wall_exists_flag = if wall_exists { 1.0 } else { 0.0 };
            *fs = if (fh + wall_exists_flag) == 1.0 && wall_exists {
                1.0 // Sunlit
            } else {
                0.0 // Shadowed or no wall
            };
        });

    // Calculate sunlit wall height (wallsun)
    // Start with total wall height minus shadow volume height
    let mut wallsun = Array2::<f64>::zeros(shape);
    Zip::from(&mut wallsun)
        .and(&walls)
        .and(&shvo)
        .par_for_each(|w, &wa, &shv| *w = wa - shv);
    wallsun.mapv_inplace(|v| if v < 0.0 { 0.0 } else { v }); // Height cannot be negative
                                                             // Remove walls in self-shadow (where facesh == 1)
    Zip::from(&mut wallsun).and(&facesh).par_for_each(|w, &fh| {
        if (fh - 1.0).abs() < EPSILON {
            *w = 0.0
        }
    });

    // Calculate shadowed wall height (wallsh)
    // Total wall height minus sunlit wall height
    let mut wallsh = Array2::<f64>::zeros(shape);
    Zip::from(&mut wallsh)
        .and(&walls)
        .and(&wallsun)
        .par_for_each(|w, &wa, &wsu| *w = wa - wsu);

    // Calculate wall height shadowed by vegetation (wallshve)
    // Start with vegetation shadow volume height on walls
    let mut wallshve = Array2::<f64>::zeros(shape);
    Zip::from(&mut wallshve)
        .and(&propagated_veg_shadow_height)
        .and(&wallbol)
        .par_for_each(|w, &sv, &wb| *w = sv * wb);
    // Subtract building shadow height (already accounted for in wallsh)
    wallshve = &wallshve - &wallsh;
    wallshve.mapv_inplace(|v| if v < 0.0 { 0.0 } else { v }); // Cannot be negative
                                                              // Cap vegetation shadow height at total wall height
    Zip::from(&mut wallshve).and(walls).par_for_each(|wsv, &w| {
        if *wsv > w {
            *wsv = w
        }
    });
    // Adjust sunlit wall height by removing vegetation shadow
    wallsun = &wallsun - &wallshve;
    // Correct potential negative values introduced by subtraction
    Zip::from(&mut wallshve)
        .and(&wallsun)
        .par_for_each(|wsv, &wsu| {
            if wsu < 0.0 {
                *wsv = 0.0 // If wallsun became negative, veg shadow was overestimated
            }
        });
    wallsun.mapv_inplace(|v| if v < 0.0 { 0.0 } else { v }); // Ensure wallsun is not negative

    (wallsh, wallsun, wallshve, facesh, facesun)
}

#[pyfunction]
/// Calculates shadow maps for buildings, vegetation, and walls given DSM and sun position.
///
/// # Arguments
/// * `dsm` - Digital Surface Model (2D array)
/// * `veg_canopy_dsm` - Vegetation canopy DSM (2D array)
/// * `veg_trunk_dsm` - Vegetation trunk DSM (2D array)
/// * `azimuth_deg` - Sun azimuth in degrees
/// * `altitude_deg` - Sun altitude in degrees
/// * `scale` - Pixel scale
/// * `amaxvalue` - Maximum height difference
/// * `bush` - Bush/low vegetation layer
/// * `walls` - Wall height layer
/// * `aspect` - Wall aspect layer
/// * `walls_scheme` - Optional wall scheme
/// * `aspect_scheme` - Optional aspect scheme
///
/// # Returns
/// * `ShadowingResult` struct with all output maps
pub fn shadowingfunction_wallheight_23(
    py: Python,
    dsm: PyReadonlyArray2<f64>,
    veg_canopy_dsm: PyReadonlyArray2<f64>,
    veg_trunk_dsm: PyReadonlyArray2<f64>,
    azimuth_deg: f64,
    altitude_deg: f64,
    scale: f64,
    amaxvalue: f64,
    bush: PyReadonlyArray2<f64>,
    walls: PyReadonlyArray2<f64>,
    aspect: PyReadonlyArray2<f64>,
    walls_scheme: Option<PyReadonlyArray2<f64>>,
    aspect_scheme: Option<PyReadonlyArray2<f64>>,
) -> PyResult<PyObject> {
    // --- Input Shape Validation ---
    let dsm_view = dsm.as_array();
    let veg_canopy_dsm_view = veg_canopy_dsm.as_array();
    let veg_trunk_dsm_view = veg_trunk_dsm.as_array();
    let bush_view = bush.as_array();
    let walls_view = walls.as_array();
    let aspect_view = aspect.as_array();
    let shape = dsm_view.shape();
    let all_shapes = [
        veg_canopy_dsm_view.shape(),
        veg_trunk_dsm_view.shape(),
        bush_view.shape(),
        walls_view.shape(),
        aspect_view.shape(),
    ];
    if all_shapes.iter().any(|&s| s != shape) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "All input arrays must have the same shape.",
        ));
    }
    if let Some(walls_scheme_py) = &walls_scheme {
        if walls_scheme_py.as_array().shape() != shape {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "walls_scheme must have the same shape as dsm.",
            ));
        }
    }
    if let Some(aspect_scheme_py) = &aspect_scheme {
        if aspect_scheme_py.as_array().shape() != shape {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "aspect_scheme must have the same shape as dsm.",
            ));
        }
    }

    // Clamp calculated slice indices to valid bounds
    let clamp =
        |v: f64, min: usize, max: usize| -> usize { v.max(min as f64).min(max as f64) as usize };

    let sizex = dsm_view.shape()[0];
    let sizey = dsm_view.shape()[1];
    // Initialize loop variables for shadow casting steps
    let mut dx: f64 = 0.0; // Shadow step offset in x-direction
    let mut dy: f64 = 0.0; // Shadow step offset in y-direction
    let mut dz: f64 = 0.0; // Shadow step height offset
                           // Temporary arrays used within the loop
    let mut temp_dsm_shifted = Array2::<f64>::zeros((sizex, sizey)); // Renamed from temp
    let mut temp_veg_canopy_shifted = Array2::<f64>::zeros((sizex, sizey)); // Renamed from tempvegdem
    let mut temp_veg_trunk_shifted = Array2::<f64>::zeros((sizex, sizey)); // Renamed from tempvegdem2
    let mut temp_prev_veg_canopy_shifted = Array2::<f64>::zeros((sizex, sizey)); // Renamed from templastfabovea
    let mut temp_prev_veg_trunk_shifted = Array2::<f64>::zeros((sizex, sizey)); // Renamed from templastgabovea
                                                                                // Initialize output/intermediate shadow arrays
    let is_bush_map = bush_view.mapv(|v| if v > 1.0 { 1.0 } else { 0.0 }); // Renamed from bushplant
    let mut bldg_shadow_map = Array2::<f64>::zeros((sizex, sizey)); // Renamed from sh (1=shadow, 0=sun initially)
    let mut vbshvegsh = Array2::<f64>::zeros((sizex, sizey)); // Vegetation blocking building shadow map (Kept name)
    let mut veg_shadow_map = is_bush_map.clone(); // Renamed from vegsh (initialised with bush presence)
    let mut propagated_bldg_shadow_height = dsm_view.to_owned(); // Renamed from f (starts as DSM)
    let mut propagated_veg_shadow_height = veg_canopy_dsm_view.to_owned(); // Renamed from shvoveg (starts as vegdem)
                                                                           // Pre-calculate trigonometric values and constants for the loop
    let sinazimuth = azimuth_deg.to_radians().sin();
    let cosazimuth = azimuth_deg.to_radians().cos();
    let tanazimuth = azimuth_deg.to_radians().tan();
    let signsinazimuth = sinazimuth.signum();
    let signcosazimuth = cosazimuth.signum();
    let dssin = (1.0 / sinazimuth).abs(); // Incremental distance for dy step
    let dscos = (1.0 / cosazimuth).abs(); // Incremental distance for dx step
    let tanaltitudebyscale = altitude_deg.to_radians().tan() / scale; // Tangent of altitude adjusted by scale
    let mut index = 0.0; // Loop counter, represents steps away from the source pixel
    let mut dzprev = 0.0; // Previous dz value (for pergola logic)

    // --- Main Shadow Casting Loop ---
    // Loop continues as long as the shadow height (dz) is less than the max possible height
    // and the shadow offset (dx, dy) is within the image bounds.
    while amaxvalue >= dz && dx.abs() < sizex as f64 && dy.abs() < sizey as f64 {
        // Calculate shadow step offsets (dx, dy) based on azimuth
        // This determines the direction of the shadow step for the current index.
        // Logic handles different azimuth quadrants to ensure correct stepping direction.
        if (PI_OVER_4 <= azimuth_deg.to_radians() && azimuth_deg.to_radians() < THREE_PI_OVER_4) // Azimuth 45 to 135 deg
            || (FIVE_PI_OVER_4 <= azimuth_deg.to_radians() && azimuth_deg.to_radians() < SEVEN_PI_OVER_4)
        // Azimuth 225 to 315 deg
        {
            // Step primarily in y-direction
            dy = signsinazimuth * index;
            dx = -1.0 * signcosazimuth * (index / tanazimuth).round().abs();
        } else {
            // Step primarily in x-direction
            dy = signsinazimuth * (index * tanazimuth).round().abs();
            dx = -1.0 * signcosazimuth * index;
        }
        // Determine incremental distance (ds) based on primary step direction
        let ds = if (PI_OVER_4 <= azimuth_deg.to_radians()
            && azimuth_deg.to_radians() < THREE_PI_OVER_4)
            || (FIVE_PI_OVER_4 <= azimuth_deg.to_radians()
                && azimuth_deg.to_radians() < SEVEN_PI_OVER_4)
        {
            dssin
        } else {
            dscos
        };
        // Calculate vertical shadow offset (dz) for this step
        dz = (ds * index) * tanaltitudebyscale;
        // Reset temporary arrays
        temp_dsm_shifted.fill(0.0);
        temp_veg_canopy_shifted.fill(0.0);
        temp_veg_trunk_shifted.fill(0.0);
        temp_prev_veg_canopy_shifted.fill(0.0);
        temp_prev_veg_trunk_shifted.fill(0.0);
        // Calculate slicing indices for shifting arrays based on dx, dy
        // xc1, yc1, xc2, yc2: Source array slice bounds
        // xp1, yp1, xp2, yp2: Target array slice bounds (current pixel perspective)
        let xc1 = clamp((dx + dx.abs()) / 2.0, 0, sizex);
        let xc2 = clamp(sizex as f64 + (dx - dx.abs()) / 2.0, 0, sizex);
        let yc1 = clamp((dy + dy.abs()) / 2.0, 0, sizey);
        let yc2 = clamp(sizey as f64 + (dy - dy.abs()) / 2.0, 0, sizey);
        let xp1 = clamp(-(dx - dx.abs()) / 2.0, 0, sizex);
        let xp2 = clamp(sizex as f64 - (dx + dx.abs()) / 2.0, 0, sizex);
        let yp1 = clamp(-(dy - dy.abs()) / 2.0, 0, sizey);
        let yp2 = clamp(sizey as f64 - (dy + dy.abs()) / 2.0, 0, sizey);
        // Shift DSM, vegdem, vegdem2 arrays by (-dx, -dy) and subtract dz
        // This simulates looking 'back' along the sun ray from the current pixel.
        if xc2 > xc1 && yc2 > yc1 && xp2 > xp1 && yp2 > yp1 {
            // Ensure slice bounds are valid
            temp_veg_canopy_shifted
                .slice_mut(s![xp1..xp2, yp1..yp2])
                .assign(&(&veg_canopy_dsm_view.slice(s![xc1..xc2, yc1..yc2]) - dz));
            temp_veg_trunk_shifted
                .slice_mut(s![xp1..xp2, yp1..yp2])
                .assign(&(&veg_trunk_dsm_view.slice(s![xc1..xc2, yc1..yc2]) - dz));
            temp_dsm_shifted
                .slice_mut(s![xp1..xp2, yp1..yp2])
                .assign(&(&dsm_view.slice(s![xc1..xc2, yc1..yc2]) - dz));
        }
        // Propagate building shadow height (propagated_bldg_shadow_height)
        // Takes the maximum height seen so far along the sun ray path (current or shifted DSM)
        Zip::from(&mut propagated_bldg_shadow_height)
            .and(&temp_dsm_shifted)
            .par_for_each(|fv, &tv| *fv = fv.max(tv));
        // Propagate vegetation shadow volume height (propagated_veg_shadow_height)
        // Takes the maximum height seen so far (current or shifted veg canopy)
        Zip::from(&mut propagated_veg_shadow_height)
            .and(&temp_veg_canopy_shifted)
            .par_for_each(|sv, &tv| *sv = sv.max(tv));
        // Update building shadow map (bldg_shadow_map)
        // Pixel is in building shadow if propagated height is greater than original DSM
        Zip::from(&mut bldg_shadow_map)
            .and(&propagated_bldg_shadow_height)
            .and(&dsm_view)
            .par_for_each(|s, &fv, &av| {
                *s = if fv > av {
                    SHADOW_FLAG_INTERMEDIATE
                } else {
                    SUNLIT_FLAG_INTERMEDIATE
                }
            });
        // --- Pergola Logic --- (Handles thin vertical vegetation layers)
        // Check if current step's shifted vegetation canopy/trunk is above DSM
        let mut canopy_above_dsm = Array2::<f64>::zeros(temp_veg_canopy_shifted.dim()); // Renamed from fabovea
        Zip::from(&mut canopy_above_dsm)
            .and(&temp_veg_canopy_shifted)
            .and(&dsm_view)
            .par_for_each(|fab, &tv, &av| *fab = if tv > av { 1.0 } else { 0.0 });
        let mut trunk_above_dsm = Array2::<f64>::zeros(temp_veg_trunk_shifted.dim()); // Renamed from gabovea
        Zip::from(&mut trunk_above_dsm)
            .and(&temp_veg_trunk_shifted)
            .and(&dsm_view)
            .par_for_each(|gab, &tv, &av| *gab = if tv > av { 1.0 } else { 0.0 });
        // Shift vegdem/vegdem2 using the *previous* dz value (dzprev)
        if xc2 > xc1 && yc2 > yc1 && xp2 > xp1 && yp2 > yp1 {
            temp_prev_veg_canopy_shifted
                .slice_mut(s![xp1..xp2, yp1..yp2])
                .assign(&(&veg_canopy_dsm_view.slice(s![xc1..xc2, yc1..yc2]) - dzprev));
            temp_prev_veg_trunk_shifted
                .slice_mut(s![xp1..xp2, yp1..yp2])
                .assign(&(&veg_trunk_dsm_view.slice(s![xc1..xc2, yc1..yc2]) - dzprev));
        }
        // Check if previous step's shifted veg layers were above DSM
        let mut prev_canopy_above_dsm = Array2::<f64>::zeros(temp_prev_veg_canopy_shifted.dim()); // Renamed from lastfabovea
        Zip::from(&mut prev_canopy_above_dsm)
            .and(&temp_prev_veg_canopy_shifted)
            .and(&dsm_view)
            .par_for_each(|fab, &tv, &av| *fab = if tv > av { 1.0 } else { 0.0 });
        let mut prev_trunk_above_dsm = Array2::<f64>::zeros(temp_prev_veg_trunk_shifted.dim()); // Renamed from lastgabovea
        Zip::from(&mut prev_trunk_above_dsm)
            .and(&temp_prev_veg_trunk_shifted)
            .and(&dsm_view)
            .par_for_each(|gab, &tv, &av| *gab = if tv > av { 1.0 } else { 0.0 });
        dzprev = dz; // Update dzprev for the next iteration

        // Combine current and previous step checks to detect shadow from thin layers (pergola effect)
        // Sum the four boolean-like maps (0.0 or 1.0)
        let pergola_conditions_met =
            &canopy_above_dsm + &trunk_above_dsm + &prev_canopy_above_dsm + &prev_trunk_above_dsm;
        let pergola_shadow_map = pergola_conditions_met.mapv(|v| {
            // Shadow if any condition met (v > 0), but not all four (v < 4, implies solid object)
            if v > SUNLIT_FLAG_INTERMEDIATE && v < PERGOLA_SOLID_THRESHOLD {
                SHADOW_FLAG_INTERMEDIATE
            } else {
                SUNLIT_FLAG_INTERMEDIATE
            }
        });

        // Update overall vegetation shadow map (veg_shadow_map)
        // Takes the maximum shadow seen so far (existing veg_shadow_map or new pergola shadow)
        Zip::from(&mut veg_shadow_map)
            .and(&pergola_shadow_map)
            .par_for_each(|v, &v2| *v = f64::max(*v, v2));
        // Remove vegetation shadow where building shadow already exists
        Zip::from(&mut veg_shadow_map)
            .and(&bldg_shadow_map)
            .par_for_each(|v, &s| {
                // If both are shadow (1.0 * 1.0 > 0.0), remove veg shadow
                if *v * s > SUNLIT_FLAG_INTERMEDIATE {
                    *v = SUNLIT_FLAG_INTERMEDIATE
                }
            });
        // Accumulate vegetation shadow that blocks potential building shadow
        // (Used later to correct building shadow map - vbshvegsh logic kept as is)
        vbshvegsh = &veg_shadow_map + &vbshvegsh;
        index += 1.0; // Increment loop counter
    }

    // --- Post-Loop Processing ---
    // Finalize building shadow map (bldg_shadow_map): Invert (1=shadow -> 0=shadow, 0=sun -> 1=sun)
    bldg_shadow_map.mapv_inplace(|v| FINAL_SUNLIT_VALUE - v); // 1.0 - 1.0 = 0.0 (shadow), 1.0 - 0.0 = 1.0 (sun)
                                                              // Finalize vegetation-blocking-building-shadow map (vbshvegsh)
    vbshvegsh.mapv_inplace(|v| if v > 0.0 { 1.0 } else { v }); // Threshold to 0 or 1
                                                               // Subtract final vegetation shadow (veg_shadow_map) - Complex logic, kept as is
    vbshvegsh = &vbshvegsh - &veg_shadow_map; // Subtract current veg shadow from accumulated blocking shadow
    vbshvegsh.mapv_inplace(|v| 1.0 - v); // Invert result (Interpretation depends on original intent)

    // Finalize vegetation shadow map (veg_shadow_map): Threshold and Invert (1=shadow -> 0=shadow, 0=sun -> 1=sun)
    veg_shadow_map.mapv_inplace(|v| {
        if v > 0.0 {
            SHADOW_FLAG_INTERMEDIATE
        } else {
            SUNLIT_FLAG_INTERMEDIATE
        }
    });
    veg_shadow_map.mapv_inplace(|v| FINAL_SUNLIT_VALUE - v); // 1.0 - 1.0 = 0.0 (shadow), 1.0 - 0.0 = 1.0 (sun)

    // Calculate final vegetation shadow volume height (propagated_veg_shadow_height)
    // Height difference where vegetation shadow exists (where final veg_shadow_map is 0.0)
    let final_veg_shadow_mask = veg_shadow_map.mapv(|v| FINAL_SUNLIT_VALUE - v); // Invert back to 1=shadow, 0=sun
    propagated_veg_shadow_height =
        (&propagated_veg_shadow_height - &dsm_view) * &final_veg_shadow_mask;

    // Calculate wall shadows using the helper function (first call)
    let (wallsh, wallsun, wallshve, facesh, facesun) = shade_on_walls(
        azimuth_deg.to_radians(),
        aspect_view,
        walls_view,
        dsm_view,
        propagated_bldg_shadow_height.view(),
        propagated_veg_shadow_height.view(),
    );

    // --- Optional Scheme Logic ---
    let mut shade_on_wall_result: Option<Array2<f64>> = None;
    if let (Some(walls_scheme_py), Some(aspect_scheme_py)) = (walls_scheme, aspect_scheme) {
        let walls_scheme_view = walls_scheme_py.as_array();
        let aspect_scheme_view = aspect_scheme_py.as_array();

        // Call shade_on_walls again with scheme arrays
        let (wallsh_, _wallsun_, wallshve_, _facesh_, _facesun_) = shade_on_walls(
            azimuth_deg.to_radians(),
            aspect_scheme_view,                   // Use scheme aspect
            walls_scheme_view,                    // Use scheme walls
            dsm_view,                             // DSM remains the same
            propagated_bldg_shadow_height.view(), // Propagated shadow height remains the same
            propagated_veg_shadow_height.view(),  // Vegetation shadow volume remains the same
        );

        // Combine results: shade_on_wall = np.maximum(wallsh_, wallshve_)
        let mut shade_on_wall_combined = Array2::<f64>::zeros(wallsh_.dim());
        Zip::from(&mut shade_on_wall_combined)
            .and(&wallsh_)
            .and(&wallshve_)
            .par_for_each(|sow, &wsh, &wsv| *sow = f64::max(wsh, wsv)); // Elementwise maximum

        shade_on_wall_result = Some(shade_on_wall_combined);
    }

    // --- Prepare and Return Results ---
    // Create the result struct
    let result = ShadowingResult {
        veg_shadow_map: veg_shadow_map.into_pyarray(py).to_owned().into(), // Renamed field
        bldg_shadow_map: bldg_shadow_map.into_pyarray(py).to_owned().into(), // Renamed field
        vbshvegsh: vbshvegsh.into_pyarray(py).to_owned().into(),           // Kept field name
        wallsh: wallsh.into_pyarray(py).to_owned().into(),
        wallsun: wallsun.into_pyarray(py).to_owned().into(),
        wallshve: wallshve.into_pyarray(py).to_owned().into(),
        facesh: facesh.into_pyarray(py).to_owned().into(),
        facesun: facesun.into_pyarray(py).to_owned().into(),
        // Assign the optional result, converting to Py<PyArray2> if Some
        shade_on_wall: shade_on_wall_result.map(|arr| arr.into_pyarray(py).to_owned().into()),
    };
    // Convert the Rust struct into a Python object (PyObject)
    result
        .into_pyobject(py)
        .map(|bound| bound.unbind().into()) // Convert Py<ShadowingResult> to PyObject (Py<PyAny>)
        .map_err(|e| e.into())
}
