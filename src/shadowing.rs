use ndarray::{s, Array2, ArrayView2, Zip};
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::IntoPyObject;

// Constants
const SHADOW_FLAG_INTERMEDIATE: f64 = 1.0; // Intermediate value indicating shadow
const SUNLIT_FLAG_INTERMEDIATE: f64 = 0.0; // Intermediate value indicating sunlit
const FINAL_SUNLIT_VALUE: f64 = 1.0; // Final output value representing sunlit
const PERGOLA_SOLID_THRESHOLD: f64 = 4.0; // Threshold for pergola shadow logic
const PI_OVER_4: f64 = std::f64::consts::FRAC_PI_4;
const THREE_PI_OVER_4: f64 = 3.0 * PI_OVER_4;
const FIVE_PI_OVER_4: f64 = 5.0 * PI_OVER_4;
const SEVEN_PI_OVER_4: f64 = 7.0 * PI_OVER_4;
const TAU: f64 = std::f64::consts::TAU; // 2 * PI
const EPSILON: f64 = 1e-8; // Small value for float comparisons

#[pyclass]
/// Result of the shadowing function, containing all output shadow maps.
pub struct ShadowingResult {
    #[pyo3(get)]
    pub veg_shadow_map: Py<PyArray2<f64>>,
    #[pyo3(get)]
    pub bldg_shadow_map: Py<PyArray2<f64>>,
    #[pyo3(get)]
    /// Vegetation Blocking Building Shadow: Indicates where vegetation prevents building shadow.
    pub vbshvegsh: Py<PyArray2<f64>>,
    #[pyo3(get)]
    pub wallsh: Py<PyArray2<f64>>, // Shadowed wall height (by buildings)
    #[pyo3(get)]
    pub wallsun: Py<PyArray2<f64>>, // Sunlit wall height
    #[pyo3(get)]
    pub wallshve: Py<PyArray2<f64>>, // Wall height shadowed by vegetation
    #[pyo3(get)]
    pub facesh: Py<PyArray2<f64>>, // Wall face shadow mask (1 if face away from sun)
    #[pyo3(get)]
    pub facesun: Py<PyArray2<f64>>, // Sunlit wall face mask (1 if face towards sun and not obstructed)
    #[pyo3(get)]
    /// Combined building and vegetation shadow height on walls (optional scheme).
    pub shade_on_wall: Option<Py<PyArray2<f64>>>,
}

/// Calculates shadow heights and masks on walls based on sun position, geometry, and propagated shadows.
///
/// This function determines:
/// - Which wall faces are directly illuminated by the sun (considering only aspect and azimuth).
/// - The height of shadows cast by buildings onto walls.
/// - The height of shadows cast by vegetation onto walls.
/// - The resulting sunlit and shadowed portions of walls.
fn shade_on_walls(
    azimuth: f64,                                   // Sun azimuth in radians
    aspect: ArrayView2<f64>,                        // Wall aspect/orientation
    walls: ArrayView2<f64>,                         // Wall height
    dsm: ArrayView2<f64>,                           // Digital Surface Model
    propagated_bldg_shadow_height: ArrayView2<f64>, // Max shadow height from buildings seen at each pixel
    propagated_veg_shadow_height: ArrayView2<f64>, // Max shadow height from vegetation seen at each pixel
) -> (
    Array2<f64>, // shadowed_wall_height (wallsh)
    Array2<f64>, // sunlit_wall_height (wallsun)
    Array2<f64>, // veg_shadow_on_wall_height (wallshve)
    Array2<f64>, // wall_face_shadow_mask (facesh)
    Array2<f64>, // sunlit_wall_face_mask (facesun)
) {
    let shape = walls.dim();
    // Create a mask indicating wall presence (1.0 where wall height > 0).
    let mut wall_mask = Array2::<f64>::zeros(shape);
    Zip::from(&mut wall_mask)
        .and(&walls)
        .par_for_each(|mask_val, &wall_h| *mask_val = if wall_h > 0.0 { 1.0 } else { 0.0 });

    // Calculate wall face shadow mask (1 if face is oriented away from the sun).
    let azilow = azimuth - std::f64::consts::FRAC_PI_2; // 90 degrees counter-clockwise from sun
    let azihigh = azimuth + std::f64::consts::FRAC_PI_2; // 90 degrees clockwise from sun
    let mut wall_face_shadow_mask = Array2::<f64>::zeros(shape);
    // Handle azimuth wrapping around 0/TAU degrees.
    if azilow >= 0.0 && azihigh < TAU {
        Zip::from(&mut wall_face_shadow_mask)
            .and(aspect)
            .and(&wall_mask)
            .par_for_each(|f_sh, &asp, &w_mask| {
                // Shadow if aspect is outside the sun-facing range [azilow, azihigh).
                // Original MATLAB: (asp < azilow || asp >= azihigh) - wb + 1
                *f_sh = if asp < azilow || asp >= azihigh {
                    SHADOW_FLAG_INTERMEDIATE // Shadow
                } else {
                    SUNLIT_FLAG_INTERMEDIATE // Sunlit
                } - w_mask
                    + 1.0;
            });
    } else if azilow < 0.0 && azihigh <= TAU {
        let azilow_wrapped = azilow + TAU;
        Zip::from(&mut wall_face_shadow_mask)
            .and(aspect)
            .par_for_each(|f_sh, &asp| {
                // Shadow if aspect is within the wrapped range (azilow_wrapped, azihigh].
                // Original MATLAB: (asp > azilow_wrapped || asp <= azihigh) * -1 + 1
                *f_sh = if asp > azilow_wrapped || asp <= azihigh {
                    -1.0 // Shadow indicator (original MATLAB logic)
                } else {
                    0.0
                } + 1.0;
            });
    } else {
        // azilow > 0.0 && azihigh >= TAU
        let azihigh_wrapped = azihigh - TAU;
        Zip::from(&mut wall_face_shadow_mask)
            .and(aspect)
            .par_for_each(|f_sh, &asp| {
                // Shadow if aspect is within the wrapped range (azilow, azihigh_wrapped].
                // Original MATLAB: (asp > azilow || asp <= azihigh_wrapped) * -1 + 1
                *f_sh = if asp > azilow || asp <= azihigh_wrapped {
                    -1.0 // Shadow indicator (original MATLAB logic)
                } else {
                    0.0
                } + 1.0;
            });
    }

    // Calculate building shadow volume height relative to the ground (DSM).
    let mut building_shadow_volume_height = Array2::<f64>::zeros(shape);
    Zip::from(&mut building_shadow_volume_height)
        .and(&propagated_bldg_shadow_height)
        .and(&dsm)
        .par_for_each(|sh_vol, &prop_h, &dsm_h| *sh_vol = prop_h - dsm_h);

    // Calculate sunlit wall face mask (1 if wall exists and face is not in self-shadow).
    // Original MATLAB: ((facesh + (walls > 0)) == 1) & (walls > 0)
    let mut sunlit_wall_face_mask = Array2::<f64>::zeros(shape);
    Zip::from(&mut sunlit_wall_face_mask)
        .and(&wall_face_shadow_mask)
        .and(walls)
        .par_for_each(|sf_mask, &f_sh_mask, &w_h| {
            let wall_exists = w_h > 0.0;
            let wall_exists_flag = if wall_exists { 1.0 } else { 0.0 };
            *sf_mask = if (f_sh_mask + wall_exists_flag) == 1.0 && wall_exists {
                1.0 // Sunlit face
            } else {
                0.0 // Shadowed face or no wall
            };
        });

    // Calculate sunlit wall height: Start with total wall height minus building shadow volume height.
    let mut sunlit_wall_height = Array2::<f64>::zeros(shape);
    Zip::from(&mut sunlit_wall_height)
        .and(&walls)
        .and(&building_shadow_volume_height)
        .par_for_each(|sun_h, &wall_h, &sh_vol_h| *sun_h = wall_h - sh_vol_h);
    sunlit_wall_height.par_mapv_inplace(|v| v.max(0.0)); // Height cannot be negative.
                                                         // Remove walls in self-shadow (where face is oriented away from the sun).
    Zip::from(&mut sunlit_wall_height)
        .and(&wall_face_shadow_mask)
        .par_for_each(|sun_h, &f_sh_mask| {
            if (f_sh_mask - 1.0).abs() < EPSILON {
                *sun_h = 0.0
            }
        });

    // Calculate shadowed wall height (due to buildings): Total wall height minus sunlit height.
    let mut shadowed_wall_height = Array2::<f64>::zeros(shape);
    Zip::from(&mut shadowed_wall_height)
        .and(&walls)
        .and(&sunlit_wall_height)
        .par_for_each(|sh_h, &wall_h, &sun_h| *sh_h = wall_h - sun_h);

    // Calculate wall height shadowed *only* by vegetation.
    // Start with propagated vegetation shadow height, masked by wall presence.
    let mut veg_shadow_on_wall_height = Array2::<f64>::zeros(shape);
    Zip::from(&mut veg_shadow_on_wall_height)
        .and(&propagated_veg_shadow_height)
        .and(&wall_mask)
        .par_for_each(|veg_sh_h, &prop_veg_h, &w_mask| *veg_sh_h = prop_veg_h * w_mask);
    // Subtract building shadow height (already accounted for in shadowed_wall_height).
    Zip::from(&mut veg_shadow_on_wall_height)
        .and(&shadowed_wall_height)
        .par_for_each(|veg_sh_h, &bldg_sh_h| *veg_sh_h -= bldg_sh_h);
    veg_shadow_on_wall_height.par_mapv_inplace(|v| v.max(0.0)); // Cannot be negative.
                                                                // Cap vegetation shadow height at the total wall height.
    Zip::from(&mut veg_shadow_on_wall_height)
        .and(walls)
        .par_for_each(|veg_sh_h, &wall_h| {
            if *veg_sh_h > wall_h {
                *veg_sh_h = wall_h
            }
        });
    // Adjust sunlit wall height by removing the calculated vegetation shadow.
    Zip::from(&mut sunlit_wall_height)
        .and(&veg_shadow_on_wall_height)
        .par_for_each(|sun_h, &veg_sh_h| *sun_h -= veg_sh_h);
    // Correct potential negative values introduced by subtraction.
    Zip::from(&mut veg_shadow_on_wall_height)
        .and(&sunlit_wall_height)
        .par_for_each(|veg_sh_h, &sun_h| {
            if sun_h < 0.0 {
                // If sunlit height became negative, veg shadow was overestimated or overlapped fully with building shadow.
                *veg_sh_h = 0.0
            }
        });
    sunlit_wall_height.par_mapv_inplace(|v| v.max(0.0)); // Ensure sunlit height is not negative.

    (
        shadowed_wall_height,
        sunlit_wall_height,
        veg_shadow_on_wall_height,
        wall_face_shadow_mask,
        sunlit_wall_face_mask,
    )
}

#[pyfunction]
/// Calculates shadow maps for buildings, vegetation, and walls given DSM and sun position.
///
/// This function implements a shadow casting algorithm. It iterates outwards from each
/// pixel in the direction opposite to the sun's azimuth, calculating the height of
/// potential shadow casters (buildings, vegetation) at increasing distances. It determines
/// ground and wall shadows based on whether the propagated shadow height exceeds the
/// surface/wall height at that pixel. Special logic handles thin vegetation (pergolas).
///
/// # Arguments
/// * `dsm` - Digital Surface Model (buildings, ground)
/// * `veg_canopy_dsm` - Vegetation canopy height DSM
/// * `veg_trunk_dsm` - Vegetation trunk height DSM (defines bottom of canopy)
/// * `azimuth_deg` - Sun azimuth in degrees (0=N, 90=E, 180=S, 270=W)
/// * `altitude_deg` - Sun altitude/elevation in degrees (0=horizon, 90=zenith)
/// * `scale` - Pixel size (meters)
/// * `amaxvalue` - Maximum height difference in the DSM (optimization hint)
/// * `bush` - Bush/low vegetation layer (binary or height)
/// * `walls` - Wall height layer
/// * `aspect` - Wall aspect/orientation layer (radians or degrees, consistent with azimuth)
/// * `walls_scheme` - Optional alternative wall height layer for specific calculations
/// * `aspect_scheme` - Optional alternative wall aspect layer
///
/// # Returns
/// * `ShadowingResult` struct containing various shadow maps (ground, vegetation, walls).
pub fn shadowingfunction_wallheight_25(
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
    // --- Input Validation & View Creation ---
    let dsm_view = dsm.as_array();
    let veg_canopy_dsm_view = veg_canopy_dsm.as_array();
    let veg_trunk_dsm_view = veg_trunk_dsm.as_array();
    let bush_view = bush.as_array();
    let walls_view = walls.as_array();
    let aspect_view = aspect.as_array();
    let shape = dsm_view.shape();
    // Ensure all core input arrays have the same dimensions.
    let all_shapes = [
        veg_canopy_dsm_view.shape(),
        veg_trunk_dsm_view.shape(),
        bush_view.shape(),
        walls_view.shape(),
        aspect_view.shape(),
    ];
    if all_shapes.iter().any(|&s| s != shape) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "All input arrays (dsm, veg*, bush, walls, aspect) must have the same shape.",
        ));
    }
    // Validate optional scheme arrays if provided.
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

    // Helper to clamp slice indices within valid array bounds.
    let clamp =
        |v: f64, min: usize, max: usize| -> usize { v.max(min as f64).min(max as f64) as usize };

    let sizex = dsm_view.shape()[0];
    let sizey = dsm_view.shape()[1];

    // --- Initialization ---
    // Shadow step offsets (updated each iteration).
    let mut dx: f64 = 0.0; // Step offset in x-direction
    let mut dy: f64 = 0.0; // Step offset in y-direction
    let mut dz: f64 = 0.0; // Vertical height offset for the current step
    let mut prev_dz: f64 = 0.0; // Vertical height offset from the *previous* step (for pergola logic)

    // Pre-allocate temporary arrays for shifted height maps within the loop.
    let mut temp_dsm_shifted = Array2::<f64>::zeros((sizex, sizey));
    let mut temp_veg_canopy_shifted = Array2::<f64>::zeros((sizex, sizey));
    let mut temp_veg_trunk_shifted = Array2::<f64>::zeros((sizex, sizey));
    let mut temp_prev_veg_canopy_shifted = Array2::<f64>::zeros((sizex, sizey));
    let mut temp_prev_veg_trunk_shifted = Array2::<f64>::zeros((sizex, sizey));

    // Initialize output/intermediate shadow arrays.
    // is_bush_map: Mask indicating pixels considered as low vegetation/bush.
    let is_bush_map = bush_view.mapv(|v| if v > 1.0 { 1.0 } else { 0.0 });
    // bldg_shadow_map: Intermediate map, 1.0 = shadow, 0.0 = sunlit. Finalized later.
    let mut bldg_shadow_map = Array2::<f64>::zeros((sizex, sizey));
    // vbshvegsh: Accumulates vegetation shadow that blocks potential building shadow. Finalized later.
    let mut vbshvegsh = Array2::<f64>::zeros((sizex, sizey));
    // veg_shadow_map: Intermediate map for vegetation shadow (including pergola). Finalized later.
    let mut veg_shadow_map = Array2::<f64>::zeros((sizex, sizey));
    veg_shadow_map.assign(&is_bush_map); // Initialize with bush locations.
                                         // propagated_bldg_shadow_height: Tracks the maximum building height encountered along the sun ray path for each pixel.
    let mut propagated_bldg_shadow_height = Array2::<f64>::zeros((sizex, sizey));
    propagated_bldg_shadow_height.assign(&dsm_view); // Initialize with local DSM height.
                                                     // propagated_veg_shadow_height: Tracks the maximum vegetation canopy height encountered along the sun ray path.
    let mut propagated_veg_shadow_height = Array2::<f64>::zeros((sizex, sizey));
    propagated_veg_shadow_height.assign(&veg_canopy_dsm_view); // Initialize with local canopy height.

    // Pre-calculate trigonometric values and constants for the loop.
    let azimuth_rad = azimuth_deg.to_radians();
    let altitude_rad = altitude_deg.to_radians();
    let sinazimuth = azimuth_rad.sin();
    let cosazimuth = azimuth_rad.cos();
    let tanazimuth = azimuth_rad.tan();
    let signsinazimuth = sinazimuth.signum();
    let signcosazimuth = cosazimuth.signum();
    let dssin = (1.0 / sinazimuth).abs(); // Incremental distance for a unit step in y
    let dscos = (1.0 / cosazimuth).abs(); // Incremental distance for a unit step in x
    let tanaltitudebyscale = altitude_rad.tan() / scale; // Tangent of altitude adjusted by pixel scale
    let mut index = 0.0; // Loop counter, represents steps along the sun ray path.

    // --- Main Shadow Casting Loop ---
    // Iterates outwards from each pixel, simulating the sun's rays.
    // Stops when the potential shadow height (dz) exceeds the max possible height difference (amaxvalue)
    // or when the shadow offset (dx, dy) goes beyond the array boundaries.
    while amaxvalue >= dz && dx.abs() < sizex as f64 && dy.abs() < sizey as f64 {
        // Calculate horizontal shadow step offsets (dx, dy) based on azimuth quadrant.
        // This determines the primary direction (x or y) for stepping at this index.
        if (PI_OVER_4 <= azimuth_rad && azimuth_rad < THREE_PI_OVER_4) // 45 to 135 deg
            || (FIVE_PI_OVER_4 <= azimuth_rad && azimuth_rad < SEVEN_PI_OVER_4)
        // 225 to 315 deg
        {
            // Step primarily in y-direction.
            dy = signsinazimuth * index;
            dx = -1.0 * signcosazimuth * (index / tanazimuth).round().abs();
        } else {
            // Step primarily in x-direction.
            dy = signsinazimuth * (index * tanazimuth).round().abs();
            dx = -1.0 * signcosazimuth * index;
        }

        // Determine incremental distance (ds) along the sun ray for this step.
        let ds = if (PI_OVER_4 <= azimuth_rad && azimuth_rad < THREE_PI_OVER_4)
            || (FIVE_PI_OVER_4 <= azimuth_rad && azimuth_rad < SEVEN_PI_OVER_4)
        {
            dssin // Use distance related to y-step
        } else {
            dscos // Use distance related to x-step
        };

        // Calculate vertical shadow height offset (dz) for this step distance.
        dz = (ds * index) * tanaltitudebyscale;

        // Reset temporary shifted arrays.
        temp_dsm_shifted.fill(0.0);
        temp_veg_canopy_shifted.fill(0.0);
        temp_veg_trunk_shifted.fill(0.0);
        temp_prev_veg_canopy_shifted.fill(0.0);
        temp_prev_veg_trunk_shifted.fill(0.0);

        // Calculate slice indices for shifting arrays by (-dx, -dy).
        // This simulates looking 'back' along the sun ray from the current pixel's perspective.
        // xc1, yc1, xc2, yc2: Source array slice bounds (pixels casting shadow).
        // xp1, yp1, xp2, yp2: Target array slice bounds (pixels receiving shadow).
        let xc1 = clamp((dx + dx.abs()) / 2.0, 0, sizex);
        let xc2 = clamp(sizex as f64 + (dx - dx.abs()) / 2.0, 0, sizex);
        let yc1 = clamp((dy + dy.abs()) / 2.0, 0, sizey);
        let yc2 = clamp(sizey as f64 + (dy - dy.abs()) / 2.0, 0, sizey);
        let xp1 = clamp(-(dx - dx.abs()) / 2.0, 0, sizex);
        let xp2 = clamp(sizex as f64 - (dx + dx.abs()) / 2.0, 0, sizex);
        let yp1 = clamp(-(dy - dy.abs()) / 2.0, 0, sizey);
        let yp2 = clamp(sizey as f64 - (dy + dy.abs()) / 2.0, 0, sizey);

        // Shift DSM, veg_canopy, veg_trunk arrays and subtract dz to get potential shadow caster heights relative to current pixel's ground.
        if xc2 > xc1 && yc2 > yc1 && xp2 > xp1 && yp2 > yp1 {
            // Ensure slice bounds are valid
            // Parallel copy and subtract dz for current step.
            Zip::from(temp_veg_canopy_shifted.slice_mut(s![xp1..xp2, yp1..yp2]))
                .and(veg_canopy_dsm_view.slice(s![xc1..xc2, yc1..yc2]))
                .par_for_each(|target, &source| *target = source - dz);
            Zip::from(temp_veg_trunk_shifted.slice_mut(s![xp1..xp2, yp1..yp2]))
                .and(veg_trunk_dsm_view.slice(s![xc1..xc2, yc1..yc2]))
                .par_for_each(|target, &source| *target = source - dz);
            Zip::from(temp_dsm_shifted.slice_mut(s![xp1..xp2, yp1..yp2]))
                .and(dsm_view.slice(s![xc1..xc2, yc1..yc2]))
                .par_for_each(|target, &source| *target = source - dz);

            // Parallel copy and subtract prev_dz for *previous* step (used in pergola logic).
            Zip::from(temp_prev_veg_canopy_shifted.slice_mut(s![xp1..xp2, yp1..yp2]))
                .and(veg_canopy_dsm_view.slice(s![xc1..xc2, yc1..yc2]))
                .par_for_each(|target, &source| *target = source - prev_dz);
            Zip::from(temp_prev_veg_trunk_shifted.slice_mut(s![xp1..xp2, yp1..yp2]))
                .and(veg_trunk_dsm_view.slice(s![xc1..xc2, yc1..yc2]))
                .par_for_each(|target, &source| *target = source - prev_dz);
        }

        // Update propagated building shadow height: take the maximum height seen so far along the sun ray.
        Zip::from(&mut propagated_bldg_shadow_height)
            .and(&temp_dsm_shifted)
            .par_for_each(|prop_h, &shifted_h| *prop_h = prop_h.max(shifted_h));

        // Update propagated vegetation shadow height: take the maximum canopy height seen so far.
        Zip::from(&mut propagated_veg_shadow_height)
            .and(&temp_veg_canopy_shifted)
            .par_for_each(|prop_veg_h, &shifted_veg_h| *prop_veg_h = prop_veg_h.max(shifted_veg_h));

        // Update intermediate building shadow map: Pixel is shadowed if propagated height > local DSM height.
        Zip::from(&mut bldg_shadow_map)
            .and(&propagated_bldg_shadow_height)
            .and(&dsm_view)
            .par_for_each(|sh_flag, &prop_h, &dsm_h| {
                *sh_flag = if prop_h > dsm_h {
                    SHADOW_FLAG_INTERMEDIATE // Shadow
                } else {
                    SUNLIT_FLAG_INTERMEDIATE // Sunlit
                }
            });

        // --- Pergola Logic ---
        // Detects shadows cast by thin vertical vegetation structures (like pergolas or sparse canopies).
        // A shadow is cast if the sun ray passes between the trunk and canopy level in either the
        // current step (dz) or the previous step (prev_dz), but not through solid canopy at both steps.
        Zip::from(&mut veg_shadow_map)
            .and(&temp_veg_canopy_shifted) // Canopy height at current step distance (relative to ground)
            .and(&temp_veg_trunk_shifted) // Trunk height at current step distance
            .and(&temp_prev_veg_canopy_shifted) // Canopy height at previous step distance
            .and(&temp_prev_veg_trunk_shifted) // Trunk height at previous step distance
            .and(&dsm_view) // Local ground height
            .par_for_each(|v_sh_flag, &tvc_s, &tvt_s, &tpvc_s, &tpvt_s, &dsm_val| {
                // Check four conditions: Is the ray blocked by canopy/trunk at current/previous step?
                let cond1 = if tvc_s > dsm_val { 1.0 } else { 0.0 }; // Current canopy blocks
                let cond2 = if tvt_s > dsm_val { 1.0 } else { 0.0 }; // Current trunk blocks (should be <= cond1)
                let cond3 = if tpvc_s > dsm_val { 1.0 } else { 0.0 }; // Previous canopy blocks
                let cond4 = if tpvt_s > dsm_val { 1.0 } else { 0.0 }; // Previous trunk blocks

                // Sum conditions. A sum between 1 and 3 indicates the ray passed through the 'gap'.
                // Sum = 4 means solid canopy block (handled by regular veg shadow). Sum = 0 means no block.
                let conditions_sum = cond1 + cond2 + cond3 + cond4;

                // Determine if this step contributes pergola shadow.
                let pergola_shadow = if conditions_sum > SUNLIT_FLAG_INTERMEDIATE
                    && conditions_sum < PERGOLA_SOLID_THRESHOLD
                {
                    SHADOW_FLAG_INTERMEDIATE // Pergola shadow
                } else {
                    SUNLIT_FLAG_INTERMEDIATE // No pergola shadow
                };

                // Update overall vegetation shadow map (take max of existing and new pergola shadow).
                *v_sh_flag = f64::max(*v_sh_flag, pergola_shadow);
            });

        prev_dz = dz; // Store current dz for the next iteration's pergola check.

        // Remove vegetation shadow where building shadow already exists (building shadow takes precedence).
        Zip::from(&mut veg_shadow_map)
            .and(&bldg_shadow_map)
            .par_for_each(|v_sh_flag, &b_sh_flag| {
                if *v_sh_flag > SUNLIT_FLAG_INTERMEDIATE && b_sh_flag > SUNLIT_FLAG_INTERMEDIATE {
                    *v_sh_flag = SUNLIT_FLAG_INTERMEDIATE // Remove veg shadow flag
                }
            });

        // Accumulate vegetation shadow that blocks potential building shadow (vbshvegsh).
        // This tracks where *only* vegetation is casting the shadow at this step.
        Zip::from(&mut vbshvegsh)
            .and(&veg_shadow_map)
            .par_for_each(|vbs_acc, &v_sh_flag| *vbs_acc += v_sh_flag); // Accumulate steps where veg shadow exists

        index += 1.0; // Increment step counter.
    } // --- End of Main Shadow Casting Loop ---

    // --- Post-Loop Processing & Finalization ---

    // Finalize building shadow map: Invert flags (0=shadow, 1=sunlit).
    bldg_shadow_map.par_mapv_inplace(|v| FINAL_SUNLIT_VALUE - v); // 1.0 - 1.0 = 0.0 (shadow), 1.0 - 0.0 = 1.0 (sun)

    // Finalize vegetation-blocking-building-shadow map (vbshvegsh).
    vbshvegsh.par_mapv_inplace(|v| if v > 0.0 { 1.0 } else { 0.0 }); // Consolidate accumulated flags to 0 or 1.
                                                                     // Subtract the final *overall* vegetation shadow. This leaves only areas where veg blocked sun *but* buildings would have if veg wasn't there.
                                                                     // The interpretation might need review based on the exact desired output of vbshvegsh.
    vbshvegsh = &vbshvegsh - &veg_shadow_map.mapv(|v| if v > 0.0 { 1.0 } else { 0.0 }); // Use thresholded veg_shadow_map
    vbshvegsh.par_mapv_inplace(|v| 1.0 - v.max(0.0)); // Invert and ensure >= 0. (Result: 1 where building shadow *would* be if not for veg)

    // Finalize vegetation shadow map: Threshold and Invert flags (0=shadow, 1=sunlit).
    veg_shadow_map.par_mapv_inplace(|v| {
        if v > 0.0 {
            SHADOW_FLAG_INTERMEDIATE // Consolidate to 1.0 if any shadow flag was set
        } else {
            SUNLIT_FLAG_INTERMEDIATE
        }
    });
    veg_shadow_map.par_mapv_inplace(|v| FINAL_SUNLIT_VALUE - v); // 1.0 - 1.0 = 0.0 (shadow), 1.0 - 0.0 = 1.0 (sun)

    // Calculate final vegetation shadow *volume height* relative to ground.
    // This is the difference between propagated veg height and DSM height, only where veg shadow exists.
    let final_veg_shadow_mask = veg_shadow_map.mapv(|v| FINAL_SUNLIT_VALUE - v); // Create mask (1=shadow, 0=sun)
                                                                                 // Calculate height difference and apply mask in one step.
    Zip::from(&mut propagated_veg_shadow_height)
        .and(&dsm_view)
        .and(&final_veg_shadow_mask)
        .par_for_each(|prop_veg_h, &dsm_h, &mask| {
            *prop_veg_h = (*prop_veg_h - dsm_h).max(0.0) * mask; // Calculate difference, ensure non-negative, apply mask
        });

    // Calculate wall shadows using the helper function with the main wall/aspect layers.
    let (wallsh, wallsun, wallshve, facesh, facesun) = shade_on_walls(
        azimuth_rad,
        aspect_view,
        walls_view,
        dsm_view,
        propagated_bldg_shadow_height.view(), // Final propagated building shadow height
        propagated_veg_shadow_height.view(),  // Final vegetation shadow volume height
    );

    // --- Optional Scheme Logic ---
    // If alternative wall/aspect schemes are provided, calculate an additional shadow metric.
    let mut shade_on_wall_result: Option<Array2<f64>> = None;
    if let (Some(walls_scheme_py), Some(aspect_scheme_py)) = (walls_scheme, aspect_scheme) {
        let walls_scheme_view = walls_scheme_py.as_array();
        let aspect_scheme_view = aspect_scheme_py.as_array();

        // Call shade_on_walls again using the scheme layers.
        // Note: Propagated shadow heights remain the same, only wall geometry changes.
        let (
            scheme_shadowed_wall_height,
            _scheme_sunlit_wall_height,
            scheme_veg_shadow_on_wall_height,
            _scheme_facesh,
            _scheme_facesun,
        ) = shade_on_walls(
            azimuth_rad,
            aspect_scheme_view, // Use scheme aspect
            walls_scheme_view,  // Use scheme walls
            dsm_view,
            propagated_bldg_shadow_height.view(),
            propagated_veg_shadow_height.view(),
        );

        // Combine results: Calculate total shadow height on the scheme walls (max of building and veg shadow).
        let mut shade_on_wall_combined = Array2::<f64>::zeros(scheme_shadowed_wall_height.dim());
        Zip::from(&mut shade_on_wall_combined)
            .and(&scheme_shadowed_wall_height)
            .and(&scheme_veg_shadow_on_wall_height)
            .par_for_each(|sow, &wsh, &wsv| *sow = f64::max(wsh, wsv)); // Element-wise maximum

        shade_on_wall_result = Some(shade_on_wall_combined);
    }

    // --- Prepare and Return Results ---
    // Convert ndarray results to PyArray and wrap in the ShadowingResult struct.
    let result = ShadowingResult {
        veg_shadow_map: veg_shadow_map.into_pyarray(py).to_owned().into(),
        bldg_shadow_map: bldg_shadow_map.into_pyarray(py).to_owned().into(),
        vbshvegsh: vbshvegsh.into_pyarray(py).to_owned().into(),
        wallsh: wallsh.into_pyarray(py).to_owned().into(),
        wallsun: wallsun.into_pyarray(py).to_owned().into(),
        wallshve: wallshve.into_pyarray(py).to_owned().into(),
        facesh: facesh.into_pyarray(py).to_owned().into(),
        facesun: facesun.into_pyarray(py).to_owned().into(),
        shade_on_wall: shade_on_wall_result.map(|arr| arr.into_pyarray(py).to_owned().into()),
    };

    // Convert the Rust struct into a Python object.
    result
        .into_pyobject(py)
        .map(|bound| bound.unbind().into()) // Convert Py<ShadowingResult> to PyObject
        .map_err(|e| e.into()) // Convert PyO3 error
}
