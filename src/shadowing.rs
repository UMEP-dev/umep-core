use ndarray::{Array2, ArrayView2, Zip};
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::IntoPyObject;

// Constants
const PI_OVER_4: f32 = std::f32::consts::FRAC_PI_4;
const THREE_PI_OVER_4: f32 = 3.0 * PI_OVER_4;
const FIVE_PI_OVER_4: f32 = 5.0 * PI_OVER_4;
const SEVEN_PI_OVER_4: f32 = 7.0 * PI_OVER_4;
const TAU: f32 = std::f32::consts::TAU; // 2 * PI
const EPSILON: f32 = 1e-8; // Small value for float comparisons

/// Rust-native result struct for internal shadow calculations.
pub(crate) struct ShadowingResultRust {
    pub veg_shadow_map: Array2<f32>,
    pub bldg_shadow_map: Array2<f32>,
    pub vbshvegsh: Array2<f32>,
    pub wallsh: Option<Array2<f32>>,
    pub wallsun: Option<Array2<f32>>,
    pub wallshve: Option<Array2<f32>>,
    pub facesh: Option<Array2<f32>>,
    pub facesun: Option<Array2<f32>>,
    pub shade_on_wall: Option<Array2<f32>>,
}

#[pyclass]
/// Result of the shadowing function, containing all output shadow maps (Python version).
pub struct ShadowingResult {
    #[pyo3(get)]
    pub veg_shadow_map: Py<PyArray2<f32>>,
    #[pyo3(get)]
    pub bldg_shadow_map: Py<PyArray2<f32>>,
    #[pyo3(get)]
    /// Vegetation Blocking Building Shadow: Indicates where vegetation prevents building shadow.
    pub vbshvegsh: Py<PyArray2<f32>>,
    #[pyo3(get)]
    pub wallsh: Option<Py<PyArray2<f32>>>, // Shadowed wall height (by buildings) - Optional
    #[pyo3(get)]
    pub wallsun: Option<Py<PyArray2<f32>>>, // Sunlit wall height - Optional
    #[pyo3(get)]
    pub wallshve: Option<Py<PyArray2<f32>>>, // Wall height shadowed by vegetation - Optional
    #[pyo3(get)]
    pub facesh: Option<Py<PyArray2<f32>>>, // Wall face shadow mask (1 if face away from sun) - Optional
    #[pyo3(get)]
    pub facesun: Option<Py<PyArray2<f32>>>, // Sunlit wall face mask (1 if face towards sun and not obstructed) - Optional
    #[pyo3(get)]
    /// Combined building and vegetation shadow height on walls (optional scheme).
    pub shade_on_wall: Option<Py<PyArray2<f32>>>,
}

// Helper function to safely get values from ArrayView2, handling out-of-bounds
#[inline]
fn get_view_value(view: &ArrayView2<f32>, r: isize, c: isize, rows: usize, cols: usize) -> f32 {
    if r >= 0 && r < rows as isize && c >= 0 && c < cols as isize {
        // Safety: Bounds check performed above
        unsafe { *view.uget([r as usize, c as usize]) }
    } else {
        0.0 // Return 0.0 for out-of-bounds access, mimicking shift behavior
    }
}

/// Internal Rust function for shadow calculations.
/// Operates purely on ndarray types.
#[allow(clippy::too_many_arguments)] // Keep arguments for clarity
pub(crate) fn calculate_shadows_rust(
    dsm_view: ArrayView2<f32>,
    veg_canopy_dsm_view: ArrayView2<f32>,
    veg_trunk_dsm_view: ArrayView2<f32>,
    azimuth_deg: f32,
    altitude_deg: f32,
    scale: f32,
    amaxvalue: f32,
    bush_view: ArrayView2<f32>,
    walls_view_opt: Option<ArrayView2<f32>>,
    aspect_view_opt: Option<ArrayView2<f32>>,
    walls_scheme_view_opt: Option<ArrayView2<f32>>,
    aspect_scheme_view_opt: Option<ArrayView2<f32>>,
) -> ShadowingResultRust {
    let shape = dsm_view.shape();
    let sizex = shape[0];
    let sizey = shape[1];

    let mut dx: f32 = 0.0;
    let mut dy: f32 = 0.0;
    let mut dz: f32 = 0.0;
    let mut ds: f32;
    let mut prev_dz: f32 = 0.0;

    // Bush mask threshold should match Python: bushplant = bush > 1
    let is_bush_map = bush_view.mapv(|v| if v > 1.0 { 1.0 } else { 0.0 });
    let mut bldg_shadow_map = Array2::<f32>::zeros((sizex, sizey));
    let mut vbshvegsh = Array2::<f32>::zeros((sizex, sizey));
    // Initialize veg_shadow_map with bush locations
    let mut veg_shadow_map = is_bush_map;

    let mut propagated_bldg_shadow_height = Array2::<f32>::zeros((sizex, sizey));
    propagated_bldg_shadow_height.assign(&dsm_view);
    let mut propagated_veg_shadow_height = Array2::<f32>::zeros((sizex, sizey));
    propagated_veg_shadow_height.assign(&veg_canopy_dsm_view);

    let azimuth_rad = azimuth_deg.to_radians();
    let altitude_rad = altitude_deg.to_radians();
    let sinazimuth = azimuth_rad.sin();
    let cosazimuth = azimuth_rad.cos();
    let tanazimuth = azimuth_rad.tan();
    let signsinazimuth = sinazimuth.signum();
    let signcosazimuth = cosazimuth.signum();
    let dssin = (1.0 / sinazimuth).abs();
    let dscos = (1.0 / cosazimuth).abs();
    let tanaltitudebyscale = altitude_rad.tan() / scale;
    let mut index = 0.0;

    while amaxvalue >= dz && dx.abs() < sizex as f32 && dy.abs() < sizey as f32 {
        // Calculate offsets dx, dy, dz based on index and sun angles
        if (PI_OVER_4 <= azimuth_rad && azimuth_rad < THREE_PI_OVER_4)
            || (FIVE_PI_OVER_4 <= azimuth_rad && azimuth_rad < SEVEN_PI_OVER_4)
        {
            dy = signsinazimuth * index;
            dx = -1.0 * signcosazimuth * (index / tanazimuth).round().abs();
            ds = dssin;
        } else {
            dy = signsinazimuth * (index * tanazimuth).round().abs();
            dx = -1.0 * signcosazimuth * index;
            ds = dscos;
        }
        dz = (ds * index) * tanaltitudebyscale;

        // --- Update propagated heights and shadow maps directly ---
        Zip::indexed(&mut propagated_bldg_shadow_height) // 1 (Indices)
            .and(&mut propagated_veg_shadow_height) // 2
            .and(&mut bldg_shadow_map) // 3
            .and(&mut veg_shadow_map) // 4
            // Removed .and(&dsm_view) - Now 5 items total
            .par_for_each(
                |(tx, ty), // Index from indexed()
                 prop_bldg_h,
                 prop_veg_h,
                 bldg_sh_flag,
                 veg_sh_flag| {
                    // Get target DSM height using the index
                    let dsm_h_target = dsm_view[[tx, ty]];

                    // Calculate source coordinates (integer for indexing)
                    // Apply offset consistent with the working slice logic (target + offset)
                    let sx_i = (tx as f32 + dx).round() as isize;
                    let sy_i = (ty as f32 + dy).round() as isize;

                    // Fetch source values safely using helper function
                    let source_dsm = get_view_value(&dsm_view, sx_i, sy_i, sizex, sizey);
                    let source_veg_canopy =
                        get_view_value(&veg_canopy_dsm_view, sx_i, sy_i, sizex, sizey);
                    let source_veg_trunk =
                        get_view_value(&veg_trunk_dsm_view, sx_i, sy_i, sizex, sizey);

                    // Calculate shifted heights for the current target pixel
                    let shifted_dsm = source_dsm - dz;
                    let shifted_veg_canopy = source_veg_canopy - dz;
                    let shifted_veg_trunk = source_veg_trunk - dz;
                    let prev_shifted_veg_canopy = source_veg_canopy - prev_dz;
                    let prev_shifted_veg_trunk = source_veg_trunk - prev_dz;

                    // Update propagated building height
                    *prop_bldg_h = prop_bldg_h.max(shifted_dsm);

                    // Update propagated veg height
                    *prop_veg_h = prop_veg_h.max(shifted_veg_canopy);

                    // Update building shadow flag (based on updated propagated height at target)
                    *bldg_sh_flag = if *prop_bldg_h > dsm_h_target {
                        // Add epsilon for float comparison
                        1.0
                    } else {
                        0.0
                    };

                    // Update veg shadow flag (Pergola logic)
                    // Compare shifted heights with target DSM height
                    let cond1 = if shifted_veg_canopy > dsm_h_target {
                        1.0
                    } else {
                        0.0
                    };
                    let cond2 = if shifted_veg_trunk > dsm_h_target {
                        1.0
                    } else {
                        0.0
                    };
                    let cond3 = if prev_shifted_veg_canopy > dsm_h_target {
                        1.0
                    } else {
                        0.0
                    };
                    let cond4 = if prev_shifted_veg_trunk > dsm_h_target {
                        1.0
                    } else {
                        0.0
                    };
                    let conditions_sum = cond1 + cond2 + cond3 + cond4;

                    let pergola_shadow = if conditions_sum > 0.0 && conditions_sum < 4.0 {
                        1.0
                    } else {
                        0.0
                    };
                    // Update main veg shadow flag (accumulates)
                    *veg_sh_flag = f32::max(*veg_sh_flag, pergola_shadow);
                },
            );

        // --- End of combined update ---

        // Clear veg_shadow_map where building shadow exists FIRST (matches Python order)
        Zip::from(&mut veg_shadow_map)
            .and(&bldg_shadow_map)
            .par_for_each(|v_sh_flag, &b_sh_flag| {
                if *v_sh_flag > 0.0 && b_sh_flag > 0.0 {
                    *v_sh_flag = 0.0;
                }
            });

        // Accumulate the potentially cleared veg_shadow_map into vbshvegsh SECOND
        // Use veg_shadow_map which holds the max accumulated shadow flag up to this point,
        // potentially cleared by building shadow in the step above.
        Zip::from(&mut vbshvegsh)
            .and(&veg_shadow_map) // Use the potentially cleared map
            .par_for_each(|vbs_acc, &v_sh_flag| {
                // Use the flag from veg_shadow_map
                if v_sh_flag > 0.0 {
                    // Check against 0.0 (consistent with old logic's implicit check)
                    *vbs_acc += v_sh_flag; // Accumulate the potentially persistent flag
                }
            });

        prev_dz = dz;

        index += 1.0;
    } // End of while loop

    // --- Final processing of shadow maps ---
    bldg_shadow_map.par_mapv_inplace(|v| 1.0 - v);

    vbshvegsh.par_mapv_inplace(|v| if v > 0.0 { 1.0 } else { 0.0 });
    vbshvegsh = &vbshvegsh - &veg_shadow_map.mapv(|v| if v > 0.0 { 1.0 } else { 0.0 });
    vbshvegsh.par_mapv_inplace(|v| 1.0 - v.max(0.0));

    veg_shadow_map.par_mapv_inplace(|v| if v > 0.0 { 1.0 } else { 0.0 });
    veg_shadow_map.par_mapv_inplace(|v| 1.0 - v);

    let final_veg_shadow_mask = veg_shadow_map.mapv(|v| 1.0 - v);
    Zip::from(&mut propagated_veg_shadow_height)
        .and(&dsm_view)
        .and(&final_veg_shadow_mask)
        .par_for_each(|prop_veg_h, &dsm_h, &mask| {
            *prop_veg_h = (*prop_veg_h - dsm_h).max(0.0) * mask;
        });

    let mut wallsh_opt: Option<Array2<f32>> = None;
    let mut wallsun_opt: Option<Array2<f32>> = None;
    let mut wallshve_opt: Option<Array2<f32>> = None;
    let mut facesh_opt: Option<Array2<f32>> = None;
    let mut facesun_opt: Option<Array2<f32>> = None;
    let mut shade_on_wall_result: Option<Array2<f32>> = None;

    if let (Some(walls_view), Some(aspect_view)) = (walls_view_opt, aspect_view_opt) {
        let (wallsh, wallsun, wallshve, facesh, facesun) = shade_on_walls(
            azimuth_rad,
            aspect_view,
            walls_view,
            dsm_view,
            propagated_bldg_shadow_height.view(),
            propagated_veg_shadow_height.view(),
        );
        wallsh_opt = Some(wallsh);
        wallsun_opt = Some(wallsun);
        wallshve_opt = Some(wallshve);
        facesh_opt = Some(facesh);
        facesun_opt = Some(facesun);

        if let (Some(walls_scheme_view), Some(aspect_scheme_view)) =
            (walls_scheme_view_opt, aspect_scheme_view_opt)
        {
            let (
                scheme_shadowed_wall_height,
                _scheme_sunlit_wall_height,
                scheme_veg_shadow_on_wall_height,
                _scheme_facesh,
                _scheme_facesun,
            ) = shade_on_walls(
                azimuth_rad,
                aspect_scheme_view,
                walls_scheme_view,
                dsm_view,
                propagated_bldg_shadow_height.view(),
                propagated_veg_shadow_height.view(),
            );
            let mut shade_on_wall_combined =
                Array2::<f32>::zeros(scheme_shadowed_wall_height.dim());
            Zip::from(&mut shade_on_wall_combined)
                .and(&scheme_shadowed_wall_height)
                .and(&scheme_veg_shadow_on_wall_height)
                .par_for_each(|sow, &wsh, &wsv| *sow = f32::max(wsh, wsv));
            shade_on_wall_result = Some(shade_on_wall_combined);
        }
    }

    ShadowingResultRust {
        veg_shadow_map,
        bldg_shadow_map,
        vbshvegsh,
        wallsh: wallsh_opt,
        wallsun: wallsun_opt,
        wallshve: wallshve_opt,
        facesh: facesh_opt,
        facesun: facesun_opt,
        shade_on_wall: shade_on_wall_result,
    }
}

fn shade_on_walls(
    azimuth: f32,
    aspect: ArrayView2<f32>,
    walls: ArrayView2<f32>,
    dsm: ArrayView2<f32>,
    propagated_bldg_shadow_height: ArrayView2<f32>,
    propagated_veg_shadow_height: ArrayView2<f32>,
) -> (
    Array2<f32>,
    Array2<f32>,
    Array2<f32>,
    Array2<f32>,
    Array2<f32>,
) {
    let shape = walls.dim();
    let mut wall_mask = Array2::<f32>::zeros(shape);
    Zip::from(&mut wall_mask)
        .and(&walls)
        .par_for_each(|mask_val, &wall_h| *mask_val = if wall_h > 0.0 { 1.0 } else { 0.0 });

    let azilow = azimuth - std::f32::consts::FRAC_PI_2;
    let azihigh = azimuth + std::f32::consts::FRAC_PI_2;
    let mut wall_face_shadow_mask = Array2::<f32>::zeros(shape);
    if azilow >= 0.0 && azihigh < TAU {
        Zip::from(&mut wall_face_shadow_mask)
            .and(aspect)
            .and(&wall_mask)
            .par_for_each(|f_sh, &asp, &w_mask| {
                *f_sh = if asp < azilow || asp >= azihigh {
                    1.0
                } else {
                    0.0
                } - w_mask
                    + 1.0;
            });
    } else if azilow < 0.0 && azihigh <= TAU {
        let azilow_wrapped = azilow + TAU;
        Zip::from(&mut wall_face_shadow_mask)
            .and(aspect)
            .par_for_each(|f_sh, &asp| {
                *f_sh = if asp > azilow_wrapped || asp <= azihigh {
                    -1.0
                } else {
                    0.0
                } + 1.0;
            });
    } else {
        // if azilow > 0.0 && azihigh >= TAU
        let azihigh_wrapped = azihigh - TAU;
        Zip::from(&mut wall_face_shadow_mask)
            .and(aspect)
            .par_for_each(|f_sh, &asp| {
                *f_sh = if asp > azilow || asp <= azihigh_wrapped {
                    -1.0
                } else {
                    0.0
                } + 1.0;
            });
    }

    let mut building_shadow_volume_height = Array2::<f32>::zeros(shape);
    Zip::from(&mut building_shadow_volume_height)
        .and(&propagated_bldg_shadow_height)
        .and(&dsm)
        .par_for_each(|sh_vol, &prop_h, &dsm_h| *sh_vol = prop_h - dsm_h);

    let mut sunlit_wall_face_mask = Array2::<f32>::zeros(shape);
    Zip::from(&mut sunlit_wall_face_mask)
        .and(&wall_face_shadow_mask)
        .and(walls)
        .par_for_each(|sf_mask, &f_sh_mask, &w_h| {
            let wall_exists = w_h > 0.0;
            let wall_exists_flag = if wall_exists { 1.0 } else { 0.0 };
            *sf_mask = if (f_sh_mask + wall_exists_flag) == 1.0 && wall_exists {
                1.0
            } else {
                0.0
            };
        });

    let mut sunlit_wall_height = Array2::<f32>::zeros(shape);
    Zip::from(&mut sunlit_wall_height)
        .and(&walls)
        .and(&building_shadow_volume_height)
        .par_for_each(|sun_h, &wall_h, &sh_vol_h| *sun_h = wall_h - sh_vol_h);
    sunlit_wall_height.par_mapv_inplace(|v| v.max(0.0));
    Zip::from(&mut sunlit_wall_height)
        .and(&wall_face_shadow_mask)
        .par_for_each(|sun_h, &f_sh_mask| {
            if (f_sh_mask - 1.0).abs() < EPSILON {
                *sun_h = 0.0
            }
        });

    let mut shadowed_wall_height = Array2::<f32>::zeros(shape);
    Zip::from(&mut shadowed_wall_height)
        .and(&walls)
        .and(&sunlit_wall_height)
        .par_for_each(|sh_h, &wall_h, &sun_h| *sh_h = wall_h - sun_h);

    let mut veg_shadow_on_wall_height = Array2::<f32>::zeros(shape);
    Zip::from(&mut veg_shadow_on_wall_height)
        .and(&propagated_veg_shadow_height)
        .and(&wall_mask)
        .par_for_each(|veg_sh_h, &prop_veg_h, &w_mask| *veg_sh_h = prop_veg_h * w_mask);
    Zip::from(&mut veg_shadow_on_wall_height)
        .and(&shadowed_wall_height)
        .par_for_each(|veg_sh_h, &bldg_sh_h| *veg_sh_h -= bldg_sh_h);
    veg_shadow_on_wall_height.par_mapv_inplace(|v| v.max(0.0));
    Zip::from(&mut veg_shadow_on_wall_height)
        .and(walls)
        .par_for_each(|veg_sh_h, &wall_h| {
            if *veg_sh_h > wall_h {
                *veg_sh_h = wall_h
            }
        });
    Zip::from(&mut sunlit_wall_height)
        .and(&veg_shadow_on_wall_height)
        .par_for_each(|sun_h, &veg_sh_h| *sun_h -= veg_sh_h);
    Zip::from(&mut veg_shadow_on_wall_height)
        .and(&sunlit_wall_height)
        .par_for_each(|veg_sh_h, &sun_h| {
            if sun_h < 0.0 {
                *veg_sh_h = 0.0
            }
        });
    sunlit_wall_height.par_mapv_inplace(|v| v.max(0.0));

    (
        shadowed_wall_height,
        sunlit_wall_height,
        veg_shadow_on_wall_height,
        wall_face_shadow_mask,
        sunlit_wall_face_mask,
    )
}

#[pyfunction]
/// Calculates shadow maps for buildings, vegetation, and walls given DSM and sun position (Python wrapper).
///
/// This function handles Python type conversions and calls the internal Rust shadow calculation logic.
/// See `calculate_shadows_rust` for core algorithm details.
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
/// * `walls` - Optional wall height layer. If None, wall calculations are skipped.
/// * `aspect` - Optional wall aspect/orientation layer (radians or degrees). Required if `walls` is provided.
/// * `walls_scheme` - Optional alternative wall height layer for specific calculations
/// * `aspect_scheme` - Optional alternative wall aspect layer
///
/// # Returns
/// * `ShadowingResult` struct containing various shadow maps (ground, vegetation, walls) as PyArrays.
pub fn shadowingfunction_wallheight_25(
    py: Python,
    dsm: PyReadonlyArray2<f32>,
    veg_canopy_dsm: PyReadonlyArray2<f32>,
    veg_trunk_dsm: PyReadonlyArray2<f32>,
    azimuth_deg: f32,
    altitude_deg: f32,
    scale: f32,
    amaxvalue: f32,
    bush: PyReadonlyArray2<f32>,
    walls: Option<PyReadonlyArray2<f32>>,
    aspect: Option<PyReadonlyArray2<f32>>,
    walls_scheme: Option<PyReadonlyArray2<f32>>,
    aspect_scheme: Option<PyReadonlyArray2<f32>>,
) -> PyResult<PyObject> {
    let dsm_view = dsm.as_array();
    let veg_canopy_dsm_view = veg_canopy_dsm.as_array();
    let veg_trunk_dsm_view = veg_trunk_dsm.as_array();
    let bush_view = bush.as_array();
    let shape = dsm_view.shape();

    let core_shapes = [
        veg_canopy_dsm_view.shape(),
        veg_trunk_dsm_view.shape(),
        bush_view.shape(),
    ];
    if core_shapes.iter().any(|&s| s != shape) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Input arrays (dsm, veg*, bush) must have the same shape.",
        ));
    }

    let walls_view_opt = walls.as_ref().map(|w| w.as_array());
    let aspect_view_opt = aspect.as_ref().map(|a| a.as_array());
    if walls_view_opt.is_some() != aspect_view_opt.is_some() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Both 'walls' and 'aspect' must be provided together, or both must be None.",
        ));
    }
    if let Some(walls_view) = walls_view_opt {
        if walls_view.shape() != shape {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "walls must have the same shape as dsm.",
            ));
        }
    }
    if let Some(aspect_view) = aspect_view_opt {
        if aspect_view.shape() != shape {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "aspect must have the same shape as dsm.",
            ));
        }
    }
    let walls_scheme_view_opt = walls_scheme.as_ref().map(|w| w.as_array());
    let aspect_scheme_view_opt = aspect_scheme.as_ref().map(|a| a.as_array());
    if walls_scheme_view_opt.is_some() != aspect_scheme_view_opt.is_some() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Both 'walls_scheme' and 'aspect_scheme' must be provided together, or both must be None.",
        ));
    }
    if let Some(walls_scheme_view) = walls_scheme_view_opt {
        if walls_scheme_view.shape() != shape {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "walls_scheme must have the same shape as dsm.",
            ));
        }
    }
    if let Some(aspect_scheme_view) = aspect_scheme_view_opt {
        if aspect_scheme_view.shape() != shape {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "aspect_scheme must have the same shape as dsm.",
            ));
        }
    }

    let rust_result = calculate_shadows_rust(
        dsm_view,
        veg_canopy_dsm_view,
        veg_trunk_dsm_view,
        azimuth_deg,
        altitude_deg,
        scale,
        amaxvalue,
        bush_view,
        walls_view_opt,
        aspect_view_opt,
        walls_scheme_view_opt,
        aspect_scheme_view_opt,
    );

    let py_result = ShadowingResult {
        veg_shadow_map: rust_result
            .veg_shadow_map
            .into_pyarray(py)
            .to_owned()
            .into(),
        bldg_shadow_map: rust_result
            .bldg_shadow_map
            .into_pyarray(py)
            .to_owned()
            .into(),
        vbshvegsh: rust_result.vbshvegsh.into_pyarray(py).to_owned().into(),
        wallsh: rust_result
            .wallsh
            .map(|arr| arr.into_pyarray(py).to_owned().into()),
        wallsun: rust_result
            .wallsun
            .map(|arr| arr.into_pyarray(py).to_owned().into()),
        wallshve: rust_result
            .wallshve
            .map(|arr| arr.into_pyarray(py).to_owned().into()),
        facesh: rust_result
            .facesh
            .map(|arr| arr.into_pyarray(py).to_owned().into()),
        facesun: rust_result
            .facesun
            .map(|arr| arr.into_pyarray(py).to_owned().into()),
        shade_on_wall: rust_result
            .shade_on_wall
            .map(|arr| arr.into_pyarray(py).to_owned().into()),
    };

    py_result
        .into_pyobject(py)
        .map(|bound| bound.unbind().into())
        .map_err(|e| e.into())
}
