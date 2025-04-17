use ndarray::{s, Array2, ArrayView2, Zip};
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::IntoPyObject;

#[pyclass]
pub struct ShadowingResult {
    #[pyo3(get)]
    pub vegsh: Py<PyArray2<f64>>,
    #[pyo3(get)]
    pub sh: Py<PyArray2<f64>>,
    #[pyo3(get)]
    pub vbshvegsh: Py<PyArray2<f64>>,
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
}

fn shade_on_walls(
    azimuth: f64,
    aspect: ArrayView2<f64>,
    walls: ArrayView2<f64>,
    dsm: ArrayView2<f64>,
    f: ArrayView2<f64>,
    shvoveg: ArrayView2<f64>,
) -> (
    Array2<f64>,
    Array2<f64>,
    Array2<f64>,
    Array2<f64>,
    Array2<f64>,
) {
    let shape = walls.dim();
    let mut wallbol = Array2::<f64>::zeros(shape);
    Zip::from(&mut wallbol)
        .and(&walls)
        .for_each(|w, &v| *w = if v > 0.0 { 1.0 } else { 0.0 });

    let azilow = azimuth - std::f64::consts::FRAC_PI_2;
    let mut azihigh = azimuth + std::f64::consts::FRAC_PI_2;
    let mut facesh = Array2::<f64>::zeros(shape);
    if azilow >= 0.0 && azihigh < 2.0 * std::f64::consts::PI {
        Zip::from(&mut facesh)
            .and(aspect)
            .and(&wallbol)
            .for_each(|f, &asp, &wb| {
                *f = (if asp < azilow || asp >= azihigh {
                    1.0
                } else {
                    0.0
                }) - wb
                    + 1.0;
            });
    } else if azilow < 0.0 && azihigh <= 2.0 * std::f64::consts::PI {
        let azilow = azilow + 2.0 * std::f64::consts::PI;
        Zip::from(&mut facesh).and(aspect).for_each(|f, &asp| {
            *f = (if asp > azilow || asp <= azihigh {
                -1.0
            } else {
                0.0
            }) + 1.0;
        });
    } else if azilow > 0.0 && azihigh >= 2.0 * std::f64::consts::PI {
        azihigh -= 2.0 * std::f64::consts::PI;
        Zip::from(&mut facesh).and(aspect).for_each(|f, &asp| {
            *f = (if asp > azilow || asp <= azihigh {
                -1.0
            } else {
                0.0
            }) + 1.0;
        });
    }

    let mut shvo = Array2::<f64>::zeros(shape);
    Zip::from(&mut shvo)
        .and(&f)
        .and(&dsm)
        .for_each(|s, &fv, &dv| *s = fv - dv);

    let mut facesun = Array2::<f64>::zeros(shape);
    Zip::from(&mut facesun)
        .and(&facesh)
        .and(walls)
        .for_each(|fs, &fh, &w| {
            *fs = if (fh + if w > 0.0 { 1.0 } else { 0.0 }) == 1.0 && w > 0.0 {
                1.0
            } else {
                0.0
            };
        });

    let mut wallsun = Array2::<f64>::zeros(shape);
    Zip::from(&mut wallsun)
        .and(&walls)
        .and(&shvo)
        .for_each(|w, &wa, &shv| *w = wa - shv);
    wallsun.mapv_inplace(|v| if v < 0.0 { 0.0 } else { v });
    Zip::from(&mut wallsun).and(&facesh).for_each(|w, &fh| {
        if fh == 1.0 {
            *w = 0.0
        }
    });

    let mut wallsh = Array2::<f64>::zeros(shape);
    Zip::from(&mut wallsh)
        .and(&walls)
        .and(&wallsun)
        .for_each(|w, &wa, &wsu| *w = wa - wsu);

    let mut wallshve = Array2::<f64>::zeros(shape);
    Zip::from(&mut wallshve)
        .and(&shvoveg)
        .and(&wallbol)
        .for_each(|w, &sv, &wb| *w = sv * wb);
    wallshve = &wallshve - &wallsh;
    wallshve.mapv_inplace(|v| if v < 0.0 { 0.0 } else { v });
    Zip::from(&mut wallshve).and(walls).for_each(|wsv, &w| {
        if *wsv > w {
            *wsv = w
        }
    });
    wallsun = &wallsun - &wallshve;
    Zip::from(&mut wallshve)
        .and(&wallsun)
        .for_each(|wsv, &wsu| {
            if wsu < 0.0 {
                *wsv = 0.0
            }
        });
    wallsun.mapv_inplace(|v| if v < 0.0 { 0.0 } else { v });

    (wallsh, wallsun, wallshve, facesh, facesun)
}

#[pyfunction]
pub fn shadowingfunction_wallheight_23(
    py: Python,
    a: PyReadonlyArray2<f64>,
    vegdem: PyReadonlyArray2<f64>,
    vegdem2: PyReadonlyArray2<f64>,
    azimuth: f64,
    altitude: f64,
    scale: f64,
    amaxvalue: f64,
    bush: PyReadonlyArray2<f64>,
    walls: PyReadonlyArray2<f64>,
    aspect: PyReadonlyArray2<f64>,
) -> PyResult<PyObject> {
    let a = a.as_array();
    let vegdem = vegdem.as_array();
    let vegdem2 = vegdem2.as_array();
    let bush = bush.as_array();
    let walls = walls.as_array();
    let aspect = aspect.as_array();
    let degrees = std::f64::consts::PI / 180.0;
    let azimuth = azimuth * degrees;
    let altitude = altitude * degrees;
    let sizex = a.shape()[0];
    let sizey = a.shape()[1];
    let mut dx: f64 = 0.0;
    let mut dy: f64 = 0.0;
    let mut dz: f64 = 0.0;
    let mut temp = Array2::<f64>::zeros((sizex, sizey));
    let mut tempvegdem = Array2::<f64>::zeros((sizex, sizey));
    let mut tempvegdem2 = Array2::<f64>::zeros((sizex, sizey));
    let mut templastfabovea = Array2::<f64>::zeros((sizex, sizey));
    let mut templastgabovea = Array2::<f64>::zeros((sizex, sizey));
    let bushplant = bush.mapv(|v| if v > 1.0 { 1.0 } else { 0.0 });
    let mut sh = Array2::<f64>::zeros((sizex, sizey));
    let mut vbshvegsh = Array2::<f64>::zeros((sizex, sizey));
    let mut vegsh = bushplant.clone();
    let mut f = a.to_owned();
    let mut shvoveg = vegdem.to_owned();
    let pibyfour = std::f64::consts::FRAC_PI_4;
    let threetimespibyfour = 3.0 * pibyfour;
    let fivetimespibyfour = 5.0 * pibyfour;
    let seventimespibyfour = 7.0 * pibyfour;
    let sinazimuth = azimuth.sin();
    let cosazimuth = azimuth.cos();
    let tanazimuth = azimuth.tan();
    let signsinazimuth = sinazimuth.signum();
    let signcosazimuth = cosazimuth.signum();
    let dssin = (1.0 / sinazimuth).abs();
    let dscos = (1.0 / cosazimuth).abs();
    let tanaltitudebyscale = altitude.tan() / scale;
    let mut index = 0.0;
    let mut dzprev = 0.0;
    while amaxvalue >= dz && dx.abs() < sizex as f64 && dy.abs() < sizey as f64 {
        if (pibyfour <= azimuth && azimuth < threetimespibyfour)
            || (fivetimespibyfour <= azimuth && azimuth < seventimespibyfour)
        {
            dy = signsinazimuth * index;
            dx = -1.0 * signcosazimuth * (index / tanazimuth).round().abs();
        } else {
            dy = signsinazimuth * (index * tanazimuth).round().abs();
            dx = -1.0 * signcosazimuth * index;
        }
        let ds = if (pibyfour <= azimuth && azimuth < threetimespibyfour)
            || (fivetimespibyfour <= azimuth && azimuth < seventimespibyfour)
        {
            dssin
        } else {
            dscos
        };
        dz = (ds * index) * tanaltitudebyscale;
        temp.fill(0.0);
        tempvegdem.fill(0.0);
        tempvegdem2.fill(0.0);
        templastfabovea.fill(0.0);
        templastgabovea.fill(0.0);
        let absdx = dx.abs() as usize;
        let absdy = dy.abs() as usize;
        let xc1 = ((dx + dx.abs()) / 2.0) as usize;
        let xc2 = (sizex as f64 + (dx - dx.abs()) / 2.0) as usize;
        let yc1 = ((dy + dy.abs()) / 2.0) as usize;
        let yc2 = (sizey as f64 + (dy - dy.abs()) / 2.0) as usize;
        let xp1 = (-(dx - dx.abs()) / 2.0) as usize;
        let xp2 = (sizex as f64 - (dx + dx.abs()) / 2.0) as usize;
        let yp1 = (-(dy - dy.abs()) / 2.0) as usize;
        let yp2 = (sizey as f64 - (dy + dy.abs()) / 2.0) as usize;
        if xc2 > xc1 && yc2 > yc1 && xp2 > xp1 && yp2 > yp1 {
            tempvegdem
                .slice_mut(s![xp1..xp2, yp1..yp2])
                .assign(&(&vegdem.slice(s![xc1..xc2, yc1..yc2]) - dz));
            tempvegdem2
                .slice_mut(s![xp1..xp2, yp1..yp2])
                .assign(&(&vegdem2.slice(s![xc1..xc2, yc1..yc2]) - dz));
            temp.slice_mut(s![xp1..xp2, yp1..yp2])
                .assign(&(&a.slice(s![xc1..xc2, yc1..yc2]) - dz));
        }
        Zip::from(&mut f)
            .and(&temp)
            .for_each(|fv, &tv| *fv = fv.max(tv));
        Zip::from(&mut shvoveg)
            .and(&tempvegdem)
            .for_each(|sv, &tv| *sv = sv.max(tv));
        Zip::from(&mut sh)
            .and(&f)
            .and(&a)
            .for_each(|s, &fv, &av| *s = if fv > av { 1.0 } else { 0.0 });
        let mut fabovea = Array2::<f64>::zeros(tempvegdem.dim());
        Zip::from(&mut fabovea)
            .and(&tempvegdem)
            .and(&a)
            .for_each(|fab, &tv, &av| *fab = if tv > av { 1.0 } else { 0.0 });
        let mut gabovea = Array2::<f64>::zeros(tempvegdem2.dim());
        Zip::from(&mut gabovea)
            .and(&tempvegdem2)
            .and(&a)
            .for_each(|gab, &tv, &av| *gab = if tv > av { 1.0 } else { 0.0 });
        if xc2 > xc1 && yc2 > yc1 && xp2 > xp1 && yp2 > yp1 {
            templastfabovea
                .slice_mut(s![xp1..xp2, yp1..yp2])
                .assign(&(&vegdem.slice(s![xc1..xc2, yc1..yc2]) - dzprev));
            templastgabovea
                .slice_mut(s![xp1..xp2, yp1..yp2])
                .assign(&(&vegdem2.slice(s![xc1..xc2, yc1..yc2]) - dzprev));
        }
        let mut lastfabovea = Array2::<f64>::zeros(templastfabovea.dim());
        Zip::from(&mut lastfabovea)
            .and(&templastfabovea)
            .and(&a)
            .for_each(|fab, &tv, &av| *fab = if tv > av { 1.0 } else { 0.0 });
        let mut lastgabovea = Array2::<f64>::zeros(templastgabovea.dim());
        Zip::from(&mut lastgabovea)
            .and(&templastgabovea)
            .and(&a)
            .for_each(|gab, &tv, &av| *gab = if tv > av { 1.0 } else { 0.0 });
        dzprev = dz;
        let mut vegsh2 = &fabovea + &gabovea + &lastfabovea + &lastgabovea;
        vegsh2.mapv_inplace(|v| if v == 4.0 { 0.0 } else { v });
        vegsh2.mapv_inplace(|v| if v > 0.0 { 1.0 } else { v });
        Zip::from(&mut vegsh)
            .and(&vegsh2)
            .for_each(|v, &v2| *v = f64::max(*v, v2));
        Zip::from(&mut vegsh).and(&sh).for_each(|v, &s| {
            if *v * s > 0.0 {
                *v = 0.0
            }
        });
        vbshvegsh = &vegsh + &vbshvegsh;
        vbshvegsh.mapv_inplace(|v| if v > 0.0 { 1.0 } else { v });
        vbshvegsh = &vbshvegsh - &vegsh;
        vbshvegsh.mapv_inplace(|v| 1.0 - v);
        vegsh.mapv_inplace(|v| if v > 0.0 { 1.0 } else { v });
        vegsh.mapv_inplace(|v| 1.0 - v);
        shvoveg = (&shvoveg - &a) * &vegsh;
        index += 1.0;
    }
    sh.mapv_inplace(|v| 1.0 - v);
    vbshvegsh.mapv_inplace(|v| if v > 0.0 { 1.0 } else { v });
    vbshvegsh = &vbshvegsh - &vegsh;
    vegsh.mapv_inplace(|v| if v > 0.0 { 1.0 } else { v });
    shvoveg = (&shvoveg - &a) * &vegsh;
    vegsh.mapv_inplace(|v| 1.0 - v);
    vbshvegsh.mapv_inplace(|v| 1.0 - v);
    let (wallsh, wallsun, wallshve, facesh, facesun) = shade_on_walls(
        azimuth,
        aspect.view(),
        walls.view(),
        a.view(),
        f.view(),
        shvoveg.view(),
    );
    let result = ShadowingResult {
        vegsh: vegsh.into_pyarray(py).to_owned().into(),
        sh: sh.into_pyarray(py).to_owned().into(),
        vbshvegsh: vbshvegsh.into_pyarray(py).to_owned().into(),
        wallsh: wallsh.into_pyarray(py).to_owned().into(),
        wallsun: wallsun.into_pyarray(py).to_owned().into(),
        wallshve: wallshve.into_pyarray(py).to_owned().into(),
        facesh: facesh.into_pyarray(py).to_owned().into(),
        facesun: facesun.into_pyarray(py).to_owned().into(),
    };
    result
        .into_pyobject(py)
        .map(|bound_result| bound_result.into_py(py))
        .map_err(|e| e.into())
}
