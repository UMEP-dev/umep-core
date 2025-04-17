use pyo3::prelude::*;

mod shadowing;

#[pymodule]
fn rustalgos(py_module: &Bound<'_, PyModule>) -> PyResult<()> {
    // Register classes and functions
    // py_module.add_class::<common::Coord>()?;
    // py_module.add_function(wrap_pyfunction!(common::clipped_beta_wt, py_module)?)?;

    // Register submodules
    register_shadowing_module(py_module)?;
    py_module.add("__doc__", "UMEP algorithms implemented in Rust.")?;

    Ok(())
}

fn register_shadowing_module(py_module: &Bound<'_, PyModule>) -> PyResult<()> {
    let submodule = PyModule::new(py_module.py(), "shadowing")?;
    submodule.add("__doc__", "Shadow analysis.")?;
    // submodule.add_class::<data::DataEntry>()?;
    submodule.add_function(wrap_pyfunction!(
        shadowing::shadowingfunction_wallheight_25,
        &submodule
    )?)?;
    py_module.add_submodule(&submodule)?;
    Ok(())
}
