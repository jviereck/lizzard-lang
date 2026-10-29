fn main() {
    // `-undefined dynamic_lookup` on macOS: the extension resolves the
    // Python C API from the interpreter that loads it.
    pyo3_build_config::add_extension_module_link_args();
    // `Py_GIL_DISABLED` and friends, for the few struct fields that differ
    // between the free-threaded and GIL builds.
    pyo3_build_config::use_pyo3_cfgs();
}
