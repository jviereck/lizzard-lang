fn main() {
    // `-undefined dynamic_lookup` on macOS: the extension resolves the
    // Python C API from the interpreter that loads it.
    pyo3_build_config::add_extension_module_link_args();
}
