//! Loading extension modules from dynamic libraries.

use std::path::{Path, PathBuf};

use lizzard_abi::{string_out, InitFn, VersionFn, ABI_VERSION};
use lizzard_vm::native::Module;

/// The library file of module `name` (`libgeom.so`, `libgeom.dylib`, `geom.dll`).
pub fn library_file(name: &str) -> String {
    format!("{}{name}{}", std::env::consts::DLL_PREFIX, std::env::consts::DLL_SUFFIX)
}

/// Loads module `name` from the first of `dirs` that has its library:
/// `None` if none has, an error if the library is not a usable extension.
pub fn extension(name: &str, dirs: &[PathBuf]) -> Result<Option<Module>, String> {
    let file = library_file(name);
    let Some(path) = dirs.iter().map(|d| d.join(&file)).find(|p| p.is_file()) else { return Ok(None) };
    load(name, &path).map(Some).map_err(|e| format!("ImportError: cannot load module `{name}` from {}: {e}", path.display()))
}

fn load(name: &str, path: &Path) -> Result<Module, String> {
    // SAFETY: loading a library runs its initializers: an extension is
    // trusted code, like any library the program links.
    let lib = unsafe { libloading::Library::new(path.as_os_str()) }.map_err(|e| e.to_string())?;
    // SAFETY: the symbols have these types in every Lizzard extension.
    let version = unsafe { lib.get::<VersionFn>(b"lizzard_ext_abi_version\0") }.map_err(|_| "not a Lizzard extension (no `lizzard_ext_abi_version`)".to_string())?;
    let v = unsafe { version() };
    if v != ABI_VERSION {
        return Err(format!("it was built for Lizzard ABI version {v}, this interpreter has version {ABI_VERSION}"));
    }
    let init = unsafe { lib.get::<InitFn>(b"lizzard_ext_init\0") }.map_err(|_| "not a Lizzard extension (no `lizzard_ext_init`)".to_string())?;
    let mut err = String::new();
    // SAFETY: the version matches, so `init` takes this ABI's function table.
    let raw = unsafe { init(lizzard_vm::ext::api(), string_out(&mut err)) };
    if raw.is_null() {
        return Err(err);
    }
    // SAFETY: built through this interpreter's API by `init`.
    let module = unsafe { lizzard_vm::ext::module_from_raw(raw) };
    // The library stays loaded: the interpreter keeps its functions.
    std::mem::forget(lib);
    if module.name != name {
        return Err(format!("it defines module `{}`", module.name));
    }
    Ok(module)
}
