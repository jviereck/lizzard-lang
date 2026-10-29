//! Modules, definition files and types.
//!
//! Every imported module is backed by up to two files: a Lizzard overlay
//! (`<overlay root>/numpy/__init__.pyi`) and the module's own `.pyi`/`.py`
//! (the base). Files are parsed when the module is first imported; names are
//! resolved lazily, so only the declarations a program actually reaches are
//! evaluated. Declarations Lizzard cannot understand degrade to `Any`.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::rc::Rc;

use ruff_python_ast::{self as ast, Expr, Stmt};
use ruff_text_size::Ranged;

pub struct Env {
    pub overlay_roots: Vec<PathBuf>,
    pub search_roots: Vec<PathBuf>,
    pub version: (u32, u32),
    pub platform: String,
    /// Top-level standard library modules: imported as opaque (`Any`) until
    /// Lizzard bundles typeshed.
    pub stdlib: HashSet<String>,
    /// The installed version of a distribution, for overlay headers.
    pub pkg_version: PkgVersion,
}

pub type PkgVersion = Box<dyn Fn(&str) -> Option<String>>;

pub type FileId = usize;
pub type ModId = usize;
pub type ClassId = usize;
pub type FuncId = usize;
pub type TvId = usize;
pub type AliasId = usize;

pub struct SourceFile {
    pub path: String,
    pub text: String,
    line_starts: Vec<u32>,
}

impl SourceFile {
    fn new(path: String, text: String) -> SourceFile {
        let mut line_starts = vec![0];
        for (i, b) in text.bytes().enumerate() {
            if b == b'\n' {
                line_starts.push(i as u32 + 1);
            }
        }
        SourceFile { path, text, line_starts }
    }

    /// 1-based line and column of a byte offset.
    pub fn line_col(&self, off: u32) -> (u32, u32) {
        let line = match self.line_starts.binary_search(&off) {
            Ok(i) => i,
            Err(i) => i - 1,
        };
        let start = self.line_starts[line] as usize;
        let end = (off as usize).min(self.text.len());
        let col = self.text.get(start..end).map_or(0, |s| s.chars().count());
        (line as u32 + 1, col as u32 + 1)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Ty {
    Any,
    None,
    Bool,
    Int,
    Float,
    Str,
    Object,
    Tuple(Vec<Ty>),
    /// `tuple[T, ...]`
    TupleVar(Box<Ty>),
    Inst(ClassId, Vec<Ty>),
    /// The class object of a type: `type[T]`.
    Type(Box<Ty>),
    Module(ModId),
    Func(FuncRef),
    Var(TvId),
    /// An instance of a builtin exception class.
    Exc(String),
    /// `A | B`, flattened, without duplicates, two or more members.
    Union(Vec<Ty>),
    /// `Callable[[A, B], R]`; `None` parameters for `Callable[..., R]`.
    Callable(Option<Vec<Ty>>, Box<Ty>),
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum FuncRef {
    Def(FuncId),
    /// A method bound to its receiver.
    Bound(FuncId, Box<Ty>),
    Builtin(&'static str),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Special {
    Any,
    TypeVar,
    TypeAlias,
    Generic,
    Protocol,
    Overload,
    SelfTy,
    Literal,
    Union,
    Optional,
    Final,
    ClassVar,
    TypeChecking,
    Never,
    Tuple,
    Type,
    Callable,
}

#[derive(Clone, Debug)]
pub enum Sym {
    Any,
    Module(ModId),
    Class(ClassId),
    Func(FuncId),
    TypeVar(TvId),
    Alias(AliasId),
    Value(Ty),
    Special(Special),
    /// A builtin type name (`int`, `tuple`, ...) as the type it denotes.
    Builtin(Ty),
    /// A variable assigned at the top level of a Lizzard module.
    UserVar,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Variance {
    Inv,
    Co,
    Contra,
}

pub struct TvInfo {
    pub name: String,
    pub variance: Variance,
    /// Upper bound, checked when the variable is solved (PEP 695 bounds and
    /// `Self`; bounds of old-style `TypeVar`s are not evaluated yet).
    pub bound: Option<Ty>,
}

pub struct AliasInfo {
    pub name: String,
    pub params: Vec<TvId>,
    pub target: Ty,
}

#[derive(Clone, Debug)]
pub enum PKind {
    PosOnly,
    PosOrKw,
    VarPos,
    KwOnly,
    VarKw,
}

#[derive(Clone, Debug)]
pub struct Param {
    pub name: String,
    pub ty: Ty,
    pub kind: PKind,
    pub default: bool,
}

#[derive(Clone, Debug)]
pub struct Sig {
    pub params: Vec<Param>,
    pub ret: Ty,
    pub file: FileId,
    pub off: u32,
}

pub struct FuncInfo {
    pub name: String,
    pub m: ModId,
    pub owner: Option<ClassId>,
    defs: Vec<(Rc<ast::StmtFunctionDef>, FileId)>,
    sigs: Option<Rc<Vec<Sig>>>,
}

#[derive(Clone, Debug)]
pub enum Member {
    Method(FuncId),
    Property(FuncId),
    Attr(Ty),
}

struct ClassDecl {
    base: Option<(Rc<ast::StmtClassDef>, FileId)>,
    overlay: Option<(Rc<ast::StmtClassDef>, FileId)>,
}

pub struct ClassInfo {
    pub name: String,
    pub qualname: String,
    m: ModId,
    decl: Rc<ClassDecl>,
    params: Option<Vec<TvId>>,
    bases: Option<Vec<Ty>>,
    members: HashMap<String, Option<Member>>,
    /// `Self` in this class's methods: a variable bounded by the class.
    self_tv: Option<TvId>,
    mro: Option<Vec<ClassId>>,
    /// A base class is untyped (`Any`): unknown members are `Any`.
    any_base: bool,
    /// Declared with `Protocol` among its bases: other classes match it
    /// structurally.
    protocol: bool,
}

#[derive(Clone, Copy, Debug)]
pub struct DataclassOpts {
    pub frozen: bool,
    /// `order=True`: `<`, `<=`, `>`, `>=` compare the fields as tuples.
    pub order: bool,
    pub kw_only: bool,
    pub init: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MethodKind {
    Normal,
    Static,
    Class,
}

struct AssignDecl {
    ann: Option<Expr>,
    value: Option<Expr>,
}

#[derive(Clone)]
enum Decl {
    Class(Rc<ClassDecl>),
    Funcs(Rc<Vec<(Rc<ast::StmtFunctionDef>, FileId)>>),
    Assign(Rc<AssignDecl>),
    TypeAlias(Rc<ast::StmtTypeAlias>),
    /// `import a.b as c` (binds `c` to `a.b`); `import a.b` binds `a`.
    Module(String),
    /// `from module import name`
    From(String, String),
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum ModKind {
    Files,
    Typing,
    Builtins,
    Opaque,
}

pub struct Module {
    pub name: String,
    pub file: Option<FileId>,
    is_package: bool,
    kind: ModKind,
    /// A Lizzard module: its declarations are checked strictly.
    pub user: bool,
    decls: HashMap<String, Decl>,
    syms: HashMap<String, Sym>,
}

/// Type-evaluation scope: the module, type parameters in scope, `Self`.
pub struct Scope<'a> {
    pub m: ModId,
    pub tparams: &'a [(String, TvId)],
    pub self_ty: Option<&'a Ty>,
}

pub type Subst = HashMap<TvId, Ty>;

pub struct Db {
    pub env: Env,
    pub files: Vec<SourceFile>,
    pub modules: Vec<Module>,
    by_name: HashMap<String, Option<ModId>>,
    pub classes: Vec<ClassInfo>,
    class_keys: HashMap<(ModId, String), ClassId>,
    pub funcs: Vec<FuncInfo>,
    func_keys: HashMap<(ModId, Option<ClassId>, String), FuncId>,
    pub tvs: Vec<TvInfo>,
    tv_keys: HashMap<(ModId, String), TvId>,
    pub aliases: Vec<AliasInfo>,
    alias_keys: HashMap<(ModId, String), AliasId>,
    /// Non-generic aliases by target, with their qualified name and whether
    /// they come from a Lizzard module: `floating[_32Bit]` -> `numpy.float32`.
    reverse_alias: HashMap<Ty, (String, bool)>,
    aliases_resolved: HashSet<ModId>,
    pub diags: Vec<(FileId, u32, String)>,
    pub warnings: Vec<String>,
    /// Type variables of the function being checked (its class's type
    /// parameters and `Self`): they stand for one unknown type, so only
    /// themselves (or `Any`) are assignable to them.
    pub rigid: Vec<TvId>,
    /// The types of Lizzard modules' globals (from their top level).
    pub globals: HashMap<(ModId, String), Ty>,
    /// A property's accessors by its `@property` function and kind.
    accessors: HashMap<(FuncId, &'static str), FuncId>,
    protocol_depth: u32,
}

/// Lizzard's definitions for the builtins it types (see the file's header).
pub const BUILTINS_PYI: &str = include_str!("../stubs/builtins.pyi");
pub const BUILTINS_PATH: &str = "<lizzard>/builtins.pyi";

fn is_name(e: &Expr, s: &str) -> bool {
    match e {
        Expr::Name(n) => n.id.as_str() == s,
        Expr::Attribute(a) => a.attr.as_str() == s,
        _ => false,
    }
}

/// A property's accessor decorators (`@x.getter`, ...).
pub const ACCESSORS: [&str; 3] = ["getter", "setter", "deleter"];

/// Decorated `@name.which` (`@x.setter`).
pub fn is_accessor(d: &ast::StmtFunctionDef, name: &str, which: &str) -> bool {
    d.decorator_list.iter().any(|dec| matches!(&dec.expression, Expr::Attribute(a) if a.attr.as_str() == which && matches!(&*a.value, Expr::Name(n) if n.id.as_str() == name)))
}

fn decorated(d: &ast::StmtFunctionDef, name: &str) -> bool {
    d.decorator_list.iter().any(|dec| is_name(&dec.expression, name))
}

/// The elements of a subscript slice: `X[a, b]` -> `[a, b]`.
pub fn slice_elts(e: &Expr) -> Vec<&Expr> {
    match e {
        Expr::Tuple(t) => t.elts.iter().collect(),
        e => vec![e],
    }
}

fn int_tuple(e: &Expr) -> Option<Vec<i64>> {
    match e {
        Expr::Tuple(t) => t
            .elts
            .iter()
            .map(|x| match x {
                Expr::NumberLiteral(n) => match &n.value {
                    ast::Number::Int(i) => i.as_i64(),
                    _ => None,
                },
                _ => None,
            })
            .collect(),
        _ => None,
    }
}

fn version_ok(version: &str, spec: &str) -> bool {
    fn parts(s: &str) -> Vec<u64> {
        s.split('.').map(|p| p.chars().take_while(|c| c.is_ascii_digit()).collect::<String>().parse().unwrap_or(0)).collect()
    }
    let v = parts(version);
    spec.split(',').all(|c| {
        let c = c.trim();
        let (op, rest) = ["==", ">=", "<=", "!=", ">", "<"].iter().find_map(|op| c.strip_prefix(op).map(|r| (*op, r))).unwrap_or(("==", c));
        let w = parts(rest.trim());
        let n = v.len().max(w.len());
        let pad = |x: &Vec<u64>| (0..n).map(|i| x.get(i).copied().unwrap_or(0)).collect::<Vec<_>>();
        let ord = pad(&v).cmp(&pad(&w));
        match op {
            "==" => ord.is_eq(),
            "!=" => !ord.is_eq(),
            ">=" => ord.is_ge(),
            "<=" => ord.is_le(),
            ">" => ord.is_gt(),
            _ => ord.is_lt(),
        }
    })
}

impl Db {
    pub fn new(env: Env) -> Db {
        Db {
            env,
            files: Vec::new(),
            modules: Vec::new(),
            by_name: HashMap::new(),
            classes: Vec::new(),
            class_keys: HashMap::new(),
            funcs: Vec::new(),
            func_keys: HashMap::new(),
            tvs: Vec::new(),
            tv_keys: HashMap::new(),
            aliases: Vec::new(),
            alias_keys: HashMap::new(),
            reverse_alias: HashMap::new(),
            aliases_resolved: HashSet::new(),
            diags: Vec::new(),
            warnings: Vec::new(),
            rigid: Vec::new(),
            globals: HashMap::new(),
            accessors: HashMap::new(),
            protocol_depth: 0,
        }
    }

    fn add_file(&mut self, path: String, text: String) -> FileId {
        self.files.push(SourceFile::new(path, text));
        self.files.len() - 1
    }

    fn parse_file(&mut self, path: &Path) -> Option<(FileId, Vec<Stmt>)> {
        let text = std::fs::read_to_string(path).ok()?;
        match ruff_python_parser::parse_module(&text) {
            Ok(p) => {
                let body = p.into_syntax().body.into_iter().collect();
                Some((self.add_file(path.display().to_string(), text), body))
            }
            Err(e) => {
                self.warnings.push(format!("lizzard: cannot parse {}: {e}", path.display()));
                None
            }
        }
    }

    /// Report an error in a Lizzard module (declarations elsewhere degrade
    /// silently).
    pub fn err(&mut self, m: ModId, off: u32, msg: impl Into<String>) {
        if self.modules[m].user {
            if let Some(f) = self.modules[m].file {
                self.diags.push((f, off, msg.into()));
            }
        }
    }

    // ---- modules ----

    /// Parse a Lizzard module and register it under `name`.
    pub fn add_user_module(&mut self, name: &str, path: &str, text: &str) -> Result<(ModId, Vec<Stmt>), String> {
        let parsed = ruff_python_parser::parse_module(text);
        let file = self.add_file(path.to_string(), text.to_string());
        let body = match parsed {
            Ok(p) => p.into_syntax().body.into_iter().collect::<Vec<_>>(),
            Err(e) => {
                let (l, c) = self.files[file].line_col(e.location.start().into());
                return Err(format!("{path}:{l}:{c}: error: {}", e.error));
            }
        };
        let m = self.modules.len();
        self.modules.push(Module {
            name: name.to_string(),
            file: Some(file),
            is_package: false,
            kind: ModKind::Files,
            user: true,
            decls: HashMap::new(),
            syms: HashMap::new(),
        });
        self.by_name.insert(name.to_string(), Some(m));
        let decls = self.collect(m, file, &body, false);
        self.modules[m].decls = decls;
        Ok((m, body))
    }

    pub fn load_module(&mut self, name: &str) -> Option<ModId> {
        if let Some(r) = self.by_name.get(name) {
            return *r;
        }
        let top = name.split('.').next().unwrap_or(name);
        let kind = match name {
            "typing" | "typing_extensions" | "collections.abc" => Some(ModKind::Typing),
            "builtins" => Some(ModKind::Builtins),
            _ if self.env.stdlib.contains(top) => Some(ModKind::Opaque),
            _ => None,
        };
        if kind == Some(ModKind::Builtins) {
            let body: Vec<Stmt> = ruff_python_parser::parse_module(BUILTINS_PYI).expect("bundled builtins.pyi parses").into_syntax().body.into_iter().collect();
            let f = self.add_file(BUILTINS_PATH.to_string(), BUILTINS_PYI.to_string());
            let m = self.push_module(name, Some(f), false, ModKind::Builtins);
            let decls = self.collect(m, f, &body, false);
            self.modules[m].decls = decls;
            return Some(m);
        }
        if let Some(kind) = kind {
            return Some(self.push_module(name, None, false, kind));
        }
        let rel: PathBuf = name.split('.').collect();
        let find = |roots: &[PathBuf], exts: &[&str]| -> Option<(PathBuf, bool)> {
            for root in roots {
                for ext in exts {
                    let pkg = root.join(&rel).join(format!("__init__.{ext}"));
                    if pkg.is_file() {
                        return Some((pkg, true));
                    }
                    let file = root.join(&rel).with_extension(ext);
                    if file.is_file() {
                        return Some((file, false));
                    }
                }
            }
            None
        };
        let base = find(&self.env.search_roots, &["pyi", "py"]);
        let mut overlay = find(&self.env.overlay_roots, &["pyi"]);
        if let Some((path, _)) = &overlay {
            if let Some(spec) = std::fs::read_to_string(path).ok().and_then(|t| {
                t.lines().take(5).find_map(|l| l.trim().strip_prefix("# lizzard: overlay").map(|s| s.trim().to_string()))
            }) {
                let pkg: String = spec.chars().take_while(|c| c.is_alphanumeric() || *c == '_' || *c == '-').collect();
                let constraint = spec[pkg.len()..].trim();
                match (self.env.pkg_version)(&pkg) {
                    Some(v) if version_ok(&v, constraint) => {}
                    v => {
                        self.warnings.push(format!(
                            "lizzard: ignoring {} (made for {pkg}{constraint}, installed: {})",
                            path.display(),
                            v.as_deref().unwrap_or("none")
                        ));
                        overlay = None;
                    }
                }
            }
        }
        if base.is_none() && overlay.is_none() {
            self.by_name.insert(name.to_string(), None);
            return None;
        }
        let is_package = base.as_ref().or(overlay.as_ref()).is_some_and(|(_, p)| *p);
        let base = base.and_then(|(p, _)| self.parse_file(&p));
        let overlay = overlay.and_then(|(p, _)| self.parse_file(&p));
        let m = self.push_module(name, base.as_ref().or(overlay.as_ref()).map(|(f, _)| *f), is_package, ModKind::Files);
        let mut decls = match &base {
            Some((f, body)) => self.collect(m, *f, body, false),
            None => HashMap::new(),
        };
        if let Some((f, body)) = &overlay {
            for (name, d) in self.collect(m, *f, body, true) {
                let merged = match (&d, decls.get(&name)) {
                    (Decl::Class(o), Some(Decl::Class(b))) => {
                        Decl::Class(Rc::new(ClassDecl { base: b.base.clone(), overlay: o.overlay.clone() }))
                    }
                    _ => d,
                };
                decls.insert(name, merged);
            }
        }
        self.modules[m].decls = decls;
        Some(m)
    }

    fn push_module(&mut self, name: &str, file: Option<FileId>, is_package: bool, kind: ModKind) -> ModId {
        let m = self.modules.len();
        self.modules.push(Module {
            name: name.to_string(),
            file,
            is_package,
            kind,
            user: false,
            decls: HashMap::new(),
            syms: HashMap::new(),
        });
        self.by_name.insert(name.to_string(), Some(m));
        m
    }

    fn absolute(&self, m: ModId, level: u32, module: Option<&str>) -> String {
        if level == 0 {
            return module.unwrap_or("").to_string();
        }
        let md = &self.modules[m];
        let mut parts: Vec<&str> = md.name.split('.').collect();
        if !md.is_package {
            parts.pop();
        }
        for _ in 1..level {
            parts.pop();
        }
        let mut s = parts.join(".");
        if let Some(sub) = module {
            if !s.is_empty() {
                s.push('.');
            }
            s.push_str(sub);
        }
        s
    }

    /// Statements that apply on this interpreter: `if sys.version_info ...`,
    /// `if sys.platform ...` and `if TYPE_CHECKING` are evaluated; other
    /// conditions keep every branch.
    fn flatten<'a>(&self, stmts: &'a [Stmt], out: &mut Vec<&'a Stmt>) {
        for s in stmts {
            match s {
                Stmt::If(i) => {
                    let mut branches: Vec<(Option<&Expr>, &[Stmt])> = vec![(Some(&i.test), &i.body)];
                    for c in &i.elif_else_clauses {
                        branches.push((c.test.as_ref(), &c.body));
                    }
                    for (test, body) in branches {
                        match test.map(|t| self.cond(t)) {
                            // A true condition or the `else`: this branch, and no later one.
                            Some(Some(true)) | None => {
                                self.flatten(body, out);
                                break;
                            }
                            Some(Some(false)) => {}
                            Some(None) => self.flatten(body, out),
                        }
                    }
                }
                Stmt::Try(t) => self.flatten(&t.body, out),
                s => out.push(s),
            }
        }
    }

    fn cond(&self, e: &Expr) -> Option<bool> {
        match e {
            Expr::Name(_) | Expr::Attribute(_) if is_name(e, "TYPE_CHECKING") => Some(true),
            Expr::UnaryOp(u) if u.op == ast::UnaryOp::Not => self.cond(&u.operand).map(|b| !b),
            Expr::BoolOp(b) => {
                let vals: Vec<Option<bool>> = b.values.iter().map(|v| self.cond(v)).collect();
                match b.op {
                    ast::BoolOp::And if vals.contains(&Some(false)) => Some(false),
                    ast::BoolOp::Or if vals.contains(&Some(true)) => Some(true),
                    _ => vals.into_iter().collect::<Option<Vec<_>>>().map(|v| v[0]),
                }
            }
            Expr::Compare(c) if c.ops.len() == 1 => {
                let left = &c.operands[0];
                let right = &c.operands[1];
                if is_name(left, "version_info") {
                    let want = int_tuple(right)?;
                    let have = [self.env.version.0 as i64, self.env.version.1 as i64];
                    let n = want.len().min(2);
                    let ord = have[..n].cmp(&want[..n]);
                    return Some(match c.ops[0] {
                        ast::CmpOp::Lt => ord.is_lt(),
                        ast::CmpOp::LtE => ord.is_le(),
                        ast::CmpOp::Gt => ord.is_gt(),
                        ast::CmpOp::GtE => ord.is_ge(),
                        ast::CmpOp::Eq => ord.is_eq(),
                        ast::CmpOp::NotEq => !ord.is_eq(),
                        _ => return None,
                    });
                }
                if is_name(left, "platform") {
                    if let Expr::StringLiteral(s) = right {
                        let eq = s.value.to_str() == self.env.platform;
                        return match c.ops[0] {
                            ast::CmpOp::Eq => Some(eq),
                            ast::CmpOp::NotEq => Some(!eq),
                            _ => None,
                        };
                    }
                }
                None
            }
            _ => None,
        }
    }

    fn collect(&mut self, m: ModId, file: FileId, body: &[Stmt], overlay: bool) -> HashMap<String, Decl> {
        let mut flat = Vec::new();
        self.flatten(body, &mut flat);
        let mut decls: HashMap<String, Decl> = HashMap::new();
        for s in flat {
            match s {
                Stmt::FunctionDef(f) => {
                    let name = f.name.as_str().to_string();
                    let mut defs = match decls.get(&name) {
                        Some(Decl::Funcs(d)) => (**d).clone(),
                        _ => Vec::new(),
                    };
                    defs.push((Rc::new(f.clone()), file));
                    decls.insert(name, Decl::Funcs(Rc::new(defs)));
                }
                Stmt::ClassDef(c) => {
                    let side = Some((Rc::new(c.clone()), file));
                    let decl = if overlay { ClassDecl { base: None, overlay: side } } else { ClassDecl { base: side, overlay: None } };
                    decls.insert(c.name.as_str().to_string(), Decl::Class(Rc::new(decl)));
                }
                Stmt::AnnAssign(a) => {
                    if let Expr::Name(n) = &*a.target {
                        let d = AssignDecl { ann: Some((*a.annotation).clone()), value: a.value.as_deref().cloned() };
                        decls.insert(n.id.as_str().to_string(), Decl::Assign(Rc::new(d)));
                    }
                }
                Stmt::Assign(a) => {
                    if let [Expr::Name(n)] = a.targets.as_slice() {
                        let d = AssignDecl { ann: None, value: Some((*a.value).clone()) };
                        decls.insert(n.id.as_str().to_string(), Decl::Assign(Rc::new(d)));
                    }
                }
                Stmt::TypeAlias(t) => {
                    if let Expr::Name(n) = &*t.name {
                        decls.insert(n.id.as_str().to_string(), Decl::TypeAlias(Rc::new(t.clone())));
                    }
                }
                Stmt::Import(i) => {
                    for a in &i.names {
                        let full = a.name.as_str();
                        match &a.asname {
                            Some(asn) => decls.insert(asn.as_str().to_string(), Decl::Module(full.to_string())),
                            None => {
                                let top = full.split('.').next().unwrap_or(full);
                                decls.insert(top.to_string(), Decl::Module(top.to_string()))
                            }
                        };
                    }
                }
                Stmt::ImportFrom(i) => {
                    let module = self.absolute(m, i.level, i.module.as_ref().map(|x| x.as_str()));
                    for a in &i.names {
                        if a.name.as_str() == "*" {
                            continue;
                        }
                        let bind = a.asname.as_ref().unwrap_or(&a.name).as_str().to_string();
                        decls.insert(bind, Decl::From(module.clone(), a.name.as_str().to_string()));
                    }
                }
                _ => {}
            }
        }
        decls
    }

    // ---- symbols ----

    /// A name in module `m`'s namespace, falling back to builtins.
    pub fn lookup_name(&mut self, m: ModId, name: &str) -> Sym {
        match self.lookup_in(m, name) {
            Some(s) => s,
            None => self.builtin(name),
        }
    }

    /// A builtin name: the checker's own (`int`, exceptions) or one declared
    /// in the bundled builtins.pyi; `Any` if neither.
    pub fn builtin(&mut self, name: &str) -> Sym {
        match self.load_module("builtins") {
            Some(b) => self.lookup_in(b, name).unwrap_or(Sym::Any),
            None => builtin_sym(name),
        }
    }

    /// Whether builtins.pyi declares `name` (not the checker's own names).
    pub fn is_declared_builtin(&mut self, name: &str) -> bool {
        matches!(builtin_sym(name), Sym::Any) && self.load_module("builtins").is_some_and(|b| self.modules[b].decls.contains_key(name))
    }

    /// The class builtins.pyi declares as `name` (also for `str` and
    /// `tuple`, which the checker otherwise types structurally).
    pub fn builtin_class(&mut self, name: &str) -> Option<ClassId> {
        let b = self.load_module("builtins")?;
        match self.modules[b].decls.get(name)?.clone() {
            Decl::Class(cd) => Some(self.class_for(b, name, cd)),
            _ => None,
        }
    }

    /// The class type through which a structural type finds its methods and
    /// base classes: `str` and tuples.
    pub fn nominal(&mut self, t: &Ty) -> Option<Ty> {
        match t {
            Ty::Str => Some(Ty::Inst(self.builtin_class("str")?, Vec::new())),
            Ty::TupleVar(x) => Some(Ty::Inst(self.builtin_class("tuple")?, vec![(**x).clone()])),
            Ty::Tuple(v) => {
                let elem = match v.first() {
                    None => Ty::Any,
                    Some(f) if v.iter().all(|x| x == f) => f.clone(),
                    Some(_) => Ty::Object,
                };
                Some(Ty::Inst(self.builtin_class("tuple")?, vec![elem]))
            }
            _ => None,
        }
    }

    /// A type variable's bound stands in for it where a concrete type is
    /// needed (attributes of `self`).
    pub fn upper(&self, t: &Ty) -> Ty {
        match t {
            Ty::Var(v) => self.tvs[*v].bound.clone().unwrap_or(Ty::Any),
            t => t.clone(),
        }
    }

    /// A name defined by module `m` itself (no builtins fallback).
    pub fn lookup_in(&mut self, m: ModId, name: &str) -> Option<Sym> {
        if let Some(s) = self.modules[m].syms.get(name) {
            return Some(s.clone());
        }
        let s = match self.modules[m].kind {
            ModKind::Typing => match name {
                "Iterable" | "Iterator" | "Sequence" | "Generator" | "SupportsAbs" => self.builtin(name),
                "List" | "Dict" | "Set" | "FrozenSet" => self.builtin(&name.to_lowercase()),
                _ => special_sym(name),
            },
            ModKind::Builtins if !matches!(builtin_sym(name), Sym::Any) || !self.modules[m].decls.contains_key(name) => builtin_sym(name),
            ModKind::Opaque => Sym::Any,
            ModKind::Files | ModKind::Builtins => {
                let d = self.modules[m].decls.get(name)?.clone();
                // Placeholder against cycles.
                self.modules[m].syms.insert(name.to_string(), Sym::Any);
                self.resolve_decl(m, name, d)
            }
        };
        self.modules[m].syms.insert(name.to_string(), s.clone());
        Some(s)
    }

    /// `module.attr`: a name in the module, else a submodule.
    pub fn module_attr(&mut self, m: ModId, attr: &str) -> Option<Sym> {
        if let Some(s) = self.lookup_in(m, attr) {
            return Some(s);
        }
        let sub = format!("{}.{attr}", self.modules[m].name);
        self.load_module(&sub).map(Sym::Module)
    }

    pub fn is_user_var(&self, m: ModId, name: &str) -> bool {
        self.modules[m].user && matches!(self.modules[m].decls.get(name), Some(Decl::Assign(a)) if a.ann.is_none() || a.value.is_some())
    }

    fn resolve_decl(&mut self, m: ModId, name: &str, d: Decl) -> Sym {
        match d {
            Decl::Module(path) => self.load_module(&path).map_or(Sym::Any, Sym::Module),
            Decl::From(module, item) => match self.load_module(&module) {
                Some(mm) => self.module_attr(mm, &item).unwrap_or(Sym::Any),
                None => Sym::Any,
            },
            Decl::Class(cd) => Sym::Class(self.class_for(m, name, cd)),
            Decl::Funcs(defs) => Sym::Func(self.func_for(m, None, name, (*defs).clone())),
            Decl::Assign(a) => self.resolve_assign(m, name, &a),
            Decl::TypeAlias(t) => {
                let tps = self.type_params(m, t.type_params.as_deref());
                Sym::Alias(self.make_alias(m, name, &t.value, Some(tps)))
            }
        }
    }

    fn resolve_assign(&mut self, m: ModId, name: &str, a: &AssignDecl) -> Sym {
        if let Some(Expr::Call(c)) = &a.value {
            if matches!(self.sym_of_expr(m, &c.func), Some(Sym::Special(Special::TypeVar))) {
                let kw = |k: &str| {
                    c.arguments.keywords.iter().any(|kw| {
                        kw.arg.as_ref().is_some_and(|a| a.as_str() == k) && matches!(kw.value, Expr::BooleanLiteral(ref b) if b.value)
                    })
                };
                let variance = if kw("covariant") {
                    Variance::Co
                } else if kw("contravariant") {
                    Variance::Contra
                } else {
                    Variance::Inv
                };
                let key = (m, name.to_string());
                if let Some(&tv) = self.tv_keys.get(&key) {
                    return Sym::TypeVar(tv);
                }
                let tv = self.new_tv(name, variance);
                self.tv_keys.insert(key, tv);
                return Sym::TypeVar(tv);
            }
        }
        if let Some(ann) = &a.ann {
            if matches!(self.sym_of_expr(m, ann), Some(Sym::Special(Special::TypeAlias))) {
                if let Some(v) = &a.value {
                    return Sym::Alias(self.make_alias(m, name, v, None));
                }
            }
            if self.modules[m].user {
                return Sym::UserVar;
            }
            let ty = self.eval_type(ann, &Scope { m, tparams: &[], self_ty: None });
            return Sym::Value(ty);
        }
        if self.modules[m].user {
            return Sym::UserVar;
        }
        match &a.value {
            // `X = Y` / `X = Y[...]` in a stub: an implicit alias.
            Some(v @ (Expr::Name(_) | Expr::Attribute(_) | Expr::Subscript(_))) => Sym::Alias(self.make_alias(m, name, v, None)),
            _ => Sym::Value(Ty::Any),
        }
    }

    pub fn new_tv(&mut self, name: &str, variance: Variance) -> TvId {
        self.tvs.push(TvInfo { name: name.to_string(), variance, bound: None });
        self.tvs.len() - 1
    }

    /// PEP 695 type parameters, with their bounds.
    fn type_params(&mut self, m: ModId, tp: Option<&ast::TypeParams>) -> Vec<(String, TvId)> {
        let mut out = Vec::new();
        for p in tp.iter().flat_map(|t| t.type_params.iter()) {
            if let ast::TypeParam::TypeVar(v) = p {
                let id = self.new_tv(v.name.as_str(), Variance::Inv);
                out.push((v.name.as_str().to_string(), id));
            }
        }
        for p in tp.iter().flat_map(|t| t.type_params.iter()) {
            if let ast::TypeParam::TypeVar(v) = p {
                if let Some(b) = &v.bound {
                    let ty = self.eval_type(b, &Scope { m, tparams: &out, self_ty: None });
                    let id = out.iter().find(|(n, _)| n == v.name.as_str()).unwrap().1;
                    self.tvs[id].bound = Some(ty);
                }
            }
        }
        out
    }

    /// The `Self` type variable of class `c`.
    pub fn self_tv(&mut self, c: ClassId) -> TvId {
        if let Some(t) = self.classes[c].self_tv {
            return t;
        }
        let ps = self.class_params(c);
        let tv = self.new_tv("Self", Variance::Inv);
        self.tvs[tv].bound = Some(Ty::Inst(c, ps.into_iter().map(Ty::Var).collect()));
        self.classes[c].self_tv = Some(tv);
        tv
    }

    fn make_alias(&mut self, m: ModId, name: &str, value: &Expr, tparams: Option<Vec<(String, TvId)>>) -> AliasId {
        let key = (m, name.to_string());
        if let Some(&a) = self.alias_keys.get(&key) {
            return a;
        }
        let tps = tparams.clone().unwrap_or_default();
        let target = self.eval_type(value, &Scope { m, tparams: &tps, self_ty: None });
        let params = match tparams {
            Some(t) => t.into_iter().map(|(_, id)| id).collect(),
            None => {
                let mut v = Vec::new();
                collect_vars(&target, &mut v);
                v
            }
        };
        // Only exact targets name a runtime class (`longdouble` is an alias
        // of `floating[Any]` on some platforms).
        if params.is_empty() && matches!(target, Ty::Inst(..)) && !has_any(&target) {
            let qual = format!("{}.{name}", self.modules[m].name);
            self.reverse_alias.entry(target.clone()).or_insert((qual, self.modules[m].user));
        }
        self.aliases.push(AliasInfo { name: name.to_string(), params, target });
        let id = self.aliases.len() - 1;
        self.alias_keys.insert(key, id);
        id
    }

    /// The symbol a `Name` / `a.b.c` expression refers to.
    pub fn sym_of_expr(&mut self, m: ModId, e: &Expr) -> Option<Sym> {
        match e {
            Expr::Name(n) => Some(self.lookup_name(m, n.id.as_str())),
            Expr::Attribute(a) => match self.sym_of_expr(m, &a.value)? {
                Sym::Module(mm) => self.module_attr(mm, a.attr.as_str()),
                Sym::Any => Some(Sym::Any),
                _ => None,
            },
            _ => None,
        }
    }

    // ---- classes and functions ----

    fn class_for(&mut self, m: ModId, name: &str, decl: Rc<ClassDecl>) -> ClassId {
        let key = (m, name.to_string());
        if let Some(&c) = self.class_keys.get(&key) {
            return c;
        }
        self.classes.push(ClassInfo {
            name: name.to_string(),
            qualname: format!("{}.{name}", self.modules[m].name),
            m,
            decl,
            params: None,
            bases: None,
            members: HashMap::new(),
            self_tv: None,
            mro: None,
            any_base: false,
            protocol: false,
        });
        let c = self.classes.len() - 1;
        self.class_keys.insert(key, c);
        c
    }

    fn header(&self, c: ClassId) -> Rc<ast::StmtClassDef> {
        let d = &self.classes[c].decl;
        d.base.as_ref().or(d.overlay.as_ref()).map(|(s, _)| s.clone()).expect("class without declaration")
    }

    pub fn class_params(&mut self, c: ClassId) -> Vec<TvId> {
        if let Some(p) = &self.classes[c].params {
            return p.clone();
        }
        self.classes[c].params = Some(Vec::new());
        let m = self.classes[c].m;
        let stmt = self.header(c);
        let mut params = Vec::new();
        if let Some(tp) = &stmt.type_params {
            for p in &tp.type_params {
                if let ast::TypeParam::TypeVar(v) = p {
                    params.push(self.new_tv(v.name.as_str(), Variance::Inv));
                }
            }
        } else if let Some(args) = &stmt.arguments {
            let mut explicit = None;
            let mut vars = Vec::new();
            for base in args.args.iter() {
                if let Expr::Subscript(s) = base {
                    if matches!(self.sym_of_expr(m, &s.value), Some(Sym::Special(Special::Generic | Special::Protocol))) {
                        let mut ps = Vec::new();
                        for e in slice_elts(&s.slice) {
                            if let Some(Sym::TypeVar(tv)) = self.sym_of_expr(m, e) {
                                ps.push(tv);
                            }
                        }
                        explicit = Some(ps);
                        continue;
                    }
                }
                let ty = self.eval_type(base, &Scope { m, tparams: &[], self_ty: None });
                collect_vars(&ty, &mut vars);
            }
            params = explicit.unwrap_or(vars);
        }
        self.classes[c].params = Some(params.clone());
        params
    }

    pub fn class_bases(&mut self, c: ClassId) -> Vec<Ty> {
        if let Some(b) = &self.classes[c].bases {
            return b.clone();
        }
        self.classes[c].bases = Some(Vec::new());
        let m = self.classes[c].m;
        let stmt = self.header(c);
        let mut bases = Vec::new();
        if let Some(args) = &stmt.arguments {
            for base in args.args.iter() {
                let target = match base {
                    Expr::Subscript(s) => &s.value,
                    e => e,
                };
                match self.sym_of_expr(m, target) {
                    Some(Sym::Special(Special::Protocol)) => {
                        self.classes[c].protocol = true;
                        continue;
                    }
                    Some(Sym::Special(_)) => continue,
                    _ => {}
                }
                let tps = self.class_tparams(c);
                let ty = self.eval_type(base, &Scope { m, tparams: &tps, self_ty: None });
                if matches!(ty, Ty::Any) {
                    self.classes[c].any_base = true;
                } else {
                    bases.push(ty);
                }
            }
        }
        self.classes[c].bases = Some(bases.clone());
        bases
    }

    /// The PEP 695 type parameters of class `c`, by name (in scope in its
    /// body); old-style `TypeVar`s are module names and need no scope.
    pub fn class_tparams(&mut self, c: ClassId) -> Vec<(String, TvId)> {
        if self.header(c).type_params.is_none() {
            return Vec::new();
        }
        self.class_params(c).into_iter().map(|tv| (self.tvs[tv].name.clone(), tv)).collect()
    }

    /// Whether `c` or a class in its MRO has an untyped base.
    pub fn has_any_base(&mut self, c: ClassId) -> bool {
        self.mro(c).iter().any(|k| self.classes[*k].any_base)
    }

    pub fn is_builtin_class(&self, c: ClassId) -> bool {
        self.modules[self.classes[c].m].kind == ModKind::Builtins
    }

    /// The number of `def`s of a function (more than one is a redefinition,
    /// unless they are overloads).
    pub fn def_count(&self, f: FuncId) -> usize {
        self.funcs[f].defs.len()
    }

    pub fn is_user_class(&self, c: ClassId) -> bool {
        self.modules[self.classes[c].m].user
    }

    /// The class's own statement in a Lizzard module.
    pub fn user_class_def(&self, c: ClassId) -> Option<Rc<ast::StmtClassDef>> {
        self.is_user_class(c).then(|| self.header(c))
    }

    /// The method resolution order (C3, as CPython computes it) over the
    /// class bases that are classes.
    pub fn mro(&mut self, c: ClassId) -> Vec<ClassId> {
        if let Some(m) = &self.classes[c].mro {
            return m.clone();
        }
        self.classes[c].mro = Some(vec![c]);
        let bases: Vec<ClassId> = self.class_bases(c).iter().filter_map(|b| if let Ty::Inst(bc, _) = b { Some(*bc) } else { None }).collect();
        let mut seqs: Vec<Vec<ClassId>> = bases.iter().map(|b| self.mro(*b)).collect();
        seqs.push(bases.clone());
        let mut out = vec![c];
        let merged = loop {
            seqs.retain(|s| !s.is_empty());
            if seqs.is_empty() {
                break true;
            }
            let head = seqs.iter().map(|s| s[0]).find(|h| !seqs.iter().any(|s| s[1..].contains(h)));
            let Some(h) = head else { break false };
            out.push(h);
            for s in seqs.iter_mut() {
                if s[0] == h {
                    s.remove(0);
                }
            }
        };
        if !merged {
            // No consistent order (CPython raises TypeError): depth-first.
            out = vec![c];
            for b in bases {
                for x in self.mro(b) {
                    if !out.contains(&x) {
                        out.push(x);
                    }
                }
            }
        }
        self.classes[c].mro = Some(out.clone());
        out
    }

    /// Whether `c` is a `Protocol` class (declared with `Protocol` among its
    /// bases, or a protocol's subclass that adds `Protocol` again).
    pub fn is_protocol(&mut self, c: ClassId) -> bool {
        self.class_bases(c);
        self.classes[c].protocol
    }

    /// The members a protocol declares: its own and its protocol bases'.
    fn protocol_members(&mut self, p: ClassId) -> Vec<String> {
        let mut names = Vec::new();
        for k in self.mro(p) {
            if !self.is_protocol(k) {
                continue;
            }
            let decl = self.classes[k].decl.clone();
            for (stmt, _) in decl.overlay.iter().chain(decl.base.iter()) {
                let mut flat = Vec::new();
                self.flatten(&stmt.body, &mut flat);
                for st in flat {
                    let n = match st {
                        Stmt::FunctionDef(f) => f.name.to_string(),
                        Stmt::AnnAssign(a) => match &*a.target {
                            Expr::Name(n) => n.id.to_string(),
                            _ => continue,
                        },
                        _ => continue,
                    };
                    if !names.contains(&n) && !matches!(n.as_str(), "__init__" | "__slots__" | "__class_getitem__") {
                        names.push(n);
                    }
                }
            }
        }
        names
    }

    /// `src` (an instance of `c`) as an instance of protocol `p`: it has
    /// every member `p` declares, with compatible types.
    fn structural_assign(&mut self, c: ClassId, src: &Ty, p: ClassId, dst: &Ty, s: &mut Subst, any: &mut bool) -> bool {
        for name in self.protocol_members(p) {
            let Some((_, want)) = self.find_member(p, &name) else { return false };
            let Some((cowner, got)) = self.find_member(c, &name) else { return false };
            let ok = match (want, got) {
                (Member::Method(fp), Member::Method(fc)) => {
                    let (w, g) = (self.callable_of(&Ty::Func(FuncRef::Bound(fp, Box::new(dst.clone())))), self.callable_of(&Ty::Func(FuncRef::Bound(fc, Box::new(src.clone())))));
                    match (w, g) {
                        (Some(w), Some(g)) => self.assign(&g, &w, s, any),
                        // Overloaded: present is enough.
                        _ => true,
                    }
                }
                (want @ (Member::Attr(_) | Member::Property(_)), got) => {
                    let tp = match want {
                        Member::Property(f) => self.sigs(f).first().map_or(Ty::Any, |sg| sg.ret.clone()),
                        Member::Attr(t) => t,
                        Member::Method(_) => unreachable!(),
                    };
                    let pmap: Subst = self.class_params(p).into_iter().zip(match dst { Ty::Inst(_, a) => a.clone(), _ => Vec::new() }).collect();
                    let tp = deep_subst(&tp, &pmap);
                    let tc = match got {
                        Member::Attr(t) => {
                            let cmap: Subst = self.class_params(cowner).into_iter().zip(match src { Ty::Inst(_, a) => a.clone(), _ => Vec::new() }).collect();
                            deep_subst(&t, &cmap)
                        }
                        Member::Property(f) => self.sigs(f).first().map_or(Ty::Any, |sg| sg.ret.clone()),
                        Member::Method(_) => return false,
                    };
                    self.assign(&tc, &tp, s, any)
                }
                _ => false,
            };
            if !ok {
                return false;
            }
        }
        true
    }

    /// The builtin exception class a class derives from, if any.
    pub fn exc_base(&mut self, c: ClassId) -> Option<String> {
        for k in self.mro(c) {
            for b in self.class_bases(k) {
                if let Ty::Exc(n) = b {
                    return Some(n);
                }
            }
        }
        None
    }

    /// An accessor (`getter`, `setter`, `deleter`) of the property whose
    /// `@property` function is `f`.
    pub fn property_accessor(&self, f: FuncId, kind: &str) -> Option<FuncId> {
        ACCESSORS.iter().find(|k| **k == kind).and_then(|k| self.accessors.get(&(f, *k)).copied())
    }

    /// The setter of property `f`.
    pub fn property_setter(&self, f: FuncId) -> Option<FuncId> {
        self.property_accessor(f, "setter")
    }

    /// The function that reads property `f`: its `.getter` if it has one.
    pub fn property_getter(&self, f: FuncId) -> FuncId {
        self.property_accessor(f, "getter").unwrap_or(f)
    }

    /// The options of `@dataclass` (from `dataclasses`) on class `c`, if
    /// it has it.
    pub fn dataclass_opts(&mut self, c: ClassId) -> Option<DataclassOpts> {
        let stmt = self.header(c);
        let m = self.classes[c].m;
        for dec in &stmt.decorator_list {
            let (target, kws) = match &dec.expression {
                Expr::Call(call) => (&*call.func, Some(&call.arguments.keywords)),
                e => (e, None),
            };
            // `dataclass` imported from `dataclasses` (an opaque module), not
            // something the module defines itself.
            if !is_name(target, "dataclass") || !matches!(self.sym_of_expr(m, target), Some(Sym::Any)) {
                continue;
            }
            let flag = |name: &str, default: bool| {
                kws.and_then(|k| k.iter().find(|k| k.arg.as_ref().is_some_and(|a| a.as_str() == name)))
                    .map_or(default, |k| matches!(k.value, Expr::BooleanLiteral(ref b) if b.value))
            };
            return Some(DataclassOpts { frozen: flag("frozen", false), order: flag("order", false), kw_only: flag("kw_only", false), init: flag("init", true) });
        }
        None
    }

    /// A dataclass's fields in `__init__` order (inherited ones first): name,
    /// type, whether it has a default, keyword-only.
    pub fn dataclass_fields(&mut self, c: ClassId) -> Vec<(String, Ty, bool, bool)> {
        let mut fields: Vec<(String, Ty, bool, bool)> = Vec::new();
        let cps = self.class_params(c);
        for k in self.mro(c).into_iter().rev() {
            let Some(opts) = self.dataclass_opts(k) else { continue };
            // `k`'s type parameters as `c` instantiates them.
            let kps = self.class_params(k);
            let mut s = Subst::new();
            self.assign(&Ty::Inst(c, cps.iter().map(|p| Ty::Var(*p)).collect()), &Ty::Inst(k, kps.iter().map(|p| Ty::Var(*p)).collect()), &mut s, &mut false);
            let stmt = self.header(k);
            let m = self.classes[k].m;
            let tps = self.class_tparams(k);
            let self_ty = Ty::Var(self.self_tv(k));
            let mut flat = Vec::new();
            self.flatten(&stmt.body, &mut flat);
            for st in flat {
                let Stmt::AnnAssign(a) = st else { continue };
                let Expr::Name(n) = &*a.target else { continue };
                let head = match &*a.annotation {
                    Expr::Subscript(sub) => &*sub.value,
                    e => e,
                };
                if is_name(head, "ClassVar") {
                    continue;
                }
                let ty = self.eval_type(&a.annotation, &Scope { m, tparams: &tps, self_ty: Some(&self_ty) });
                let ty = deep_subst(&ty, &s);
                let default = match a.value.as_deref() {
                    None => false,
                    // `field(...)` has a default only with `default=` or
                    // `default_factory=`.
                    Some(Expr::Call(call)) if is_name(&call.func, "field") => {
                        call.arguments.keywords.iter().any(|kw| kw.arg.as_ref().is_some_and(|a| matches!(a.as_str(), "default" | "default_factory")))
                    }
                    Some(_) => true,
                };
                let name = n.id.to_string();
                match fields.iter_mut().find(|f| f.0 == name) {
                    Some(f) => *f = (name, ty, default, opts.kw_only),
                    None => {
                        if !default && !opts.kw_only && fields.iter().any(|f| f.2 && !f.3) {
                            self.err(m, a.range().start().into(), format!("dataclass field `{name}` without a default cannot follow fields with defaults"));
                        }
                        fields.push((name, ty, default, opts.kw_only));
                    }
                }
            }
        }
        fields
    }

    /// Whether `c` or a base is a frozen dataclass.
    pub fn is_frozen(&mut self, c: ClassId) -> bool {
        self.mro(c).into_iter().any(|k| self.dataclass_opts(k).is_some_and(|o| o.frozen))
    }

    pub fn method_kind(&self, f: FuncId) -> MethodKind {
        let defs = &self.funcs[f].defs;
        if defs.iter().any(|(d, _)| decorated(d, "staticmethod")) {
            MethodKind::Static
        } else if defs.iter().any(|(d, _)| decorated(d, "classmethod")) {
            MethodKind::Class
        } else {
            MethodKind::Normal
        }
    }

    fn func_for(&mut self, m: ModId, owner: Option<ClassId>, name: &str, defs: Vec<(Rc<ast::StmtFunctionDef>, FileId)>) -> FuncId {
        let key = (m, owner, name.to_string());
        if let Some(&f) = self.func_keys.get(&key) {
            return f;
        }
        self.funcs.push(FuncInfo { name: name.to_string(), m, owner, defs, sigs: None });
        let f = self.funcs.len() - 1;
        self.func_keys.insert(key, f);
        f
    }

    /// A function defined inside another one, under a key unique in the
    /// module (`f.<locals>.g@offset`).
    pub fn local_func(&mut self, m: ModId, key: &str, name: &str, def: Rc<ast::StmtFunctionDef>, file: FileId) -> FuncId {
        let f = self.func_for(m, None, key, vec![(def, file)]);
        self.funcs[f].name = name.to_string();
        f
    }

    /// A member declared by class `c` itself (overlay first).
    pub fn class_member(&mut self, c: ClassId, name: &str) -> Option<Member> {
        if let Some(m) = self.classes[c].members.get(name) {
            return m.clone();
        }
        self.classes[c].members.insert(name.to_string(), None);
        let decl = self.classes[c].decl.clone();
        let m = self.classes[c].m;
        let mut found = None;
        for (stmt, file) in decl.overlay.iter().chain(decl.base.iter()) {
            let mut flat = Vec::new();
            self.flatten(&stmt.body, &mut flat);
            let defs: Vec<_> = flat
                .iter()
                .filter_map(|s| match s {
                    Stmt::FunctionDef(f) if f.name.as_str() == name => Some((Rc::new(f.clone()), *file)),
                    _ => None,
                })
                .collect();
            if !defs.is_empty() {
                let prop = defs.iter().any(|(d, _)| decorated(d, "property"));
                // `@name.getter`, `@name.setter` and `@name.deleter` defs are
                // the property's accessors, each its own function.
                let mut rest = defs;
                let mut accessors = Vec::new();
                for kind in ACCESSORS {
                    let (these, others): (Vec<_>, Vec<_>) = rest.into_iter().partition(|(d, _)| is_accessor(d, name, kind));
                    rest = others;
                    accessors.push((kind, these));
                }
                let f = self.func_for(m, Some(c), name, rest);
                for (kind, these) in accessors {
                    if prop && !these.is_empty() {
                        let af = self.func_for(m, Some(c), &format!("{name}.{kind}"), these);
                        self.funcs[af].name = name.to_string();
                        self.accessors.insert((f, kind), af);
                    }
                }
                found = Some(if prop { Member::Property(f) } else { Member::Method(f) });
                break;
            }
            let ann = flat.iter().find_map(|s| match s {
                Stmt::AnnAssign(a) if matches!(&*a.target, Expr::Name(n) if n.id.as_str() == name) => Some((*a.annotation).clone()),
                _ => None,
            });
            let tps = self.class_tparams(c);
            let self_ty = Ty::Var(self.self_tv(c));
            if let Some(ann) = ann {
                let ty = self.eval_type(&ann, &Scope { m, tparams: &tps, self_ty: Some(&self_ty) });
                found = Some(Member::Attr(ty));
                break;
            }
            let plain = flat.iter().find_map(|s| match s {
                Stmt::Assign(a) if matches!(a.targets.as_slice(), [Expr::Name(n)] if n.id.as_str() == name) => Some(literal_type(&a.value)),
                _ => None,
            });
            if let Some(ty) = plain {
                found = Some(Member::Attr(ty));
                break;
            }
            // `self.name: T = ...` in `__init__`.
            let init = flat.iter().find_map(|s| match s {
                Stmt::FunctionDef(f) if f.name.as_str() == "__init__" => Some(f),
                _ => None,
            });
            if let Some(init) = init {
                if let Some(ann) = init_attr_annotation(init, name) {
                    let ty = self.eval_type(ann, &Scope { m, tparams: &tps, self_ty: Some(&self_ty) });
                    found = Some(Member::Attr(ty));
                    break;
                }
            }
        }
        if found.is_none() && name == "__init__" && self.dataclass_opts(c).is_some_and(|o| o.init) {
            found = Some(Member::Method(self.dataclass_init(c)));
        }
        if found.is_none() && matches!(name, "__lt__" | "__le__" | "__gt__" | "__ge__") && self.dataclass_opts(c).is_some_and(|o| o.order) {
            found = Some(Member::Method(self.dataclass_order(c, name)));
        }
        self.classes[c].members.insert(name.to_string(), found.clone());
        found
    }

    /// An ordering method `@dataclass(order=True)` generates: it compares
    /// with an instance of the same class.
    fn dataclass_order(&mut self, c: ClassId, name: &str) -> FuncId {
        let m = self.classes[c].m;
        let f = self.func_for(m, Some(c), name, Vec::new());
        let self_ty = Ty::Var(self.self_tv(c));
        let params = vec![
            Param { name: "self".into(), ty: self_ty.clone(), kind: PKind::PosOnly, default: false },
            Param { name: "other".into(), ty: self_ty, kind: PKind::PosOnly, default: false },
        ];
        let file = self.modules[m].file.unwrap_or(0);
        let off = self.header(c).range().start().into();
        self.funcs[f].sigs = Some(Rc::new(vec![Sig { params, ret: Ty::Bool, file, off }]));
        f
    }

    /// The `__init__` `@dataclass` generates, as a function with a signature
    /// and no body.
    fn dataclass_init(&mut self, c: ClassId) -> FuncId {
        let m = self.classes[c].m;
        let f = self.func_for(m, Some(c), "__init__", Vec::new());
        let self_ty = Ty::Var(self.self_tv(c));
        let mut params = vec![Param { name: "self".into(), ty: self_ty, kind: PKind::PosOrKw, default: false }];
        for (name, ty, default, kw_only) in self.dataclass_fields(c) {
            params.push(Param { name, ty, kind: if kw_only { PKind::KwOnly } else { PKind::PosOrKw }, default });
        }
        // Keyword-only parameters come after the others.
        params.sort_by_key(|p| matches!(p.kind, PKind::KwOnly));
        let file = self.modules[m].file.unwrap_or(0);
        let off = self.header(c).range().start().into();
        self.funcs[f].sigs = Some(Rc::new(vec![Sig { params, ret: Ty::None, file, off }]));
        f
    }

    /// A member of `c` or its bases, with the class that declares it.
    pub fn find_member(&mut self, c: ClassId, name: &str) -> Option<(ClassId, Member)> {
        self.find_member_after(c, None, name)
    }

    /// A member found in `c`'s MRO, starting after class `after` if given
    /// (`super()`).
    pub fn find_member_after(&mut self, c: ClassId, after: Option<ClassId>, name: &str) -> Option<(ClassId, Member)> {
        let mro = self.mro(c);
        let start = after.and_then(|a| mro.iter().position(|&k| k == a)).map_or(0, |i| i + 1);
        for &k in &mro[start..] {
            if let Some(m) = self.class_member(k, name) {
                return Some((k, m));
            }
        }
        None
    }

    pub fn sigs(&mut self, f: FuncId) -> Rc<Vec<Sig>> {
        if let Some(s) = &self.funcs[f].sigs {
            return s.clone();
        }
        let defs = self.funcs[f].defs.clone();
        let overloads: Vec<_> = defs.iter().filter(|(d, _)| decorated(d, "overload")).cloned().collect();
        let chosen = if overloads.is_empty() { defs.last().cloned().into_iter().collect() } else { overloads };
        let mut sigs = Vec::new();
        for (d, file) in chosen {
            sigs.push(self.sig_of(f, &d, file));
        }
        let rc = Rc::new(sigs);
        self.funcs[f].sigs = Some(rc.clone());
        rc
    }

    fn sig_of(&mut self, f: FuncId, d: &ast::StmtFunctionDef, file: FileId) -> Sig {
        let m = self.funcs[f].m;
        let owner = self.funcs[f].owner;
        let mut tparams = match owner {
            Some(c) => self.class_tparams(c),
            None => Vec::new(),
        };
        tparams.extend(self.type_params(m, d.type_params.as_deref()));
        // `self` (unannotated) and `Self` are the class's `Self` variable: it
        // solves to the receiver's own type and, through its bound, the
        // class's type parameters.
        let self_ty = owner.map(|c| Ty::Var(self.self_tv(c)));
        let is_static = decorated(d, "staticmethod");
        let is_cls = decorated(d, "classmethod");
        let p = &d.parameters;
        let mut raw: Vec<(&ast::Parameter, PKind, bool)> = Vec::new();
        for x in &p.posonlyargs {
            raw.push((&x.parameter, PKind::PosOnly, x.default.is_some()));
        }
        for x in &p.args {
            raw.push((&x.parameter, PKind::PosOrKw, x.default.is_some()));
        }
        if let Some(x) = &p.vararg {
            raw.push((x, PKind::VarPos, false));
        }
        for x in &p.kwonlyargs {
            raw.push((&x.parameter, PKind::KwOnly, x.default.is_some()));
        }
        if let Some(x) = &p.kwarg {
            raw.push((x, PKind::VarKw, false));
        }
        let mut params = Vec::new();
        for (i, (param, kind, default)) in raw.into_iter().enumerate() {
            let scope = Scope { m, tparams: &tparams, self_ty: self_ty.as_ref() };
            let ty = match (&param.annotation, &self_ty) {
                (Some(a), _) => self.eval_type(a, &scope),
                (None, Some(s)) if i == 0 && !is_static => {
                    if is_cls {
                        Ty::Type(Box::new(s.clone()))
                    } else {
                        s.clone()
                    }
                }
                _ => {
                    self.err(m, param.range().start().into(), format!("parameter `{}` needs a type annotation", param.name.as_str()));
                    Ty::Any
                }
            };
            params.push(Param { name: param.name.as_str().to_string(), ty, kind, default });
        }
        let ret = match &d.returns {
            Some(r) => self.eval_type(r, &Scope { m, tparams: &tparams, self_ty: self_ty.as_ref() }),
            None if owner.is_some() && d.name.as_str() == "__init__" => Ty::None,
            None => {
                self.err(m, d.name.range().start().into(), format!("function `{}` needs a return type annotation", d.name.as_str()));
                Ty::Any
            }
        };
        Sig { params, ret, file, off: d.range().start().into() }
    }

    // ---- type expressions ----

    pub fn eval_type(&mut self, e: &Expr, sc: &Scope) -> Ty {
        let off: u32 = e.range().start().into();
        match e {
            Expr::NoneLiteral(_) => Ty::None,
            Expr::EllipsisLiteral(_) => Ty::Any,
            Expr::Name(n) => {
                if let Some((_, tv)) = sc.tparams.iter().find(|(p, _)| p == n.id.as_str()) {
                    return Ty::Var(*tv);
                }
                let s = self.lookup_name(sc.m, n.id.as_str());
                self.sym_type(s, sc, off, n.id.as_str())
            }
            Expr::Attribute(_) => match self.sym_of_expr(sc.m, e) {
                Some(s) => self.sym_type(s, sc, off, "attribute"),
                None => {
                    self.err(sc.m, off, "not a type");
                    Ty::Any
                }
            },
            Expr::Subscript(s) => {
                let args: Vec<&Expr> = slice_elts(&s.slice);
                let base = match &*s.value {
                    Expr::Name(n) if sc.tparams.iter().any(|(p, _)| p == n.id.as_str()) => None,
                    v => self.sym_of_expr(sc.m, v),
                };
                match base {
                    Some(Sym::Class(c)) => {
                        let params = self.class_params(c);
                        let mut targs: Vec<Ty> = args.iter().map(|a| self.eval_type(a, sc)).collect();
                        if targs.len() != params.len() {
                            if self.modules[sc.m].user {
                                let n = self.classes[c].name.clone();
                                self.err(sc.m, off, format!("`{n}` takes {} type arguments, got {}", params.len(), targs.len()));
                            }
                            targs.resize(params.len(), Ty::Any);
                        }
                        Ty::Inst(c, targs)
                    }
                    Some(Sym::Builtin(Ty::TupleVar(_))) | Some(Sym::Special(Special::Tuple)) => {
                        if args.len() == 2 && matches!(args[1], Expr::EllipsisLiteral(_)) {
                            Ty::TupleVar(Box::new(self.eval_type(args[0], sc)))
                        } else if matches!(&*s.slice, Expr::Tuple(t) if t.elts.is_empty()) {
                            Ty::Tuple(Vec::new())
                        } else {
                            Ty::Tuple(args.iter().map(|a| self.eval_type(a, sc)).collect())
                        }
                    }
                    Some(Sym::Builtin(Ty::Type(_))) | Some(Sym::Special(Special::Type)) => {
                        Ty::Type(Box::new(args.first().map_or(Ty::Any, |a| self.eval_type(a, sc))))
                    }
                    Some(Sym::Alias(a)) => {
                        let targs: Vec<Ty> = args.iter().map(|x| self.eval_type(x, sc)).collect();
                        let alias = &self.aliases[a];
                        let map: Subst = alias.params.iter().copied().zip(targs.into_iter().chain(std::iter::repeat(Ty::Any))).collect();
                        subst(&alias.target, &map)
                    }
                    Some(Sym::Special(Special::Literal)) => match args.first() {
                        Some(Expr::NumberLiteral(_)) => Ty::Int,
                        Some(Expr::StringLiteral(_)) => Ty::Str,
                        Some(Expr::BooleanLiteral(_)) => Ty::Bool,
                        _ => Ty::Any,
                    },
                    Some(Sym::Special(Special::Final | Special::ClassVar)) => args.first().map_or(Ty::Any, |a| self.eval_type(a, sc)),
                    Some(Sym::Special(Special::Callable)) => {
                        let ret = args.get(1).map_or(Ty::Any, |a| self.eval_type(a, sc));
                        let params = match args.first() {
                            Some(Expr::List(l)) => Some(l.elts.iter().map(|a| self.eval_type(a, sc)).collect()),
                            _ => None,
                        };
                        Ty::Callable(params, Box::new(ret))
                    }
                    Some(Sym::Special(Special::Union)) if self.unions_ok(sc.m) => {
                        let members = args.iter().map(|a| self.eval_type(a, sc)).collect();
                        union(members)
                    }
                    Some(Sym::Special(Special::Optional)) if self.unions_ok(sc.m) => {
                        let t = args.first().map_or(Ty::Any, |a| self.eval_type(a, sc));
                        union(vec![t, Ty::None])
                    }
                    // Other definition files use unions mostly for parameters
                    // built from protocols, which are not modelled yet.
                    Some(Sym::Special(Special::Union | Special::Optional)) => Ty::Any,
                    _ => {
                        self.err(sc.m, off, "not a generic type");
                        Ty::Any
                    }
                }
            }
            Expr::BinOp(b) if b.op == ast::Operator::BitOr => {
                if !self.unions_ok(sc.m) {
                    return Ty::Any;
                }
                let (l, r) = (self.eval_type(&b.left, sc), self.eval_type(&b.right, sc));
                union(vec![l, r])
            }
            _ => {
                self.err(sc.m, off, "not a type");
                Ty::Any
            }
        }
    }

    /// Whether unions in module `m` are types (Lizzard modules and the bundled
    /// builtins) or `Any` (other definition files).
    fn unions_ok(&self, m: ModId) -> bool {
        self.modules[m].user || self.modules[m].kind == ModKind::Builtins
    }

    fn sym_type(&mut self, s: Sym, sc: &Scope, off: u32, what: &str) -> Ty {
        match s {
            Sym::Class(c) => {
                let n = self.class_params(c).len();
                Ty::Inst(c, vec![Ty::Any; n])
            }
            Sym::Builtin(t) => t,
            Sym::TypeVar(tv) => Ty::Var(tv),
            Sym::Alias(a) => {
                let alias = &self.aliases[a];
                let map: Subst = alias.params.iter().map(|&p| (p, Ty::Any)).collect();
                subst(&alias.target, &map)
            }
            Sym::Special(Special::SelfTy) => sc.self_ty.cloned().unwrap_or(Ty::Any),
            Sym::Special(Special::Tuple) => Ty::TupleVar(Box::new(Ty::Any)),
            Sym::Special(Special::Type) => Ty::Type(Box::new(Ty::Any)),
            Sym::Special(Special::Callable) => Ty::Callable(None, Box::new(Ty::Any)),
            Sym::Any | Sym::Special(_) => Ty::Any,
            Sym::Value(_) | Sym::Module(_) | Sym::Func(_) | Sym::UserVar => {
                self.err(sc.m, off, format!("`{what}` is not a type"));
                Ty::Any
            }
        }
    }

    // ---- assignability ----

    /// Whether a value of type `src` may be used where `dst` is expected.
    /// Type variables in `dst` are solved into `s`. `any` is set when the
    /// answer depended on `Any` in `src` matching a constraining `dst` (the
    /// typing spec's ambiguous overload match).
    pub fn assign(&mut self, src: &Ty, dst: &Ty, s: &mut Subst, any: &mut bool) -> bool {
        if let Ty::Var(v) = dst {
            if self.rigid.contains(v) {
                return match src {
                    Ty::Var(w) if w == v => true,
                    Ty::Any => {
                        *any = true;
                        true
                    }
                    _ => false,
                };
            }
            return match s.get(v).cloned() {
                Some(b) => self.assign(src, &b, s, any),
                None => {
                    if let (Some(bound), false) = (self.tvs[*v].bound.clone(), matches!(src, Ty::Any)) {
                        if !self.assign(src, &bound, s, any) {
                            return false;
                        }
                    }
                    s.insert(*v, src.clone());
                    true
                }
            };
        }
        if matches!(dst, Ty::Any | Ty::Object) {
            return true;
        }
        if let Ty::Union(ms) = src {
            return ms.iter().all(|m| self.assign(m, dst, s, any));
        }
        if let Ty::Union(ms) = dst {
            if ms.contains(src) {
                return true;
            }
            for m in ms {
                let (mut s2, mut any2) = (s.clone(), *any);
                if self.assign(src, m, &mut s2, &mut any2) {
                    *s = s2;
                    *any = any2;
                    return true;
                }
            }
            return false;
        }
        match (src, dst) {
            (_, Ty::Any | Ty::Object) => true,
            (Ty::Var(w), _) if self.rigid.contains(w) => match self.tvs[*w].bound.clone() {
                Some(b) => self.assign(&b, dst, s, any),
                None => false,
            },
            (Ty::Any | Ty::Var(_), _) => {
                *any = true;
                true
            }
            (Ty::None, Ty::None) | (Ty::Bool, Ty::Bool) | (Ty::Int, Ty::Int) | (Ty::Float, Ty::Float) | (Ty::Str, Ty::Str) => true,
            (Ty::Bool, Ty::Int) | (Ty::Bool | Ty::Int, Ty::Float) => true,
            (Ty::Tuple(a), Ty::Tuple(b)) => a.len() == b.len() && a.iter().zip(b).all(|(x, y)| self.assign(x, y, s, any)),
            (Ty::Tuple(a), Ty::TupleVar(b)) => a.iter().all(|x| self.assign(x, b, s, any)),
            (Ty::TupleVar(a), Ty::TupleVar(b)) => self.assign(a, b, s, any),
            (Ty::TupleVar(a), Ty::Tuple(b)) if **a == Ty::Any => {
                *any = true;
                b.iter().all(|y| self.assign(&Ty::Any, y, s, any))
            }
            (Ty::Type(a), Ty::Type(b)) => self.assign(a, b, s, any),
            (Ty::Exc(a), Ty::Exc(b)) => exc_sub(a, b),
            (Ty::Callable(ps, r), Ty::Callable(qs, q)) => {
                // Parameters are contravariant, the result covariant.
                if let (Some(ps), Some(qs)) = (ps, qs) {
                    if ps.len() != qs.len() {
                        return false;
                    }
                    for (p, q) in ps.iter().zip(qs) {
                        if !self.assign(q, p, s, any) {
                            return false;
                        }
                    }
                }
                self.assign(r, q, s, any)
            }
            (Ty::Func(_) | Ty::Type(_) | Ty::Inst(..), Ty::Callable(..)) => match self.callable_of(src) {
                Some(c) => self.assign(&c, dst, s, any),
                None => false,
            },
            (Ty::Inst(c, args), _) => self.inst_assign(*c, args, dst, s, any),
            (Ty::Str | Ty::Tuple(_) | Ty::TupleVar(_), Ty::Inst(..)) => match self.nominal(src) {
                Some(n) => self.assign(&n, dst, s, any),
                None => false,
            },
            (Ty::Module(a), Ty::Module(b)) => a == b,
            (Ty::Func(a), Ty::Func(b)) => a == b,
            _ => false,
        }
    }

    /// The `Callable` type of a function, bound method, class (its
    /// constructor) or instance with `__call__`: its positional parameters
    /// and result, if it has a single signature.
    pub fn callable_of(&mut self, t: &Ty) -> Option<Ty> {
        let (f, skip, ret) = match t {
            // The builtins the checker handles itself.
            Ty::Func(FuncRef::Builtin("len")) => return Some(Ty::Callable(Some(vec![Ty::Any]), Box::new(Ty::Int))),
            Ty::Func(FuncRef::Builtin("print")) => return Some(Ty::Callable(None, Box::new(Ty::None))),
            Ty::Func(FuncRef::Builtin(_)) => return Some(Ty::Callable(None, Box::new(Ty::Any))),
            Ty::Func(FuncRef::Def(f)) => (*f, 0, None),
            Ty::Func(FuncRef::Bound(f, _)) => (*f, 1, None),
            Ty::Type(inner) => {
                let Ty::Inst(c, _) = &**inner else { return None };
                match self.find_member(*c, "__init__") {
                    Some((_, Member::Method(f))) => (f, 1, Some((**inner).clone())),
                    _ => return Some(Ty::Callable(Some(Vec::new()), inner.clone())),
                }
            }
            Ty::Inst(c, _) => match self.find_member(*c, "__call__") {
                Some((_, Member::Method(f))) => (f, 1, None),
                _ => return None,
            },
            _ => return None,
        };
        let sigs = self.sigs(f);
        let [sig] = sigs.as_slice() else { return None };
        let mut params = Vec::new();
        for p in sig.params.iter().skip(skip) {
            match p.kind {
                PKind::PosOnly | PKind::PosOrKw if !p.default => params.push(p.ty.clone()),
                PKind::KwOnly if !p.default => return None,
                _ => {}
            }
        }
        let mut sub = Subst::new();
        // A bound method's `self` and its class's parameters come from
        // the receiver.
        if let Ty::Func(FuncRef::Bound(_, recv)) = t {
            if let Some(p0) = sig.params.first() {
                self.assign(recv, &p0.ty, &mut sub, &mut false);
            }
        }
        let ret = ret.unwrap_or_else(|| deep_subst(&sig.ret, &sub));
        Some(Ty::Callable(Some(params.iter().map(|p| deep_subst(p, &sub)).collect()), Box::new(ret)))
    }

    fn inst_assign(&mut self, c: ClassId, args: &[Ty], dst: &Ty, s: &mut Subst, any: &mut bool) -> bool {
        if let Ty::Inst(d, dargs) = dst {
            if c == *d {
                let params = self.class_params(c);
                for (i, p) in params.iter().enumerate() {
                    let (a, b) = (args.get(i).unwrap_or(&Ty::Any), dargs.get(i).unwrap_or(&Ty::Any));
                    let ok = match self.tvs[*p].variance {
                        Variance::Co => self.assign(a, b, s, any),
                        Variance::Contra => {
                            let b = subst(b, s);
                            match &b {
                                // Solve an open variable from the argument.
                                Ty::Var(v) if !self.rigid.contains(v) => {
                                    s.insert(*v, a.clone());
                                    true
                                }
                                _ => self.assign(&b, a, &mut Subst::new(), any),
                            }
                        }
                        Variance::Inv => {
                            self.assign(a, b, s, any) && {
                                let b = subst(b, s);
                                self.assign(&b, a, &mut Subst::new(), &mut false)
                            }
                        }
                    };
                    if !ok {
                        return false;
                    }
                }
                return true;
            }
        }
        let params = self.class_params(c);
        let map: Subst = params.into_iter().zip(args.iter().cloned().chain(std::iter::repeat(Ty::Any))).collect();
        for base in self.class_bases(c) {
            let base = subst(&base, &map);
            let mut s2 = s.clone();
            let mut any2 = *any;
            if self.assign(&base, dst, &mut s2, &mut any2) {
                *s = s2;
                *any = any2;
                return true;
            }
        }
        self.protocol_assign(&Ty::Inst(c, args.to_vec()), dst, s, any)
    }

    /// `Iterable[T]` and `Iterator[T]` are protocols in typeshed: a class
    /// with `__iter__` (and `__next__`) of the right types is one.
    fn protocol_assign(&mut self, src: &Ty, dst: &Ty, s: &mut Subst, any: &mut bool) -> bool {
        let (Ty::Inst(c, _), Ty::Inst(d, dargs)) = (src, dst) else { return false };
        if self.protocol_depth > 4 {
            return false;
        }
        if self.is_protocol(*d) {
            self.protocol_depth += 1;
            let ok = self.structural_assign(*c, src, *d, dst, s, any);
            self.protocol_depth -= 1;
            return ok;
        }
        if !self.is_builtin_class(*d) || dargs.len() != 1 {
            return false;
        }
        let want = match self.classes[*d].name.as_str() {
            "Iterable" => Ty::Inst(*d, Vec::new()),
            "Iterator" => Ty::Any,
            _ => return false,
        };
        let iterable = matches!(want, Ty::Inst(..));
        let method = if iterable { "__iter__" } else { "__next__" };
        if !iterable && self.find_member(*c, "__iter__").is_none() {
            return false;
        }
        let Some((_, Member::Method(f))) = self.find_member(*c, method) else { return false };
        let sigs = self.sigs(f);
        let [sig] = sigs.as_slice() else { return false };
        // The result for this receiver.
        let mut sub = Subst::new();
        if let Some(p0) = sig.params.first() {
            self.assign(src, &p0.ty, &mut sub, &mut false);
        }
        let ret = deep_subst(&sig.ret, &sub);
        self.protocol_depth += 1;
        let ok = if iterable {
            match self.builtin_class("Iterator") {
                Some(it) => self.assign(&ret, &Ty::Inst(it, dargs.clone()), s, any),
                None => false,
            }
        } else {
            self.assign(&ret, &dargs[0], s, any)
        };
        self.protocol_depth -= 1;
        ok
    }

    // ---- display and runtime classes ----

    pub fn display(&self, t: &Ty) -> String {
        match t {
            Ty::Any => "Any".into(),
            Ty::None => "None".into(),
            Ty::Bool => "bool".into(),
            Ty::Int => "int".into(),
            Ty::Float => "float".into(),
            Ty::Str => "str".into(),
            Ty::Object => "object".into(),
            Ty::Tuple(v) if v.is_empty() => "tuple[()]".into(),
            Ty::Tuple(v) => format!("tuple[{}]", v.iter().map(|x| self.display(x)).collect::<Vec<_>>().join(", ")),
            Ty::TupleVar(x) => format!("tuple[{}, ...]", self.display(x)),
            Ty::Inst(c, args) => {
                if let Some((q, _)) = self.reverse_alias.get(t) {
                    return q.rsplit('.').next().unwrap_or(q).to_string();
                }
                let n = &self.classes[*c].name;
                if args.is_empty() {
                    n.clone()
                } else {
                    format!("{n}[{}]", args.iter().map(|x| self.display(x)).collect::<Vec<_>>().join(", "))
                }
            }
            Ty::Type(x) => format!("type[{}]", self.display(x)),
            Ty::Module(m) => format!("module '{}'", self.modules[*m].name),
            Ty::Func(_) => "function".into(),
            Ty::Var(v) => self.tvs[*v].name.clone(),
            Ty::Exc(n) => n.clone(),
            Ty::Union(v) => v.iter().map(|x| self.display(x)).collect::<Vec<_>>().join(" | "),
            Ty::Callable(None, r) => format!("Callable[..., {}]", self.display(r)),
            Ty::Callable(Some(ps), r) => format!("Callable[[{}], {}]", ps.iter().map(|x| self.display(x)).collect::<Vec<_>>().join(", "), self.display(r)),
        }
    }

    /// The runtime class whose instances have type `t`, by qualified name:
    /// the class itself, or for a parameterized class the alias its module
    /// exports for it (`floating[_32Bit]` is `numpy.float32`).
    pub fn runtime_class(&mut self, t: &Ty) -> Option<String> {
        let Ty::Inst(c, args) = t else { return None };
        if args.is_empty() {
            return Some(self.classes[*c].qualname.clone());
        }
        if !self.reverse_alias.contains_key(t) {
            self.resolve_aliases(self.classes[*c].m);
        }
        match self.reverse_alias.get(t) {
            Some((q, false)) => Some(q.clone()),
            _ => None,
        }
    }

    /// Resolve every alias a module declares (once), so that the reverse
    /// alias map is complete for it.
    fn resolve_aliases(&mut self, m: ModId) {
        if !self.aliases_resolved.insert(m) {
            return;
        }
        let names: Vec<String> = self.modules[m]
            .decls
            .iter()
            .filter(|(_, d)| match d {
                Decl::TypeAlias(_) => true,
                Decl::Assign(a) => matches!(a.value, Some(Expr::Subscript(_))),
                _ => false,
            })
            .map(|(n, _)| n.clone())
            .collect();
        let mut names = names;
        // Deterministic: the first name in order wins (`float32` over `single`).
        names.sort();
        for n in names {
            self.lookup_in(m, &n);
        }
    }

    /// `t` with unsolved type variables replaced by `Any`, except the rigid
    /// ones of the function being checked.
    pub fn erase(&self, t: &Ty) -> Ty {
        match t {
            Ty::Var(v) if self.rigid.contains(v) => t.clone(),
            Ty::Var(_) => Ty::Any,
            Ty::Tuple(v) => Ty::Tuple(v.iter().map(|x| self.erase(x)).collect()),
            Ty::TupleVar(x) => Ty::TupleVar(Box::new(self.erase(x))),
            Ty::Inst(c, args) => Ty::Inst(*c, args.iter().map(|x| self.erase(x)).collect()),
            Ty::Type(x) => Ty::Type(Box::new(self.erase(x))),
            Ty::Func(FuncRef::Bound(f, r)) => Ty::Func(FuncRef::Bound(*f, Box::new(self.erase(r)))),
            Ty::Union(v) => union(v.iter().map(|x| self.erase(x)).collect()),
            Ty::Callable(ps, r) => Ty::Callable(ps.as_ref().map(|v| v.iter().map(|x| self.erase(x)).collect()), Box::new(self.erase(r))),
            t => t.clone(),
        }
    }

    /// Fresh copies of the rigid type variables occurring in `tys` (and in
    /// their bounds): a callee's own signature mentions the same variables
    /// as the body being checked, but there they are solved per call.
    pub fn fresh_rigid(&mut self, tys: &[&Ty]) -> Subst {
        let mut vars = Vec::new();
        for t in tys {
            collect_vars(t, &mut vars);
        }
        let mut i = 0;
        while i < vars.len() {
            if let Some(b) = self.tvs[vars[i]].bound.clone() {
                collect_vars(&b, &mut vars);
            }
            i += 1;
        }
        let mut map = Subst::new();
        vars.retain(|v| self.rigid.contains(v));
        for v in vars {
            let (name, variance) = (self.tvs[v].name.clone(), self.tvs[v].variance);
            map.insert(v, Ty::Var(self.new_tv(&name, variance)));
        }
        let pairs: Vec<(TvId, TvId)> = map.iter().map(|(k, v)| (*k, if let Ty::Var(n) = v { *n } else { unreachable!() })).collect();
        for (old, new) in pairs {
            self.tvs[new].bound = self.tvs[old].bound.as_ref().map(|b| subst(b, &map));
        }
        map
    }

    pub fn qualname(&self, c: ClassId) -> &str {
        &self.classes[c].qualname
    }

    pub fn decl_origin(&self, file: FileId, off: u32) -> String {
        let f = &self.files[file];
        format!("{}:{}", f.path, f.line_col(off).0)
    }
}

pub fn subst(t: &Ty, s: &Subst) -> Ty {
    match t {
        Ty::Var(v) => s.get(v).cloned().unwrap_or_else(|| t.clone()),
        Ty::Tuple(v) => Ty::Tuple(v.iter().map(|x| subst(x, s)).collect()),
        Ty::TupleVar(x) => Ty::TupleVar(Box::new(subst(x, s))),
        Ty::Inst(c, args) => Ty::Inst(*c, args.iter().map(|x| subst(x, s)).collect()),
        Ty::Type(x) => Ty::Type(Box::new(subst(x, s))),
        Ty::Func(FuncRef::Bound(f, r)) => Ty::Func(FuncRef::Bound(*f, Box::new(subst(r, s)))),
        Ty::Union(v) => union(v.iter().map(|x| subst(x, s)).collect()),
        Ty::Callable(ps, r) => Ty::Callable(ps.as_ref().map(|v| v.iter().map(|x| subst(x, s)).collect()), Box::new(subst(r, s))),
        t => t.clone(),
    }
}

/// The union of `tys`: flattened, without duplicates; `Any` absorbs the
/// rest, and a single member is itself.
pub fn union(tys: Vec<Ty>) -> Ty {
    let mut out: Vec<Ty> = Vec::new();
    for t in tys {
        let members = match t {
            Ty::Union(v) => v,
            t => vec![t],
        };
        for m in members {
            if m == Ty::Any {
                return Ty::Any;
            }
            if !out.contains(&m) {
                out.push(m);
            }
        }
    }
    match out.len() {
        0 => Ty::Any,
        1 => out.pop().unwrap(),
        _ => Ty::Union(out),
    }
}

/// The members of a union, or the type itself.
pub fn members(t: &Ty) -> Vec<Ty> {
    match t {
        Ty::Union(v) => v.clone(),
        t => vec![t.clone()],
    }
}

/// `subst`, following variables solved to other solved variables.
pub fn deep_subst(t: &Ty, s: &Subst) -> Ty {
    let mut t = subst(t, s);
    for _ in 0..8 {
        let mut vars = Vec::new();
        collect_vars(&t, &mut vars);
        if !vars.iter().any(|v| s.get(v).is_some_and(|x| *x != Ty::Var(*v))) {
            break;
        }
        t = subst(&t, s);
    }
    t
}

/// The type of a class attribute assigned without annotation.
fn literal_type(e: &Expr) -> Ty {
    match e {
        Expr::NumberLiteral(n) => match n.value {
            ast::Number::Int(_) => Ty::Int,
            ast::Number::Float(_) => Ty::Float,
            ast::Number::Complex { .. } => Ty::Any,
        },
        Expr::StringLiteral(_) => Ty::Str,
        Expr::BooleanLiteral(_) => Ty::Bool,
        _ => Ty::Any,
    }
}

/// The annotation of `self.name: T = ...` in an `__init__` body.
pub fn init_attr_annotation<'a>(init: &'a ast::StmtFunctionDef, name: &str) -> Option<&'a Expr> {
    let this = init.parameters.posonlyargs.iter().chain(init.parameters.args.iter()).next()?.parameter.name.as_str();
    fn find<'a>(body: &'a [Stmt], this: &str, name: &str) -> Option<&'a Expr> {
        for s in body {
            let found = match s {
                Stmt::AnnAssign(a) => match &*a.target {
                    Expr::Attribute(at) if at.attr.as_str() == name && matches!(&*at.value, Expr::Name(n) if n.id.as_str() == this) => {
                        Some(&*a.annotation)
                    }
                    _ => None,
                },
                Stmt::If(i) => find(&i.body, this, name).or_else(|| i.elif_else_clauses.iter().find_map(|c| find(&c.body, this, name))),
                Stmt::For(f) => find(&f.body, this, name),
                Stmt::While(w) => find(&w.body, this, name),
                _ => None,
            };
            if found.is_some() {
                return found;
            }
        }
        None
    }
    find(&init.body, this, name)
}

/// Replace unsolved type variables by `Any`.
pub fn erase_vars(t: &Ty) -> Ty {
    match t {
        Ty::Var(_) => Ty::Any,
        Ty::Tuple(v) => Ty::Tuple(v.iter().map(erase_vars).collect()),
        Ty::TupleVar(x) => Ty::TupleVar(Box::new(erase_vars(x))),
        Ty::Inst(c, args) => Ty::Inst(*c, args.iter().map(erase_vars).collect()),
        Ty::Type(x) => Ty::Type(Box::new(erase_vars(x))),
        Ty::Union(v) => union(v.iter().map(erase_vars).collect()),
        Ty::Callable(ps, r) => Ty::Callable(ps.as_ref().map(|v| v.iter().map(erase_vars).collect()), Box::new(erase_vars(r))),
        t => t.clone(),
    }
}

/// Whether `t` contains `Any` or an unsolved type variable.
pub fn has_any(t: &Ty) -> bool {
    match t {
        Ty::Any | Ty::Var(_) => true,
        Ty::Tuple(v) | Ty::Union(v) => v.iter().any(has_any),
        Ty::TupleVar(x) | Ty::Type(x) => has_any(x),
        Ty::Inst(_, args) => args.iter().any(has_any),
        Ty::Callable(ps, r) => ps.as_ref().is_some_and(|v| v.iter().any(has_any)) || has_any(r),
        _ => false,
    }
}

pub fn collect_vars(t: &Ty, out: &mut Vec<TvId>) {
    match t {
        Ty::Var(v) => {
            if !out.contains(v) {
                out.push(*v);
            }
        }
        Ty::Tuple(v) | Ty::Union(v) => v.iter().for_each(|x| collect_vars(x, out)),
        Ty::TupleVar(x) | Ty::Type(x) => collect_vars(x, out),
        Ty::Inst(_, args) => args.iter().for_each(|x| collect_vars(x, out)),
        Ty::Callable(ps, r) => {
            ps.iter().flatten().for_each(|x| collect_vars(x, out));
            collect_vars(r, out);
        }
        _ => {}
    }
}

/// The builtin exception classes the checker knows, with their bases.
const EXCEPTION_TREE: &[(&str, &str)] = &[
    ("BaseException", ""),
    ("Exception", "BaseException"),
    ("SystemExit", "BaseException"),
    ("KeyboardInterrupt", "BaseException"),
    ("GeneratorExit", "BaseException"),
    ("ArithmeticError", "Exception"),
    ("ZeroDivisionError", "ArithmeticError"),
    ("OverflowError", "ArithmeticError"),
    ("FloatingPointError", "ArithmeticError"),
    ("AssertionError", "Exception"),
    ("AttributeError", "Exception"),
    ("BufferError", "Exception"),
    ("EOFError", "Exception"),
    ("ImportError", "Exception"),
    ("ModuleNotFoundError", "ImportError"),
    ("LookupError", "Exception"),
    ("IndexError", "LookupError"),
    ("KeyError", "LookupError"),
    ("MemoryError", "Exception"),
    ("NameError", "Exception"),
    ("UnboundLocalError", "NameError"),
    ("OSError", "Exception"),
    ("FileNotFoundError", "OSError"),
    ("FileExistsError", "OSError"),
    ("PermissionError", "OSError"),
    ("IsADirectoryError", "OSError"),
    ("NotADirectoryError", "OSError"),
    ("TimeoutError", "OSError"),
    ("ConnectionError", "OSError"),
    ("ReferenceError", "Exception"),
    ("RuntimeError", "Exception"),
    ("NotImplementedError", "RuntimeError"),
    ("RecursionError", "RuntimeError"),
    ("StopIteration", "Exception"),
    ("SyntaxError", "Exception"),
    ("TypeError", "Exception"),
    ("ValueError", "Exception"),
    ("UnicodeError", "ValueError"),
    ("UnicodeDecodeError", "UnicodeError"),
    ("UnicodeEncodeError", "UnicodeError"),
    ("Warning", "Exception"),
    ("UserWarning", "Warning"),
    ("DeprecationWarning", "Warning"),
    ("RuntimeWarning", "Warning"),
];

pub fn is_exception_name(n: &str) -> bool {
    EXCEPTION_TREE.iter().any(|(e, _)| *e == n)
}

fn exc_sub(a: &str, b: &str) -> bool {
    let mut cur = a;
    loop {
        if cur == b {
            return true;
        }
        match EXCEPTION_TREE.iter().find(|(e, _)| *e == cur) {
            Some((_, p)) if !p.is_empty() => cur = p,
            _ => return false,
        }
    }
}

fn builtin_sym(name: &str) -> Sym {
    match name {
        "int" => Sym::Builtin(Ty::Int),
        "float" => Sym::Builtin(Ty::Float),
        "bool" => Sym::Builtin(Ty::Bool),
        "str" => Sym::Builtin(Ty::Str),
        "object" => Sym::Builtin(Ty::Object),
        "tuple" => Sym::Builtin(Ty::TupleVar(Box::new(Ty::Any))),
        "type" => Sym::Builtin(Ty::Type(Box::new(Ty::Any))),
        "None" => Sym::Builtin(Ty::None),
        n if is_exception_name(n) => Sym::Builtin(Ty::Exc(n.to_string())),
        _ => Sym::Any,
    }
}

fn special_sym(name: &str) -> Sym {
    Sym::Special(match name {
        "Any" => Special::Any,
        "TypeVar" => Special::TypeVar,
        "TypeAlias" => Special::TypeAlias,
        "Generic" => Special::Generic,
        "Protocol" => Special::Protocol,
        "overload" => Special::Overload,
        "Self" => Special::SelfTy,
        "Literal" => Special::Literal,
        "Union" => Special::Union,
        "Optional" => Special::Optional,
        "Final" => Special::Final,
        "ClassVar" => Special::ClassVar,
        "TYPE_CHECKING" => Special::TypeChecking,
        "Never" | "NoReturn" => Special::Never,
        "Tuple" => Special::Tuple,
        "Type" => Special::Type,
        "Callable" => Special::Callable,
        _ => return Sym::Any,
    })
}
