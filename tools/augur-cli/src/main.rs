//! `augur` command-line interface.

use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use anyhow::{anyhow, Context, Result};
use augur_frontend::{format_program, parse};
use augur_ir::lower;
use augur_mlir::{build_graph, to_dot};
use augur_runtime::{select_engine, Engine, InferOptions};
use clap::{Parser, Subcommand};
use std::str::FromStr;

#[derive(Parser)]
#[command(
    name = "augur",
    version,
    about = "TPT Augur probabilistic programming CLI"
)]
struct Cli {
    #[command(subcommand)]
    command: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Parse, type-check, and run inference on a model file.
    Run {
        file: PathBuf,
        #[arg(short, long)]
        engine: Option<String>,
        #[arg(short = 'n', long, default_value_t = 2000)]
        samples: usize,
        #[arg(short = 'c', long, default_value_t = 4)]
        chains: usize,
        #[arg(long, default_value_t = 1000)]
        warmup: usize,
        #[arg(long, default_value_t = 0xC0FFEE)]
        seed: u64,
    },
    /// Type-check a model and report diagnostics without running inference.
    Check { file: PathBuf },
    /// Pretty-print a model to canonical formatting.
    Fmt { file: PathBuf },
    /// Lower a model to TPTIR text (Augur MLIR dialect) for tpt-gpu consumption.
    Build {
        file: PathBuf,
        #[arg(long, default_value = "model")]
        entry: String,
        #[arg(long, default_value = "cpu")]
        hardware: String,
        #[arg(short, long)]
        output: Option<PathBuf>,
    },
    /// Read a model from stdin and run inference on it.
    Repl,
    /// Emit the probabilistic inference graph (Graphviz DOT) for a model.
    Graph {
        file: PathBuf,
        /// Write DOT to this path instead of stdout.
        #[arg(short, long)]
        output: Option<PathBuf>,
    },
    /// Publish an Augur package (manifest + module sources) to the local registry.
    Publish {
        /// Path to the `Augur.toml` manifest.
        manifest: PathBuf,
        /// Module source files exported by the package (relative paths preserved).
        #[arg(short, long = "src")]
        sources: Vec<PathBuf>,
    },
    /// Install a package from the local registry into a destination directory.
    Install {
        name: String,
        version: String,
        #[arg(short, long, default_value = ".")]
        dest: PathBuf,
    },
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    match run(cli) {
        Ok(code) => code,
        Err(e) => {
            eprintln!("error: {e:#}");
            ExitCode::FAILURE
        }
    }
}

fn run(cli: Cli) -> Result<ExitCode> {
    match cli.command {
        Cmd::Run {
            file,
            engine,
            samples,
            chains,
            warmup,
            seed,
        } => cmd_run(&file, engine.as_deref(), samples, chains, warmup, seed),
        Cmd::Check { file } => cmd_check(&file),
        Cmd::Fmt { file } => cmd_fmt(&file),
        Cmd::Build {
            file,
            entry,
            hardware,
            output,
        } => cmd_build(&file, &entry, &hardware, output.as_ref()),
        Cmd::Repl => cmd_repl(),
        Cmd::Graph { file, output } => cmd_graph(&file, output.as_ref()),
        Cmd::Publish { manifest, sources } => cmd_publish(&manifest, &sources),
        Cmd::Install {
            name,
            version,
            dest,
        } => cmd_install(&name, &version, &dest),
    }
}

fn read_source(file: &PathBuf) -> Result<String> {
    std::fs::read_to_string(file).with_context(|| format!("reading `{}`", file.display()))
}

fn cmd_run(
    file: &PathBuf,
    engine: Option<&str>,
    samples: usize,
    chains: usize,
    warmup: usize,
    seed: u64,
) -> Result<ExitCode> {
    let src = read_source(file)?;
    let parsed = parse(&src);
    if parsed.has_errors() {
        report_frontend_diags(&parsed.diagnostics);
        return Ok(ExitCode::FAILURE);
    }
    let lowered = lower(&parsed.program);
    if lowered.diagnostics.iter().any(|d| d.is_error()) {
        report_ir_diags(&lowered.diagnostics);
        return Ok(ExitCode::FAILURE);
    }
    for w in &lowered.diagnostics {
        if !w.is_error() {
            eprintln!("warning: {}", w.message);
        }
    }

    let chosen = match engine {
        Some(name) => Engine::from_str(name).map_err(|e| anyhow!(e))?,
        None => select_engine(&lowered.model),
    };
    let opts = InferOptions {
        engine: Some(chosen),
        num_chains: chains,
        num_warmup: warmup,
        num_samples: samples,
        seed,
        ..Default::default()
    };

    println!(
        "engine: {} (auto-selected: {})",
        chosen.as_str(),
        engine.is_none()
    );
    println!("chains: {chains}, warmup: {warmup}, samples: {samples}\n");

    let result = augur_runtime::run(&lowered.model, &opts);
    print_summary(&result);

    let max_rhat = result
        .summaries
        .iter()
        .map(|s| s.rhat)
        .fold(0.0_f64, f64::max);
    if max_rhat > 1.1 {
        eprintln!(
            "\nwarning: R-hat > 1.1 for at least one parameter; chains may not have converged."
        );
    }
    Ok(ExitCode::SUCCESS)
}

fn cmd_check(file: &PathBuf) -> Result<ExitCode> {
    let src = read_source(file)?;
    let parsed = parse(&src);
    if parsed.has_errors() {
        report_frontend_diags(&parsed.diagnostics);
        return Ok(ExitCode::FAILURE);
    }
    let lowered = lower(&parsed.program);
    if lowered.diagnostics.iter().any(|d| d.is_error()) {
        report_ir_diags(&lowered.diagnostics);
        return Ok(ExitCode::FAILURE);
    }
    println!(
        "ok: {} statement(s), {} prior variable(s); suggested engine: {}",
        lowered.model.items.len(),
        lowered.model.prior_order.len(),
        select_engine(&lowered.model).as_str()
    );
    Ok(ExitCode::SUCCESS)
}

fn cmd_fmt(file: &PathBuf) -> Result<ExitCode> {
    let src = read_source(file)?;
    let parsed = parse(&src);
    if parsed.has_errors() {
        report_frontend_diags(&parsed.diagnostics);
        return Ok(ExitCode::FAILURE);
    }
    let formatted = format_program(&parsed.program);
    print!("{formatted}");
    Ok(ExitCode::SUCCESS)
}

fn cmd_build(
    file: &PathBuf,
    entry: &str,
    hardware: &str,
    output: Option<&PathBuf>,
) -> Result<ExitCode> {
    let src = read_source(file)?;
    let parsed = parse(&src);
    if parsed.has_errors() {
        report_frontend_diags(&parsed.diagnostics);
        return Ok(ExitCode::FAILURE);
    }
    let lowered = lower(&parsed.program);
    if lowered.diagnostics.iter().any(|d| d.is_error()) {
        report_ir_diags(&lowered.diagnostics);
        return Ok(ExitCode::FAILURE);
    }
    let (tptir, changes) = augur_mlir::compile_model_to_tptir(&lowered.model, entry, hardware);
    if let Some(out) = output {
        std::fs::write(out, &tptir).with_context(|| format!("writing `{}`", out.display()))?;
        println!(
            "wrote TPTIR for `{}` (hardware={}, {} optimization change(s)) to {}",
            file.display(),
            hardware,
            changes,
            out.display()
        );
    } else {
        println!(
            "// TPTIR for `{}` (hardware={}, {} optimization change(s))",
            file.display(),
            hardware,
            changes
        );
        print!("{tptir}");
    }
    Ok(ExitCode::SUCCESS)
}

fn cmd_publish(manifest: &Path, sources: &[PathBuf]) -> Result<ExitCode> {
    let man = augur_pkg::Manifest::load(manifest)
        .with_context(|| format!("loading manifest `{}`", manifest.display()))?;
    let mut files = Vec::new();
    for src in sources {
        let contents = std::fs::read_to_string(src)
            .with_context(|| format!("reading source `{}`", src.display()))?;
        let path = src
            .file_name()
            .and_then(|n| n.to_str())
            .ok_or_else(|| anyhow!("source `{}` has no file name", src.display()))?
            .to_string();
        files.push(augur_pkg::PackageFile { path, contents });
    }
    let pkg = augur_pkg::PublishedPackage {
        manifest: man,
        sources: files,
    };
    let registry = augur_pkg::Registry::open_default()?;
    registry
        .publish(&pkg)
        .context("publishing to local registry")?;
    println!(
        "published {}@{} ({}/{} bytes of module sources)",
        pkg.manifest.package.name,
        pkg.manifest.package.version,
        pkg.sources.len(),
        pkg.sources.iter().map(|f| f.contents.len()).sum::<usize>()
    );
    Ok(ExitCode::SUCCESS)
}

fn cmd_install(name: &str, version: &str, dest: &Path) -> Result<ExitCode> {
    let registry = augur_pkg::Registry::open_default()?;
    let installed = registry
        .install(name, version, dest)
        .with_context(|| format!("installing {name}@{version}"))?;
    for p in &installed {
        println!("installed {}", p.display());
    }
    Ok(ExitCode::SUCCESS)
}

fn cmd_graph(file: &PathBuf, output: Option<&PathBuf>) -> Result<ExitCode> {
    let src = read_source(file)?;
    let parsed = parse(&src);
    if parsed.has_errors() {
        report_frontend_diags(&parsed.diagnostics);
        return Ok(ExitCode::FAILURE);
    }
    let lowered = lower(&parsed.program);
    if lowered.diagnostics.iter().any(|d| d.is_error()) {
        report_ir_diags(&lowered.diagnostics);
        return Ok(ExitCode::FAILURE);
    }
    let graph = build_graph(&lowered.model);
    let dot = to_dot(&graph);
    match output {
        Some(out) => {
            std::fs::write(out, &dot).with_context(|| format!("writing `{}`", out.display()))?;
            println!(
                "wrote inference graph ({} op(s), {} prior(s)) to {}",
                graph.op_count(),
                graph.prior_order.len(),
                out.display()
            );
        }
        None => print!("{dot}"),
    }
    Ok(ExitCode::SUCCESS)
}

fn cmd_repl() -> Result<ExitCode> {
    let mut src = String::new();
    std::io::stdin()
        .read_to_string(&mut src)
        .context("reading stdin")?;
    let parsed = parse(&src);
    if parsed.has_errors() {
        report_frontend_diags(&parsed.diagnostics);
        return Ok(ExitCode::FAILURE);
    }
    let lowered = lower(&parsed.program);
    if lowered.diagnostics.iter().any(|d| d.is_error()) {
        report_ir_diags(&lowered.diagnostics);
        return Ok(ExitCode::FAILURE);
    }
    let engine = select_engine(&lowered.model);
    let opts = InferOptions {
        engine: Some(engine),
        ..Default::default()
    };
    let result = augur_runtime::run(&lowered.model, &opts);
    print_summary(&result);
    Ok(ExitCode::SUCCESS)
}

fn print_summary(r: &augur_runtime::InferenceResult) {
    println!(
        "{:<14}{:>12}{:>12}{:>12}{:>12}{:>10}",
        "param", "mean", "sd", "2.5%", "97.5%", "r_hat"
    );
    println!("{}", "-".repeat(72));
    for s in &r.summaries {
        println!(
            "{:<14}{:>12.4}{:>12.4}{:>12.4}{:>12.4}{:>10.3}",
            s.name, s.mean, s.sd, s.q2_5, s.q97_5, s.rhat
        );
    }
}

fn report_frontend_diags(diags: &[augur_frontend::Diagnostic]) {
    for d in diags {
        eprintln!(
            "error: {} (span {}-{})",
            d.message, d.span.start, d.span.end
        );
    }
}

fn report_ir_diags(diags: &[augur_ir::Diagnostic]) {
    for d in diags {
        let kind = if d.is_error() { "error" } else { "warning" };
        eprintln!("{kind}: {}", d.message);
    }
}
