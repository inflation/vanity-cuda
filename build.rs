use std::path::{Path, PathBuf};
use std::process::Command;

const KERNEL: &str = "kernels/vanity.cu";

fn main() {
    println!("cargo:rerun-if-changed={KERNEL}");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    let src = std::fs::read_to_string(KERNEL).unwrap();
    for name in ["BATCH", "BLOCKS_PER_SM"] {
        let value = src
            .lines()
            .find_map(|l| l.strip_prefix(&format!("#define {name} ")))
            .unwrap_or_else(|| panic!("kernel has no `#define {name}`"))
            .trim();
        println!("cargo:rustc-env=VANITY_{name}={value}");
    }

    let out = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let cuda = std::env::var("CUDA_PATH").expect("CUDA_PATH is not set; install the CUDA Toolkit");
    let mut nvcc = Command::new(Path::new(&cuda).join("bin").join("nvcc"));
    nvcc.args(["-ptx", "-O3", "-arch=sm_89", "-o"])
        .arg(out.join("vanity.ptx"))
        .arg(KERNEL);
    if let Some(cl) = cc::windows_registry::find_tool("x86_64-pc-windows-msvc", "cl.exe") {
        nvcc.arg("-ccbin").arg(cl.path().parent().unwrap());
    }
    let status = nvcc.status().expect("failed to run nvcc");
    assert!(status.success(), "nvcc failed to compile {KERNEL}");
}
