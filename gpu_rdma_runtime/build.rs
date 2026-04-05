use std::env;
use std::path::PathBuf;

fn main() {
    // println!("cargo:rerun-if-changed=CMakeLists.txt");
    // println!("cargo:rerun-if-changed=kernel_scaffold.cu");

    // let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR is set by Cargo"));
    // let build_dir = manifest_dir.join("build");

    // run(
    //     Command::new("cmake")
    //         .arg("-S")
    //         .arg(&manifest_dir)
    //         .arg("-B")
    //         .arg(&build_dir),
    //     "cmake configure",
    // );

    // let jobs = std::thread::available_parallelism()
    //     .map(usize::from)
    //     .unwrap_or(1)
    //     .to_string();
    // run(
    //     Command::new("cmake")
    //         .arg("--build")
    //         .arg(&build_dir)
    //         .arg("-j")
    //         .arg(jobs),
    //     "cmake build",
    // );
}
