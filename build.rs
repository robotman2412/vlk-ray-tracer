use std::process::Command;

fn main() {
    Command::new("glslc")
        .args(&["shader/vert.vert", "-o", "shader/vert.spv"])
        .status()
        .expect("Failed to compile vertex shader");
    Command::new("glslc")
        .args(&["shader/frag.frag", "-o", "shader/frag.spv"])
        .status()
        .expect("Failed to compile fragment shader");
    Command::new("glslc")
        .args(&[
            "-fshader-stage=comp",
            "-std=450core",
            "shader/rt.glsl",
            "-o",
            "shader/rt.spv",
        ])
        .status()
        .expect("Failed to compile vertex shader");
    println!("cargo:rerun-if-changed=shader/vert.vert");
    println!("cargo:rerun-if-changed=shader/frag.frag");
    println!("cargo:rerun-if-changed=shader/rt.glsl");
}
