#![deny(unsafe_code)]

fn main() {
    if let Err(error) = nekoui_build::shader::build_shaders() {
        println!("cargo::error=NekoUI framework shader build failed");
        for line in error.lines() {
            println!("cargo::error={line}");
        }
        std::process::exit(1);
    }
}
