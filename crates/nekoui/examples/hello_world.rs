use nekoui::Application;
use nekoui::window::WindowOptions;

fn main() {
    let run = Application::new()
        .run_test(|cx| {
            cx.windows()
                .open(WindowOptions::new().title("Hello NekoUI"))?;
            cx.notify();
            Ok(())
        })
        .unwrap();

    println!("{:#?}", run.performance_report());
}
