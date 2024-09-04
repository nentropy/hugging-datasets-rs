/// Utils for the use of Bridge 
/// 
use indicatif::ProgressBar;
use std::thread;
use std::time::Duration;
use dialoguer::Input;

fn progress_bar() {
    let bar = ProgressBar::new(100);
    for _ in 0..100 {
        bar.inc(1);
        thread::sleep(Duration::from_millis(50));
    }
    bar.finish();
}



fn dialog_main() {
    let name: String = Input::new()
        .with_prompt("Enter your name")
        .interact_text()
        .unwrap();

    println!("Hello, {}!", name);
}