//! # Utils for the use of Bridge
//!
//! This module provides utility functions for enhancing user interaction in the `Bridge` project. 
//! It includes a progress bar and a user input dialog, making it easier to track progress and 
//! interact with users during runtime.

use indicatif::ProgressBar;
use std::thread;
use std::time::Duration;
use dialoguer::Input;

/// Displays a progress bar that simulates the progress of a task.
///
/// The progress bar increments by 1 every 50 milliseconds until it reaches 100, at which point
/// it finishes. This is useful for showing progress to users during tasks that take time to complete.
///
/// # Example
///
/// ```rust
/// use indicatif::ProgressBar;
/// use std::thread;
/// use std::time::Duration;
///
/// fn main() {
///     progress_bar();
/// }
///
/// fn progress_bar() {
///     let bar = ProgressBar::new(100);
///     for _ in 0..100 {
///         bar.inc(1);
///         thread::sleep(Duration::from_millis(50));
///     }
///     bar.finish();
/// }
/// ```
fn progress_bar() {
    let bar = ProgressBar::new(100);
    for _ in 0..100 {
        bar.inc(1);
        thread::sleep(Duration::from_millis(50));
    }
    bar.finish();
}

/// Prompts the user for input and prints a greeting message.
///
/// This function uses the `dialoguer` crate to prompt the user to input their name. Once the user
/// enters their name, the function prints a greeting message. This is useful for simple interactions
/// where the program needs to gather information from the user.
///
/// # Example
///
/// ```rust
/// use dialoguer::Input;
///
/// fn main() {
///     dialog_main();
/// }
///
/// fn dialog_main() {
///     let name: String = Input::new()
///         .with_prompt("Enter your name")
///         .interact_text()
///         .unwrap();
///
///     println!("Hello, {}!", name);
/// }
/// ```
fn dialog_main() {
    let name: String = Input::new()
        .with_prompt("Enter your name")
        .interact_text()
        .unwrap();

    println!("Hello, {}!", name);
}
