use nalgebra::{Matrix2, Vector2};
use utils::progress_bar;
use syn_crabs::configure_logger;
use uuidv4 as id_q;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dataloader() {
        let dataset = ParquetSecurityDataset {
            data: vec![vec![1.0], vec![2.0], vec![3.0], vec![4.0], vec![5.0]],
        };
        let mut dataloader = DataLoader::new(dataset, 2, true);

        let mut total_batches = 0;
        let mut seen_items = std::collections::HashSet::new();

        for batch in &mut dataloader {
            total_batches += 1;
            for item in batch {
                seen_items.insert(item[0] as i32);
            }
        }

        assert_eq!(total_batches, 3);
        assert_eq!(seen_items.len(), 5);
    }

#[test]
fn test_initialization_matrix2_vector2() {
    let weights = Matrix2::new(1.0, 0.0, 0.0, 1.0);
    let input_data = Vector2::new(1.0, 2.0);

    assert_eq!(weights[(0, 0)], 1.0);
    assert_eq!(weights[(1, 1)], 1.0);
    assert_eq!(input_data[0], 1.0);
    assert_eq!(input_data[1], 2.0);
}

use nalgebra::{Matrix2, Vector2};
use simba::simd::f32x4;
use syn_crabs::configure_logger;
use uuidv4 as id_q;

#[test]
fn test_handling_empty_input_data() {
    let weights = Matrix2::new(1.0, 0.0, 0.0, 1.0);
    let input_data = Vector2::new(0.0, 0.0);

    let predictions = weights * input_data;
    assert_eq!(predictions[0], 0.0);
    assert_eq!(predictions[1], 0.0);

    let inputs_simd = f32x4::new(input_data[0], input_data[1], 0.0, 0.0);
    let weights_simd = f32x4::new(weights[(0, 0)], weights[(1, 1)], 0.0, 0.0);

    let predictions_simd = weights_simd * inputs_simd;
    assert_eq!(predictions_simd.extract(0), 0.0);
    assert_eq!(predictions_simd.extract(1), 0.0);
}

fn test_handling_empty_input_data() {
    let weights = Matrix2::new(1.0, 0.0, 0.0, 1.0);
    let input_data = Vector2::new(0.0, 0.0);

    let predictions = weights * input_data;
    assert_eq!(predictions[0], 0.0);
    assert_eq!(predictions[1], 0.0);

    let inputs_simd = f32x4::new(input_data[0], input_data[1], 0.0, 0.0);
    let weights_simd = f32x4::new(weights[(0, 0)], weights[(1, 1)], 0.0, 0.0);

    let predictions_simd = weights_simd * inputs_simd;
    assert_eq!(predictions_simd.extract(0), 0.0);
    assert_eq!(predictions_simd.extract(1), 0.0);
}

}