
use std::env;
use std::fs::File;
use std::io::Read;
use std::time::{Instant};
use std::sync::atomic::{AtomicBool, Ordering, ATOMIC_BOOL_INIT};

extern crate nix;

use nix::sys::signal;

extern crate rand;

use rand::{StdRng, SeedableRng};
use rand::distributions::{IndependentSample, Range};

extern crate byteorder;

use byteorder::{ReadBytesExt, BigEndian};

static EXIT_NOW: AtomicBool = ATOMIC_BOOL_INIT;

extern fn early_exit(_: i32) {
	if EXIT_NOW.load(Ordering::Relaxed) {
		panic!("second ctrl+c");
	}
	EXIT_NOW.store(true, Ordering::Relaxed);
}

// Loads image data as:
// u32 image_x size
// u32 image_y size
// Vec< array of images
//	Vec<u8 individual image data
fn load_images_from_file(filename: &str) -> Option<(u32,u32,Vec<Vec<u8>>)> {

	let mut training_images_file = File::open(filename).expect(&format!("Could not open {}", filename));

	let images_magic = training_images_file.read_u32::<BigEndian>().unwrap();
	if images_magic != 0x00000803 {

		println!("Corrupt {} file, magic was {}", filename, images_magic);
		return None;
	}

	let num_training_images = training_images_file.read_u32::<BigEndian>().unwrap();
	let image_x = training_images_file.read_u32::<BigEndian>().unwrap();
	let image_y = training_images_file.read_u32::<BigEndian>().unwrap();
	println!("Found {} training images {}x{}", num_training_images, image_x, image_y);

	let mut image_data: Vec<Vec<u8>> = vec![vec![0; (image_x * image_y) as usize]; num_training_images as usize];
	for item_index in 0..num_training_images {
		training_images_file.read_exact(&mut image_data[item_index as usize]).unwrap();
	}

	return Some((image_x, image_y, image_data));
}

// Loads the labels as an array of u8's
fn load_labels_from_file(filename: &str) -> Option<Vec<u8>> {

	let mut training_labels_file = File::open(filename).expect(&format!("Could not open {}", filename));

	let labels_magic = training_labels_file.read_u32::<BigEndian>().unwrap();
	if labels_magic != 0x00000801 {

		println!("Corrupt {} file, magic was {}", filename, labels_magic);
		return None;
	}

	let num_training_labels = training_labels_file.read_u32::<BigEndian>().unwrap();
	println!("Found {} training labels", num_training_labels);

	let mut labels: Vec<u8> = vec![0; num_training_labels as usize];
	for item_index in 0..num_training_labels {
		labels[item_index as usize] = training_labels_file.read_u8().unwrap();
	}

	return Some(labels);
}

// Activation functions
fn sigmoid(input: f32) -> f32 {
	return 1.0 / (1.0 + (-input).exp());
}

fn sigmoid_derivative(input: f32) -> f32 {
	return input * (1.0 - input);
}

fn relu(input: f32) -> f32 {
	return input.max(0.0);
}

fn relu_derivative(input: f32) -> f32 {
	return if input > 0.0 { 1.0 } else { 0.0 };
}

fn main() {

	let learning_rate: f32 = env::args().nth(1).expect("Expecting argument").parse::<f32>().ok().expect("Expecting f32");
	let seed_num: usize = env::args().nth(2).expect("Expecting argument").parse::<usize>().ok().expect("Expecting usize");

	// Run function on ctrl+c
	let sig_action = signal::SigAction::new(signal::SigHandler::Handler(early_exit), signal::SaFlags::empty(), signal::SigSet::empty());
	unsafe { signal::sigaction(signal::SIGINT, &sig_action).unwrap(); }
	unsafe { signal::sigaction(signal::SIGTERM, &sig_action).unwrap(); }
	let seed: &[_] = &[seed_num];
	let mut rng: StdRng = SeedableRng::from_seed(seed);

	let training_images = load_images_from_file("train-images-idx3-ubyte").unwrap();
	let image_x = training_images.0;
	let image_y = training_images.1;
	let num_training_images = training_images.2.len();

	let training_labels = load_labels_from_file("train-labels-idx1-ubyte").unwrap();
	let num_training_labels = training_labels.len();

	assert!(num_training_images == num_training_labels, "Num training images should be the same as num labels");

	let image_data: Vec<Vec<u8>> = training_images.2;
	let labels: Vec<u8> = training_labels;

	// Vec<: neurons
	// 	f32: neuron activation
	// 	Vec<: weights
	// 		f32: weight from each input neuron
	let mut layer1: Vec<(f32, Vec<f32>)> = vec![(0.0, vec![0.0; 28*28]); 128];
	let mut layer1_bias = vec![0.0; 128];
	let mut layer2: Vec<(f32, Vec<f32>)> = vec![(0.0, vec![0.0; 128]); 64];
	let mut layer2_bias = vec![0.0; 64];
	let mut layer3: Vec<(f32, Vec<f32>)> = vec![(0.0, vec![0.0; 64]); 10];
	let mut layer3_bias = vec![0.0; 10];

	// Initialise the weights with random values 0..1
	let weights_range = Range::new(0.0, 1.0);
	for b in &mut layer1_bias {
		*b = weights_range.ind_sample(&mut rng);
	}
	for n in &mut layer1 {
		for w in &mut n.1 {
			*w = weights_range.ind_sample(&mut rng);
		}
	}
	for b in &mut layer2_bias {
		*b = weights_range.ind_sample(&mut rng);
	}
	for n in &mut layer2 {
		for w in &mut n.1 {
			*w = weights_range.ind_sample(&mut rng);
		}
	}
	for b in &mut layer3_bias {
		*b = weights_range.ind_sample(&mut rng);
	}
	for n in &mut layer3 {
		for w in &mut n.1 {
			*w = weights_range.ind_sample(&mut rng);
		}
	}

	let mut training_correct_test_results = 0;
	let mut training_incorrect_test_results = 0;

	// Training
	// Continuous looping until the user cancels
	// On each loop we:
	//	set the input activations from the input image
	// 	get the average activation for that neuron
	//	loop through the neurons in the layer, setting the weights to
	//		learning_rate * input * err
	//	find the highest activation and check if it is correct
	let mut loop_counter = 0;
	let mut input = vec![0.0; 28*28];

	let start_training = Instant::now();

	loop {

		training_correct_test_results = 0;
		training_incorrect_test_results = 0;

		for item_index in 0..num_training_images {

			// Get input
			let label = labels[item_index as usize];
			assert!(image_data[item_index as usize].len() as u32 == image_x * image_y);
			for (i,n) in image_data[item_index as usize].iter().enumerate() {
				input[i] = f32::from(*n) / 256.0;
			}

			// Feed forward
			for (i,n) in &mut layer1.iter_mut().enumerate() {
				n.0 = layer1_bias[i];
				for (j,w) in &mut n.1.iter().enumerate() {
					n.0 += input[j] * w;
				}
				n.0 /= n.1.len() as f32;
				n.0 = relu(n.0);
			}

			for (i,n) in &mut layer2.iter_mut().enumerate() {
				n.0 = layer2_bias[i];
				for (j,w) in &mut n.1.iter().enumerate() {
					n.0 += layer1[j].0 * w;
				}
				n.0 /= n.1.len() as f32;
				n.0 = relu(n.0);
			}

			for (i,n) in &mut layer3.iter_mut().enumerate() {
				n.0 = layer3_bias[i];
				for (j,w) in &mut n.1.iter().enumerate() {
					n.0 += layer2[j].0 * w;
				}
				n.0 /= n.1.len() as f32;
				n.0 = relu(n.0);
			}

			let mut layer3_errorsum = vec![0.0; layer2.len()];
			let mut layer2_errorsum = vec![0.0; layer1.len()];

			// Feed backwards using errors and adjusting weights
			for (i,n) in &mut layer3.iter_mut().enumerate() {

				let err = if i == label as usize { 1.0 } else { 0.0 } - n.0;
				let error_signal = err * relu_derivative(n.0);

				for (j,w) in &mut n.1.iter_mut().enumerate() {

					*w += learning_rate * layer2[j].0 * error_signal;
					layer3_errorsum[j] += error_signal * *w;
				}

				layer3_bias[i] += learning_rate * error_signal;
			}

			for (i,n) in &mut layer2.iter_mut().enumerate() {

				let mut errsum = layer3_errorsum[i];
				errsum *= relu_derivative(n.0);

				for (j,w) in &mut n.1.iter_mut().enumerate() {

					*w += learning_rate * layer1[j].0 * errsum;
					layer2_errorsum[j] += errsum * *w;
				}

				layer2_bias[i] += learning_rate * errsum;
			}

			for (i,n) in &mut layer1.iter_mut().enumerate() {

				let mut errsum = layer2_errorsum[i];

				errsum *= relu_derivative(n.0);

				for (j,w) in &mut n.1.iter_mut().enumerate() {

					*w += learning_rate * input[j] * errsum;
				}

				layer1_bias[i] += learning_rate * errsum;
			}

			// Find highest activated neuron "soft" max
			let mut highest_activation = (-10000.0, 0);
			for (i, n) in layer3.iter().enumerate() {
				if n.0 > highest_activation.0 {
					highest_activation.0 = n.0;
					highest_activation.1 = i;
				}
			}

			assert!(highest_activation.1 < 10);
			if highest_activation.1 as u8 == label {
				training_correct_test_results += 1;
			} else {
				training_incorrect_test_results += 1;
			}
		}
		if EXIT_NOW.load(Ordering::Relaxed) {
			break;
		}
		loop_counter += 1;

		println!("Tests passed: {}, tests failed: {} ({}%), {} epochs", training_correct_test_results,
			training_incorrect_test_results,
			training_correct_test_results as f32 /
				((training_correct_test_results + training_incorrect_test_results) as f32) * 100.0, loop_counter);
	}

	let training_time = start_training.elapsed();

	println!("Using testing data set");

	let testing_images = load_images_from_file("t10k-images-idx3-ubyte").unwrap();
	let image_x = testing_images.0;
	let image_y = testing_images.1;
	let num_testing_images = testing_images.2.len();

	let testing_labels = load_labels_from_file("t10k-labels-idx1-ubyte").unwrap();
	let num_testing_labels = testing_labels.len();

	assert!(num_testing_images == num_testing_labels, "Num testing images should be the same as num labels");

	let image_data: Vec<Vec<u8>> = testing_images.2;
	let labels: Vec<u8> = testing_labels;

	let mut correct_test_results = 0;
	let mut incorrect_test_results = 0;

	let start_testing = Instant::now();

	// Testing trained network
	// Loop through the testing images
	//	print out the number
	//	set the inputs from the image
	// 	activate the neurons
	//	find the highest activation neuron and test against what the result should be
	for item_index in 0..num_testing_images {

		// Get input
		let label = labels[item_index as usize];
		assert!(image_data[item_index as usize].len() as u32 == image_x * image_y);
		for (i,n) in image_data[item_index as usize].iter().enumerate() {
			input[i] = f32::from(*n) / 256.0;
		}


		// Feed forward
		for (i,n) in &mut layer1.iter_mut().enumerate() {
			n.0 = layer1_bias[i];
			for (j,w) in &mut n.1.iter().enumerate() {
				n.0 += input[j] * w;
			}
			n.0 /= n.1.len() as f32;
			n.0 = relu(n.0);
		}

		for (i,n) in &mut layer2.iter_mut().enumerate() {
			n.0 = layer2_bias[i];
			for (j,w) in &mut n.1.iter().enumerate() {
				n.0 += layer1[j].0 * w;
			}
			n.0 /= n.1.len() as f32;
			n.0 = relu(n.0);
		}

		for (i,n) in &mut layer3.iter_mut().enumerate() {
			n.0 = layer3_bias[i];
			for (j,w) in &mut n.1.iter().enumerate() {
				n.0 += layer2[j].0 * w;
			}
			n.0 /= n.1.len() as f32;
			n.0 = relu(n.0);
		}

		// Find highest activated neuron "soft" max
		let mut highest_activation = (-10000.0, 0);
		for (i, n) in layer3.iter().enumerate() {
			if n.0 > highest_activation.0 {
				highest_activation.0 = n.0;
				highest_activation.1 = i;
			}
		}

		assert!(highest_activation.1 < 10);
		if highest_activation.1 as u8 == label {
			correct_test_results += 1;
			println!("Test passed {} == {}", highest_activation.1 as u8, label);
		} else {
			incorrect_test_results += 1;
			println!("Test failed {} != {}", highest_activation.1 as u8, label);

			for y in 0..image_y {
				for x in 0..image_x {
					let value = image_data[item_index as usize][(y * image_x + x) as usize];
					let output = match value {
						0 => " ",
						1...128 => "░",
						129...250 => "▒",
						_ => "▓"
					};
					print!("{}", output);
				}
				print!("\n");
			}
		}
	}

	let testing_time = start_testing.elapsed();

	println!("Tests passed: {}, tests failed: {} ({}%) with {} epochs training ({}% training)", correct_test_results,
		incorrect_test_results,
		correct_test_results as f32 / ((correct_test_results + incorrect_test_results) as f32) * 100.0, loop_counter,
		training_correct_test_results as f32 / ((training_correct_test_results + training_incorrect_test_results) as f32) * 100.0);
	println!("Training time {}s, testing time {}s",
		(training_time.as_secs() as f64) + (training_time.subsec_nanos() as f64 / 1000_000_000.0),
		(testing_time.as_secs() as f64) + (testing_time.subsec_nanos() as f64 / 1000_000_000.0));
}
