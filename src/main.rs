
#![feature(exclusive_range_pattern)]


extern crate nix;

use nix::sys::signal;
use std::sync::atomic::{AtomicBool, Ordering, ATOMIC_BOOL_INIT};


extern crate rand;

use std::env;
use rand::{thread_rng, sample};
use rand::distributions::{IndependentSample, Range};

extern crate byteorder;
use byteorder::{ReadBytesExt, LittleEndian, BigEndian};

use std::fs::File;
use std::io::Read;

#[derive(Debug, Clone)]
struct Neuron {
	activation: f32,
	weights: Vec<f32>,
	bias: f32
}

impl Neuron {
	fn new(num_connections: usize) -> Neuron {
		Neuron {activation: 0.0, weights: vec![0.0; num_connections], bias: 0.0}
	}
}

#[derive(Debug, Clone)]
struct Layer {
	neurons: Vec<Neuron>
}

impl Layer {
	fn new(num_neurons: usize, next_layer_size: usize) -> Layer {
		Layer {neurons: vec![Neuron::new(next_layer_size); num_neurons]}
	}
	fn propogate(&self, to_layer: &mut Layer) {

		for dest_neuron in &mut to_layer.neurons {

			dest_neuron.activation = 0.0;
		}

		for neuron in &self.neurons {

			for (index, weight) in neuron.weights.iter().enumerate() {

				to_layer.neurons[index].activation += *weight * neuron.activation;
				println!("a += w * a: {} += {} * {}", to_layer.neurons[index].activation, *weight, neuron.activation);
			}
		}

		for neuron in &mut to_layer.neurons {

			neuron.activation = sigmoid(neuron.activation + neuron.bias);
			print!("{} ", neuron.activation);
		}
		print!("\n");
	}
	fn print(&self) {

		print!("{} neurons: ", self.neurons.len());
		for neuron in &self.neurons {
			print!("{:.1} ", neuron.activation);
		}
		print!("\n");
	}
}

fn sigmoid(input: f32) -> f32 {

	return 1.0 / (1.0 + (-input).exp());
}

fn sigmoid_derivative(input: f32) -> f32 {

	return input * (1.0 - input);
}

static EXIT_NOW: AtomicBool = ATOMIC_BOOL_INIT;

extern fn early_exit(_: i32) {
	if EXIT_NOW.load(Ordering::Relaxed) {
		panic!("second ctrl+c");
	}
	EXIT_NOW.store(true, Ordering::Relaxed);
}

fn main() {

	let training_images_filename = "train-images-idx3-ubyte";
	let training_labels_filename = "train-labels-idx1-ubyte";

	println!("Loading training images");

	let mut training_images_file = File::open(training_images_filename).expect(&format!("Could not open {}", training_images_filename));

	{
		let images_magic = training_images_file.read_u32::<BigEndian>().unwrap();
		if images_magic != 0x00000803 {

			panic!("Corrupt {} file, magic was {}", training_images_filename, images_magic);
		}
	}

	let num_training_images = training_images_file.read_u32::<BigEndian>().unwrap();
	let image_x = training_images_file.read_u32::<BigEndian>().unwrap();
	let image_y = training_images_file.read_u32::<BigEndian>().unwrap();

	println!("Found {} training images {}x{}", num_training_images, image_x, image_y);

	println!("Loading training labels");

	let mut training_labels_file = File::open(training_labels_filename).expect(&format!("Could not open {}", training_labels_filename));

	{
		let labels_magic = training_labels_file.read_u32::<BigEndian>().unwrap();
		if labels_magic != 0x00000801 {

			panic!("Corrupt {} file, magic was {}", training_labels_filename, labels_magic);
		}
	}

	let num_training_labels = training_labels_file.read_u32::<BigEndian>().unwrap();

	println!("Found {} training labels", num_training_labels);

	assert!(num_training_images == num_training_labels, "Num training images should be the same as num labels");

	let mut image_data: Vec<Vec<u8>> = vec![vec![0; (image_x * image_y) as usize]; num_training_images as usize];
	let mut labels: Vec<u8> = vec![0; num_training_images as usize];

	for item_index in 0..num_training_images {

		let result = training_images_file.read_exact(&mut image_data[item_index as usize]).unwrap();
		labels[item_index as usize] = training_labels_file.read_u8().unwrap();
	}


	let mut rng = thread_rng();

	//let learning_rate: f64 = env::args().nth(1).expect("Expecting argument").parse::<f64>().ok().expect("Expecting f64");
	//let num_iterations: i32 = env::args().nth(2).expect("Expecting argument").parse::<i32>().ok().expect("Expecting i32");

	// Network
	let mut input_layer: Layer = Layer::new((image_x * image_y) as usize, 16);

	let mut layer1: Layer = Layer::new(16, 16);
	let mut layer2: Layer = Layer::new(16, 10);
	let mut output_layer: Layer = Layer::new(10, 0);


	// Init weights & biases
	let weights_range = Range::new(0.0, 1.0);
	for neuron in &mut input_layer.neurons {
		for weight in &mut neuron.weights {
			*weight = weights_range.ind_sample(&mut rng);
		}
	}
	for neuron in &mut layer1.neurons {
		for weight in &mut neuron.weights {
			*weight = weights_range.ind_sample(&mut rng);
		}
		neuron.bias = weights_range.ind_sample(&mut rng);
	}
	for neuron in &mut layer2.neurons {
		for weight in &mut neuron.weights {
			*weight = weights_range.ind_sample(&mut rng);
		}
		neuron.bias = weights_range.ind_sample(&mut rng);
	}
	for neuron in &mut output_layer.neurons {
		neuron.bias = weights_range.ind_sample(&mut rng);
	}

	println!("input weights:");
	for neuron in &mut input_layer.neurons {
		for i in 0..10 {
			print!("{:.1} ", neuron.weights[i]);
		}
		print!("\n");
	}
	println!("layer2 weights:");
	for neuron in &mut layer2.neurons {
		for i in 0..10 {
			print!("{:.1} ", neuron.weights[i]);
		}
		print!("\n");
	}
    // define an action to take (the key here is 'signal::SigHandler::Handler(early_exit)'
    //    early_exit being the function we defined above
    let sig_action = signal::SigAction::new(signal::SigHandler::Handler(early_exit),
                                            signal::SaFlags::empty(),
                                            signal::SigSet::empty());
    // use sig_action for SIGINT
    unsafe { signal::sigaction(signal::SIGINT, &sig_action); }
    // use sig_action for SIGTERM
    unsafe { signal::sigaction(signal::SIGTERM, &sig_action); }



	// Training
	let mut correct_test_results = 0;
	let mut incorrect_test_results = 0;

	let mut loop_counter = 0;
	loop {
		for item_index in 0..num_training_images {
			if item_index != 0 {
				continue;
			}

			let label = labels[item_index as usize];

			assert!(image_data[item_index as usize].len() as u32 == image_x * image_y);

			for y in 0..image_y {

				for x in 0..image_x {

					let value = image_data[item_index as usize][(y * image_x + x) as usize];

					let output = match value {

						0 => " ",
						1..128 => "░",
						129..254 => "▒",
						_ => "▓"
					};
	//				print!("{}", output);

					input_layer.neurons[(y * image_x + x) as usize].activation = f32::from(value);
				}
	//			print!("\n");
			}

			if item_index == 0 {
				println!("pre propogation");
				layer1.print();
				layer2.print();
				output_layer.print();
			}

			// Run network
			input_layer.propogate(&mut layer1);
			layer1.propogate(&mut layer2);
			layer2.propogate(&mut output_layer);

			if item_index == 0 {
				println!("post propogation");
				layer1.print();
				layer2.print();
				output_layer.print();
			}

			// Layer 2 error
			let mut l2_error: Vec<f32> = vec![0.0; 10];
			for i in 0..10 {
				l2_error[i] = (if i == label as usize { 1.0 } else { 0.0 }) - output_layer.neurons[i].activation;
			}
			let mut l2_delta: Vec<f32> = vec![0.0; 10];
			for i in 0..10 {
				l2_delta[i] = l2_error[i] * sigmoid_derivative(output_layer.neurons[i].activation);
				println!("l2d = l2e * sd(a): {} = {} * {}", l2_delta[i], l2_error[i], sigmoid_derivative(output_layer.neurons[i].activation));
			}
			println!("layer2 weights");
			for neuron in &mut layer2.neurons {
				for i in 0..10 {
					neuron.weights[i] *= l2_delta[i];
				}
			}


			let mut highest_activation = (10000.0, 0);
			for (index, neuron) in output_layer.neurons.iter().enumerate() {

				if neuron.activation > highest_activation.0 {

					highest_activation.0 = neuron.activation;
					highest_activation.1 = index;
				}
			}

			assert!(highest_activation.1 < 10);

			if highest_activation.1 as u8 == label {

				correct_test_results += 1;
				//println!("Test passed {} == {}", highest_activation.1 as u8, label);
			} else {

				incorrect_test_results += 1;
				//println!("Test failed {} != {}", highest_activation.1 as u8, label);
			}
		}
		if EXIT_NOW.load(Ordering::Relaxed) {
			break;
		}
		loop_counter += 1;

		println!("Tests passed: {}, tests failed: {} ({}%)", correct_test_results, incorrect_test_results, correct_test_results as f32 / ((correct_test_results + incorrect_test_results) as f32) * 100.0);
		for neuron in &mut layer2.neurons {
			for i in 0..10 {
				print!("{:.1} ", neuron.weights[i]);
			}
			print!("\n");
		}
	}

	println!("Tests passed: {}, tests failed: {} ({}%)", correct_test_results, incorrect_test_results, correct_test_results as f32 / ((correct_test_results + incorrect_test_results) as f32) * 100.0);
}
