/*
 * Copyright 2015 Simone Filice and Giuseppe Castellucci and Danilo Croce and Roberto Basili
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

//package it.uniroma2.sag.kelp.examples.demo.clustering;

import java.io.File;

import it.uniroma2.sag.kelp.data.clustering.Cluster;
import it.uniroma2.sag.kelp.data.clustering.ClusterExample;
import it.uniroma2.sag.kelp.data.clustering.ClusterList;
import it.uniroma2.sag.kelp.data.dataset.SimpleDataset;
import it.uniroma2.sag.kelp.kernel.Kernel;
import it.uniroma2.sag.kelp.kernel.cache.KernelCache;
import it.uniroma2.sag.kelp.kernel.standard.NormalizationKernel;
import it.uniroma2.sag.kelp.kernel.tree.PartialTreeKernel;
import it.uniroma2.sag.kelp.kernel.tree.SubSetTreeKernel;
import it.uniroma2.sag.kelp.kernel.cache.FixSizeKernelCache;
import it.uniroma2.sag.kelp.kernel.vector.LinearKernel;
import it.uniroma2.sag.kelp.learningalgorithm.clustering.kernelbasedkmeans.KernelBasedKMeansEngine;
import it.uniroma2.sag.kelp.learningalgorithm.clustering.kernelbasedkmeans.KernelBasedKMeansExample;
import it.uniroma2.sag.kelp.utils.JacksonSerializerWrapper;
import it.uniroma2.sag.kelp.utils.ObjectSerializer;
import it.uniroma2.sag.kelp.utils.evaluation.ClusteringEvaluator;
import it.uniroma2.sag.kelp.data.example.Example;
import java.util.HashMap;

/**
 * This class contains an example of the usage of the Kernel-based clustering.
 * The clustering engine implements the Kernel Based K-means described in:
 * 
 * Brian Kulis, Sugato Basu, Inderjit Dhillon, and Raymond Mooney.
 * Semi-supervised graph clustering: a kernel approach. Machine Learning,
 * 74(1):1-22, January 2009.
 * 
 * The source code are provided by Kelp in the kernel-clustering project.
 * 
 * The dataset used in this example is the IRIS dataset. It can be downloaded
 * from: https://archive.ics.uci.edu/ml/datasets/Iris
 * 
 * @author Danilo Croce
 * 
 */
public class KernelBasedFrameSimilarity {

	private Kernel kernel;
	public static void main(String[] args) throws Exception {
		KernelBasedFrameSimilarity kernelBasedFrameSimilarity = new KernelBasedFrameSimilarity(args[0],args[1],args[2],args[3]);
	}

	public KernelBasedFrameSimilarity(String file_path_1, String index_1, String file_path_2, String index_2){
		// Load the dataset from file_path_1
		SimpleDataset dataset_1 = new SimpleDataset();
		try {
			dataset_1.populate(file_path_1);
		} catch (Exception exception) {}
		// Load the dataset from file_path_2
		SimpleDataset dataset_2 = new SimpleDataset();
		try {
			dataset_2.populate(file_path_2);
		} catch (Exception exception) {}

		// The representation considered from the algorithm
		String representationName = "frame_syntaxtree";

		// Set kernel dataset_1
		// Initialize the kernel function
		this.kernel = new PartialTreeKernel(0.4f, 0.4f, 0.5f, representationName);
		this.kernel = new NormalizationKernel(kernel);
		// Initialize the cache
		KernelCache kernelCache = new FixSizeKernelCache(dataset_1.getNumberOfExamples());
		kernel.setKernelCache(kernelCache);

		int idx_1 = new Integer(index_1);
		int idx_2 = new Integer(index_2);
		float similarity = this.calculateSimilarity(dataset_1.getExample(idx_1), dataset_2.getExample(idx_2));
		System.out.print(similarity);
	}
	public float evaluateKernel(Example e1, Example e2) {
		return this.kernel.innerProduct(e1, e2);
	}
	private float calculateSimilarity(Example example1, Example example2) {
		return (float) this.evaluateKernel(example1, example2);
	  }
}