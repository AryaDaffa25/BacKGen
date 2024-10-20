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
import java.util.Random;

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
import it.uniroma2.sag.kelp.data.example.SimpleExample;

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
public class KernelBasedFrameClustering {

	public static void main(String[] args) throws Exception {

		if (args.length != 3) {
			System.err
					.println("Usage: file_path num_of_clusters num_of_iteration");
			return;
		}

		// Number of clusters computed by the Kernel-based K-means algorithm
		int K = new Integer(args[1]);
		// Number of iteration of the Kernel-based K-means algorithm
		int tMax = new Integer(args[2]);
		// Load the dataset
		SimpleDataset dataset = new SimpleDataset();
		String file_path = args[0];
		dataset.populate(file_path);
		dataset.shuffleExamples(new Random(0));

		// The representation considered from the algorithm
		String representationName = "frame_syntaxtree";

		// Initialize the kernel function
		//Kernel kernel = new SubSetTreeKernel(representationName);
		Kernel kernel = new PartialTreeKernel(0.4f, 0.4f, 0.5f, representationName);
		kernel = new NormalizationKernel(kernel);

		// Initialize the cache
		KernelCache kernelCache = new FixSizeKernelCache(dataset.getNumberOfExamples());
		kernel.setKernelCache(kernelCache);

		// Initializing the clustering engine
		KernelBasedKMeansEngine clusteringEngine = new KernelBasedKMeansEngine(kernel, K, tMax);

		// Example of serialization of the engine via JSON
		ObjectSerializer serializer = new JacksonSerializerWrapper();
		// System.out.println(serializer.writeValueAsString(clusteringEngine));

		// Run the clustering
		ClusterList clusterList = clusteringEngine.cluster(dataset);

		// System.out.println("\n==================");
		// System.out.println("Resulting clusters");
		// System.out.println("==================\n");
		// Writing the resulting clusters and cluster members
		for (Cluster cluster : clusterList) {
			for (ClusterExample clusterMember : cluster.getExamples()) {
				float dist = ((KernelBasedKMeansExample) clusterMember).getDist();
				System.out.println(dist + "\t" + cluster.getLabel() + "\t" + clusterMember.getExample());
			}
			// System.out.println();
		}

		float avgExs = 0;
		int numSingleton = 0;
		for (Cluster cluster : clusterList) {
			float numberOfElementInTheCluster = cluster.getExamples().size();
			// System.out.println();
			avgExs += numberOfElementInTheCluster;

			if(cluster.getExamples().size()==1){
				System.err.println("Warning with singleton " + cluster.getExamples().get(0).getExample() );
				numSingleton++;
			}
		}
		avgExs/=(float) clusterList.size();
		System.err.println("avgExs:\t" + avgExs);
		System.err.println("numSingleton:\t" + numSingleton);
		System.err.println(ClusteringEvaluator.getStatistics(clusterList));

		System.exit(0);
		/*
		 * Kernel here acts as a Similarity function. If the kernel is normalized K(t1, t2)=1 when t1==t2 and 0 when they do not share ANY subfragment  
		 */
		SimpleExample queryExample = null;
		for (Cluster cluster : clusterList) {
			if(cluster.getExamples().size()==0)
			{
				continue;
			}
			
			//This is the medoid!!! 
			SimpleExample medoid = (SimpleExample) (cluster.getExamples().get(0).getExample());

			float sim = kernel.innerProduct(queryExample, medoid);
			// System.out.println();
		}
	}
}