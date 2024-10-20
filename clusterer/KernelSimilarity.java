import java.io.PrintStream;

import it.uniroma2.sag.kelp.data.dataset.SimpleDataset;
import it.uniroma2.sag.kelp.data.example.Example;
import it.uniroma2.sag.kelp.kernel.Kernel;
import it.uniroma2.sag.kelp.kernel.standard.NormalizationKernel;
import it.uniroma2.sag.kelp.kernel.tree.PartialTreeKernel;

public class KernelSimilarity {

	public static void main(String[] args) throws Exception {

		if (args.length != 4) {
			System.err.println("ERROR: query_file_path medoid_file_path output_file_path rep_name");
			System.exit(0);
		}

		String queryFilePath = args[0];

		String MedoidFilePath = args[1];

		String outputFilePath = args[2];

		String representationName = args[3];

		Kernel kernel = new PartialTreeKernel(0.4f, 0.4f, 0.5f, representationName);
		kernel = new NormalizationKernel(kernel);

		SimpleDataset querydataset = new SimpleDataset();
		querydataset.populate(queryFilePath);

		SimpleDataset medoidDataset = new SimpleDataset();
		medoidDataset.populate(MedoidFilePath);

		PrintStream ps = new PrintStream(outputFilePath);

		for (Example q : querydataset.getExamples()) {
			StringBuilder sb = new StringBuilder();
			for (Example m : medoidDataset.getExamples()) {
				float k = kernel.innerProduct(q, m);
				sb.append(k + " ");
			}
			ps.println(sb.toString().trim());

		}

		ps.close();

	}

}
