import subprocess

def calculate_similarity(lib_folder_path,input_file_path_1,syntaxtree_idx_1,input_file_path_2,syntaxtree_idx_2):
    run_path = f"cd {lib_folder_path} && java -cp .:./lib/kelp-additional-algorithms-2.2.2.jar:lib/kelp-core-2.2.2.jar:lib/kelp-full2.0.2.jar KernelBasedFrameSimilarity {input_file_path_1} {syntaxtree_idx_1} {input_file_path_2} {syntaxtree_idx_2}"
    return float(subprocess.Popen(run_path, shell=True, stdout=subprocess.PIPE).stdout.read())