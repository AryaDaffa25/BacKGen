from converter.io import write_jsonl
import pandas as pd

def framecluster2jsonl(input_file_path,output_file_path,cluster_name,cluster_polarity,cluster_name_type="without_frame_label",return_mode="no_return"):
    cluster = pd.read_csv(input_file_path,header=None,delimiter="\t")
    dic_cluster = cluster.groupby(1)[2].apply(list).to_dict()
    list_cluster = []
    for key,val in dic_cluster.items():
        medoid_frame_label = val[0].split(' |BS:original_text|')[0]
        if cluster_name_type == "with_frame_label":
            cluster_id = f"{cluster_name}_{medoid_frame_label}_{key}"
        else:
            cluster_id = f"{cluster_name}_{cluster_polarity}_{key}"
        medoid_text_id = val[0].split('|BS:id| ')[1].split('|ES|')[0]
        medoid_original_text = val[0].split('|BS:original_text| ')[1].split('|ES|')[0]
        medoid_syntax_tree = val[0].split('|BT:frame_syntaxtree| ')[1].split('|ET|')[0]
        medoid_symbolic_tree = val[0].split('|BS:frame_symbolic| ')[1].split('|ES|')[0]
        list_frame_symbolic,list_text_source = [],[]
        for value in val:
            list_frame_symbolic.append(value.split('|BS:frame_symbolic| ')[1].split('|ES|')[0])
            list_text_source.append(value.split('|BS:original_text| ')[1].split('|ES|')[0])
        list_cluster.append({'cluster_id':cluster_id,
                            'polarity_label':cluster_polarity,
                            'medoid_frame_label':medoid_frame_label,
                            'medoid_text_id':medoid_text_id,
                            'medoid_original_text':medoid_original_text,
                            'medoid_syntax_tree':medoid_syntax_tree,
                            'medoid_symbolic_tree':medoid_symbolic_tree,
                            'list_frame_symbolic':list_frame_symbolic,
                            'list_text_source':list_text_source})
    write_jsonl(list_cluster,output_file_path)
    print(f"The .jsonl of frame cluster has been saved into {output_file_path}")
    if return_mode != "no_return":
        return list_cluster