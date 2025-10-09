# Import library
import json
import numpy as np

def write_jsonl(jsons, output_filename):
  with open(output_filename, "w") as f:
    for each_json in jsons:
      json.dump(each_json,f, default=lambda o: int(o) if isinstance(o, (np.integer,)) 
                else float(o) if isinstance(o, (np.floating,))
                else str(o))
      f.write("\n")

def read_jsonl(filename):
  result = []
  with open(filename, "r") as f:
    for line in f.readlines():
      result.append(json.loads(line))
  return result