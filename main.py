from torch import true_divide
from train_test import train
import json
import argparse


if __name__ == "__main__":    

    parser = argparse.ArgumentParser()
    
    parser.add_argument("-task",type=str, default="BRCA",help="available tasks: BRCA, ROSMAP, KIPAN, LGG")
    parser.add_argument("-test_only",type=bool, default=False,help="test or train from scratch")
    parser.add_argument("-uni_modality",type=bool, default=True,help="whether use uni_modality data")
    parser.add_argument("-dual_modality",type=bool, default=True,help="whether use dual_modality data")
    parser.add_argument("-triple_modality",type=bool, default=True,help="whether use triple_modality data")
    
    args = parser.parse_args()

    data_folder_path="../dataset/"+args.task
    testonly = args.test_only
    modelpath = 'checkpoints'
    uni_data=args.uni_modality
    dual_data=args.dual_modality
    triple_data=args.triple_modality
    result=train(data_folder_path, modelpath, testonly,uni_data,dual_data,triple_data)
        
    with open(f"{args.task}_results.json","w") as f:
        json.dump(result,f)