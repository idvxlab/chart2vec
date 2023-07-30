"""
    Author: ChYing
    testing the chart2vec model
"""
import os,json
import torch
from tqdm import tqdm
from utils.data_loader import *
from utils.constants import *
from encoder.modeling_chart2vec import *

def test_chart2vec(test_results_save_path,trained_model_path,testing_data_file):
    """
        training the chart2vec model
        Inputs:
            `model_save_path`: The directory where model iterations are saved during the training process, the overall directory is located under models.
            `training_data_file`: Filename of the training data.
    """
    # get the path to the current file
    path = os.path.dirname(__file__)
    # get the testing data
    with open(os.path.join(path, testing_data_file)) as f:
        test_data = json.load(f)

    # load the model and the parameters of the trained model
    chart2vec_model = Chart2Vec().to(device)
    model_save_path = os.path.join(path, "models", trained_model_path+"chart2vec_base.pth")
    state_dict = torch.load(model_save_path, map_location=torch.device(device))
    chart2vec_model.load_state_dict(state_dict['model'])
    chart2vec_model.eval()

    # construct test data into the input format required by the model
    input = pad_w2v_convert_fact_to_input(test_data)
    batch_struct_one_hot = input["batch_struct_one_hot"]
    batch_indexed_tokens = input["batch_indexed_tokens"]
    batch_pos = input["batch_pos"]
    input_len=len(batch_struct_one_hot)
    
    # represent each chart in the test data as a vector and store it in the result_dict
    result_dict={}
    for i in tqdm(range(input_len), desc="data_num"):
        output_vec = chart2vec_model(
            np.array([batch_struct_one_hot[i]]), np.array([batch_indexed_tokens[i]]), np.array([batch_pos[i]]))
        result_dict[test_data[i]["fact_id"]]=output_vec[0].tolist()

    # save all chart vectors, write to file for saving
    res_path=os.path.join(path,"results", test_results_save_path+"-result.json")
    write_chart2vec_results(res_path, result_dict,test_results_save_path)

def write_chart2vec_results(res_path, result_dict,test_results_save_path):
    """
        write chart vectors to a file for saving
        Inputs:
            `res_path`: File saving path.
            `result_dict`: Chart vector storage.
            `test_results_save_path`: Storage location of the test dataset.
    """
    num=0
    with open(res_path, 'w') as json_file:
        json_file.write('{')
        for key in result_dict:
            num+=1
            json_file.write('\n')
            if(num<len(result_dict)):
                json_file.writelines('"' + str(key) + '": ' + str(result_dict[key])+",")
            else:
                json_file.writelines('"' + str(key) + '": ' + str(result_dict[key]))
        json_file.write('\n'+'}')
    print("create "+test_results_save_path+"_result.json file success!")

if __name__ == "__main__":
    test_chart2vec(test_results_save_path="chart2vec_test_results",
                   trained_model_path="chart2vec_base_batch128_struct16_epoch20+20_lr0.005-2023-07-28 20-18-40/2023-07-28 21-04-38-17/",
                   testing_data_file="dataset/testing_data.json")