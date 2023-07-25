
import os,json
import torch
from tqdm import tqdm
from utils.data_loader import *
from encoder.modeling_chart2vec import *
# from encoder.modeling_chart2vec_no_pooling import *
# from encoder.modeling_chart2vec_words_pooling import *
# from encoder.modeling_chart2vec_no_pos import *
# from encoder.modeling_chart2vec_no_fc import *
# from encoder.modeling_chart2vec_no_schema import *
# from encoder.modeling_chart2vec_no_semantic import *
# from encoder.modeling_chart2vec_word_max_pooling import *
# from encoder.modeling_chart2vec_words_max_pooling import *


def test_chart2vec():
    combine_name="chart2vec_base_128"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    path = os.path.dirname(__file__)

    model_1="chart2vec_base_128-2023-07-24 21-11-47/2023-07-25 01-13-44-97/"
    model_path = model_1+"chart2vec_base.pth"
    model_save_path = os.path.join(path, "models", model_path)
    fact_path = os.path.join(path, "dataset/testing_data.json")
    with open(fact_path) as f:
        fact_data = json.load(f)

    chart2vec_model = Chart2Vec().to(device)
    state_dict = torch.load(model_save_path, map_location=torch.device(device))
    chart2vec_model.load_state_dict(state_dict['model'])
    chart2vec_model.eval()

    input = pad_w2v_convert_fact_to_input(fact_data)
    batch_struct_one_hot = input["batch_struct_one_hot"]
    batch_indexed_tokens = input["batch_indexed_tokens"]
    batch_pos = input["batch_pos"]
    input_len=len(batch_struct_one_hot)
    
    result_dict={}
    for i in tqdm(range(input_len), desc="data_num"):
        output_vec = chart2vec_model(
            np.array([batch_struct_one_hot[i]]), np.array([batch_indexed_tokens[i]]), np.array([batch_pos[i]]))

        result_dict[fact_data[i]["fact_id"]]=output_vec[0].tolist()

    res_path=os.path.join(path,"results", combine_name+"-result.json")
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
    print("create "+combine_name+"_result.json file success!")


if __name__ == "__main__":
    test_chart2vec()