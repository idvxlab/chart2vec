import os,json
import time
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils.data_loader import *

from encoder.modeling_chart2vec import Chart2Vec
# from encoder.modeling_chart2vec_no_pooling import Chart2Vec 
# from encoder.modeling_chart2vec_words_pooling import Chart2Vec 
# from encoder.modeling_chart2vec_no_fc import Chart2Vec 
# from encoder.modeling_chart2vec_no_pos import Chart2Vec
# from encoder.modeling_chart2vec_no_schema import Chart2Vec
# from encoder.modeling_chart2vec_no_semantic import Chart2Vec
# from encoder.modeling_chart2vec_word_max_pooling import Chart2Vec
# from encoder.modeling_chart2vec_words_max_pooling import Chart2Vec
from encoder.ImprovedQuadrupletLoss import *



def train_chart2vec():
    combine_name = "modules-test"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # get current time, used for create folders
    current_time = time.strftime(
        "%Y-%m-%d %H-%M-%S", time.localtime(time.time()))

    # set store path
    path = os.path.dirname(__file__)
    model_save_path = os.path.join(path, "models")
    training_model_folder_path = os.path.join(
        model_save_path, combine_name+"-"+current_time)
    os.mkdir(training_model_folder_path)

    loss_path = os.path.join(path, "tensorboard")
    tensorboard_folder_path = os.path.join(
        loss_path, combine_name+current_time)
    os.mkdir(tensorboard_folder_path)
    tensorboard_writer = SummaryWriter(tensorboard_folder_path)
    
    data_path = os.path.join(path, "dataset", "training_data.json")
    with open(data_path) as f:
        train_data = json.load(f)

    # set training params
    epochs = 100
    batch_size = 256
    total_sample_num = len(train_data)
    batch_num = total_sample_num // batch_size
    best_loss = 1e7
    lr = 1e-3

    # ---------when use quadruplet data for training------------
    batch_gen = quadruplet_batch_generator_word2vec(train_data, batch_size)
    loss_fc = ImprovedQuadrupletLoss().to(device)
    chart_emb = Chart2Vec().to(device)
   
     # 过滤掉requires_grad = False的参数
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, chart_emb.parameters()), lr=lr)
    for i in tqdm(range(epochs), desc="epochs"):
        total_loss = 0.0
        for j in tqdm(range(batch_num), desc="batch steps"):
            # ---------use quadruplet data for training------------
            batch_chart_x1, batch_chart_x2, batch_chart_x3, batch_chart_y1 = next(batch_gen)
            res_chart_x1 = chart_emb(
                batch_chart_x1["batch_struct_one_hot"], batch_chart_x1["batch_indexed_tokens"], batch_chart_x1["batch_pos"])
            res_chart_x2 = chart_emb(batch_chart_x2["batch_struct_one_hot"],
                                 batch_chart_x2["batch_indexed_tokens"], batch_chart_x2["batch_pos"])
            res_chart_x3 = chart_emb(batch_chart_x3["batch_struct_one_hot"],
                                 batch_chart_x3["batch_indexed_tokens"], batch_chart_x3["batch_pos"])
            res_chart_y1= chart_emb(batch_chart_y1["batch_struct_one_hot"],
                                batch_chart_y1["batch_indexed_tokens"], batch_chart_y1["batch_pos"])
            chart2vec_loss = loss_fc(res_chart_x1,  res_chart_x2, res_chart_x3, res_chart_y1)

            total_loss += chart2vec_loss.cpu().item()
            chart2vec_loss.backward()
            optimizer.step()
            chart_emb.zero_grad()
            tensorboard_writer.add_scalar(
                "mean_loss", chart2vec_loss.item(), j)
            print("mean_loss:", chart2vec_loss.item())

        tensorboard_writer.add_scalar("epoch_loss", total_loss, i)
        if total_loss < best_loss:
            best_loss = total_loss
            # create folder
            ttt = time.strftime("%Y-%m-%d %H-%M-%S",
                                time.localtime(time.time()))
            folder_path = os.path.join(
                training_model_folder_path, ttt+"-"+str(i))
            os.mkdir(folder_path)
            torch.save({'model': chart_emb.state_dict()},
                       os.path.join(folder_path, "chart2vec_base.pth"))


if __name__ == "__main__":
    train_chart2vec()