"""
    Author: ChYing
    training the chart2vec model
"""
import os,json
import time
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utils.data_loader import *
from encoder.modeling_chart2vec import Chart2Vec
from encoder.ImprovedQuadrupletLoss import *
from utils.constants import *

def train_chart2vec(model_save_path, training_data_file):
    """
        training the chart2vec model
        Inputs:
            `model_save_path`: The directory where model iterations are saved during the training process, the overall directory is located under models.
            `training_data_file`: Filename of the training data.
    """
    # set training params
    epochs = 50
    batch_size = 128
    best_loss = 1e7
    lr = 1e-2

    # get the path to the current file
    path = os.path.dirname(__file__)
    # get current time, used for create folders
    current_time = time.strftime("%Y-%m-%d %H-%M-%S", time.localtime(time.time()))
    # set the location where the model is saved
    training_model_folder_path = os.path.join(os.path.join(path, "models"), model_save_path+"-"+current_time)
    os.mkdir(training_model_folder_path)
    # use tensorboard to record model loss changes
    tensorboard_folder_path = os.path.join(os.path.join(path, "tensorboard"), model_save_path+"-"+current_time)
    os.mkdir(tensorboard_folder_path)
    tensorboard_writer = SummaryWriter(tensorboard_folder_path)
    # get the training data
    with open(os.path.join(path, training_data_file)) as f:
        train_data = json.load(f)

    # use quadruplet batch data for training
    batch_gen = quadruplet_batch_generator_word2vec(train_data, batch_size)
    loss_fc = ImprovedQuadrupletLoss().to(device)
    chart_emb = Chart2Vec().to(device)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, chart_emb.parameters()), lr=lr)
    
    for i in tqdm(range(epochs), desc="epochs"):
        total_loss = 0.0
        for j in tqdm(range(len(train_data)//batch_size), desc="batch steps"):
            # represent each chart in the quaternion separately as a vector
            batch_chart_x1, batch_chart_x2, batch_chart_x3, batch_chart_y1 = next(batch_gen)
            res_chart_x1 = get_chart2vec_embeddings(chart_emb,batch_chart_x1)
            res_chart_x2 = get_chart2vec_embeddings(chart_emb,batch_chart_x2)
            res_chart_x3 = get_chart2vec_embeddings(chart_emb,batch_chart_x3)
            res_chart_y1= get_chart2vec_embeddings(chart_emb,batch_chart_y1)
            # calculate the loss
            chart2vec_loss = loss_fc(res_chart_x1,  res_chart_x2, res_chart_x3, res_chart_y1)
            total_loss += chart2vec_loss.cpu().item()
            # backpropagation for optimization
            chart2vec_loss.backward()
            optimizer.step()
            chart_emb.zero_grad()
            tensorboard_writer.add_scalar("mean_loss", chart2vec_loss.item(), j)
            print("mean_loss:", chart2vec_loss.item())
        tensorboard_writer.add_scalar("epoch_loss", total_loss, i)

        # saving of models
        if total_loss < best_loss:
            best_loss = total_loss
            # create folder
            ttt = time.strftime("%Y-%m-%d %H-%M-%S", time.localtime(time.time()))
            folder_path = os.path.join(
                training_model_folder_path, ttt+"-"+str(i))
            os.mkdir(folder_path)
            torch.save({'model': chart_emb.state_dict()}, os.path.join(folder_path, "chart2vec_base.pth"))

def get_chart2vec_embeddings(chart2vec_model, chart_data):
    """
        representation of the charts into the input format according to the required parameters of the chart2vec model, thus transforming them into the vector form corresponding to the Chart2vec model.
        Inputs:
            `chart2vec_model`: Chart2Vec model.
            `chart_data`: Chart of one of the quaternions in the training data.
        Output:
            vectors
    """
    return chart2vec_model(chart_data["batch_struct_one_hot"], chart_data["batch_indexed_tokens"], chart_data["batch_pos"])

if __name__ == "__main__":
    train_chart2vec( model_save_path="chart2vec_model_128_50_0.01",
                     training_data_file="dataset/training_data.json")