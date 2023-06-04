import torch
from models.Resnet50_model import Model
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
import torch.nn as nn
import os
from copy import deepcopy
from torchvision import models
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import warnings
warnings.filterwarnings("ignore") 

def train(model, train_dataloader, optimizer, loss_function, device, writer, epoch):
        print("--Training Step--")
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        model.train()
        for i , (x, y) in enumerate(tqdm(train_dataloader)):
            x = x.to(device)
            y = y.to(device)
            
            
            outputs = model(x)
            _, predicted = torch.max(outputs, 1)
            loss = loss_function(outputs, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            #print(f"loss = {loss.item()}")
            total_predictions += y.size(0)
            correct_predictions += predicted.eq(y).sum().item()
        epoch_avg_loss = running_loss/len(train_dataloader)
        epoch_accuracy = correct_predictions / total_predictions * 100
        print(f"train loss = {epoch_avg_loss} , train_Accuracy = {epoch_accuracy}")
        writer.add_scalar("train loss", epoch_avg_loss, epoch)
        writer.add_scalar("train accuracy", epoch_accuracy, epoch)
        return model
def val(model, val_dataloader, loss_function, device, writer, epoch):
        print("--Validation Step--")
        model.eval()
        val_running_loss = 0.0
        val_correct_predictions = 0
        val_total_predictions = 0
        for j , (val_x, val_y) in enumerate(tqdm(val_dataloader)):
            x = x.to(device)
            y = y.to(device)
            
            with torch.inference_mode():
                val_outputs = model(val_x)
                _, val_predicted = torch.max(val_outputs, 1)
                val_loss = loss_function(val_outputs, val_y)
                val_running_loss += val_loss.item()
                val_total_predictions = len(val_outputs)
                val_total_predictions += val_y.size(0)
                val_correct_predictions += val_predicted.eq(val_y).sum().item()
        val_epoch_avg_loss = (val_running_loss/float(j+1))
        val_epoch_accuracy = val_correct_predictions / val_total_predictions *100.0
        print(f"val loss = {val_epoch_avg_loss} , VAL_Accuracy = {val_epoch_accuracy}")
        writer.add_scalar("val loss", val_epoch_avg_loss, epoch)
        writer.add_scalar("val accuracy", val_epoch_accuracy, epoch)
def main():
    writer = SummaryWriter("runs/resnet50 training attempt 01")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Resize((224,224))])
    # Load the pre-trained ResNet50 model
    model = Model().to(device=device)
    loss_function = nn.CrossEntropyLoss().to(device=device)

    BATCH_SIZE = 16
    TRAIN_RATIO = 0.8
    VAL_RATIO = 1.0 - TRAIN_RATIO
    LEARNING_RATE = 0.001
    EPOCHS = 10
    SAVE_PATH = './saved_models/resnet50_attempt_01'
    #NUM_CLASSES = 90

    # Set up the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    dataset = ImageFolder('./data/animals/animals/', transform= transform)
    train_set, val_set = torch.utils.data.random_split(dataset, [TRAIN_RATIO, VAL_RATIO])

    train_dataloader = DataLoader(dataset=train_set, batch_size= BATCH_SIZE, shuffle= True, num_workers= 2)
    val_dataloader = DataLoader(dataset= val_set, batch_size= BATCH_SIZE, shuffle= False, num_workers= 1)

    for epoch in range(EPOCHS):
        model = train(model = model, train_dataloader= train_dataloader, optimizer = optimizer, loss_function = loss_function, device= device, writer= writer, epoch=epoch)
        val(model= model, val_dataloader= val_dataloader, loss_function = loss_function, device= device, writer= writer, epoch=epoch)
        torch.save(model.state_dict(), os.path.join(SAVE_PATH, str(f'resnet50_model_epoch_{epoch+1}')))

if __name__ == '__main__':
    main()