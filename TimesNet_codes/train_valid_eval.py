import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

from TimesNet_codes.losses import *

def train(model, tr_dataloader, va_dataloader):
    
    weights = torch.zeros((1,128,11), dtype=int) #원하는 타겟만 사용을 위함
    weights[: , : , 4] = 1 #우리는 target만 사용
    
    criterion = mape_loss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(1):
        epoch_loss = 0.0
        with tqdm(total=len(tr_dataloader), desc=f"epoch {epoch}", position=0) as batch_bar:
            for i, (batch_x, batch_y) in enumerate(tr_dataloader): # Also added tqdm here.

                batch_x = batch_x.float()
                
                label = batch_y[: , -128: , :].float()
                dec_inp = torch.zeros_like(batch_y[:, -128:, :]).float()
                dec_inp = torch.cat([batch_y[:, :48, :], dec_inp], dim=1).float()
                
                optimizer.zero_grad()
                outputs = model(batch_x, None, dec_inp, None)
                
                loss = criterion( outputs, label, weights )
                loss.backward()
                
                epoch_loss += loss
                
                batch_bar.set_postfix(loss=f"{epoch_loss/(i+1):.4f}")
                batch_bar.update(1)
            
            
            model.eval()
            for vali_x, vali_y in va_dataloader:
                
                vali_x = vali_x.float()
                
                label = vali_y[: , -128: , :].float()
                dec_inp = torch.zeros_like(vali_y[:, -128:, :]).float()
                dec_inp = torch.cat([vali_y[:, :48, :], dec_inp], dim=1).float()
                
                outputs = model(vali_x, None, dec_inp, None)
                
                loss = criterion( outputs, label, weights )
                print(f"validation loss : {loss}")
                
                forecast = outputs[weights==1].detach().numpy()
                label = label[weights==1].detach().numpy()
                
                plt.figure(figsize=(20,4))
                plt.plot(forecast, label='forecast')
                plt.plot(label, label='label')
                plt.legend()
                plt.xlabel('time idx')
                plt.ylabel('value')
                plt.title('validation prediction at building 1')
                plt.savefig('TimesNet_codes/valid_pred.png')
                