import torch
from torch import nn 
from torchvision import models
from torch.utils.data import DataLoader
import pandas as pd

from dataset import CTDataset

#PATHS
png_path = "PNG"
model_path = "models/123_best.pth"

if __name__ == "__main__":
    out_features = 2
    batch_size = 1
    net = models.resnet18(pretrained=True)
    net.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    net.fc = nn.Linear(in_features=512, out_features=out_features, bias=True)

    net.load_state_dict(torch.load(model_path))

    net.cuda()
    net.eval()

    test_data = CTDataset(png_path, binary_classification=True, inference=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
    results = []
    df_results = {}
    with torch.no_grad():
        for batch_idx,(X,file_name) in enumerate(test_loader):
            X = torch.reshape(X, [batch_size, 1, 512, 512]).cuda().float()
            pred = net(X)
            file_id = file_name[0].split("/")[-1].split(".")[0]
            result = (file_id,pred.argmax(1).cpu().numpy()[0])
            results.append(result)
            df_results[batch_idx] = [file_id, pred.argmax(1).cpu().numpy()[0]]
    
    with open("asama1.txt","w") as file:
        file.write("Image Id	Inme Var mi?")
        for result in results:
            file.write("\n")
            line = f"{result[0]}    {result[1]}"
            file.write(line)

    df = pd.DataFrame.from_dict(df_results, orient="index", columns=["DOSYA NO", "INME VAR MI"])
    df.to_csv("Oturum1.csv", index=False)