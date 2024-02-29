from flask import Flask, request, jsonify
from preprocessing import refine_data
import torch
import random, os, yaml
import numpy as np
from easydict import EasyDict
from pathlib import Path
from distilbert import MyTrainer


def seed_everything(seed, deterministic=False):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = not deterministic
        
def policy(dists, tlevels, fclevel):
    if fclevel in [6, 4, 2]:
        return fclevel.item()
    if (tlevels == 5).all():
        return 5 if dists[0] < 1.5 else 7
    if (tlevels == 3).all():
        return 3 if dists[0] < 1.5 else 7
    if dists[0] < 0.7:
        return fclevel.item()
    return 7

app = Flask(__name__)

with open("C:/Users/joon/dla-flask/dacon_log_analysis/storage/config.yaml", "r", encoding='utf-8') as f:
    C = EasyDict(yaml.load(f, yaml.FullLoader))
    C.result_dir = Path(C.result_dir)
    C.dataset.dir = Path(C.dataset.dir)
    seed_everything(C.seed, deterministic=False)

tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', 'distilbert-base-uncased', trust_repo=True)
trainer = MyTrainer(C, 1, 'C:/Users/joon/dla-flask/dacon_log_analysis/storage/distilbert-base-uncased-focal-AdamW-lr1e-05-ver7-os10_1.pth')
model = trainer.model

model.eval()
torch.set_grad_enabled(False)

deck1 = torch.load('C:/Users/joon/dla-flask/dacon_log_analysis/storage/distilbert-base-uncased-focal-AdamW-lr1e-05-ver7-os10_1-deck1.pth', map_location=torch.device('cpu'))
    


@app.route('/prediction/', methods=['GET', 'POST'])
def prediction():
    if request.method == 'POST':
        def hook(model, input, output):
            activation.append(output.detach().cpu())
        
        if hasattr(model, 'module'):
            model.module.pre_classifier.register_forward_hook(hook)  # set feature hook function
        else:
            model.pre_classifier.register_forward_hook(hook)
        
        data = refine_data(request.json['input_data'])
        # tokenization
        tk_data = tokenizer.encode(data,
                                   add_special_tokens=True,
                                   padding='max_length',
                                   truncation=True,
                                   max_length=512,
                                   )
        tk_data = torch.tensor(tk_data, dtype=torch.long)
        activation = []
        fclevel = model(tk_data[None].cpu())[0].argmax(dim=1).cpu()
        print(activation)
        feat = torch.stack(activation)
        feat = feat[:, 0, :]
        # fclevel = fclevel[:, 0]
        deck1["feat_"] = deck1["feat"].cpu()
        dist_ = torch.norm(deck1["feat_"] - feat[0, None], p=None, dim=1)
        dist_, indices_ = dist_.topk(4, largest=False)
        tlevels = deck1["tlevel"][indices_.cpu()]
        print(fclevel[0])
        print(tlevels)
        print(dist_)
        
        result = policy(dist_, tlevels, fclevel[0])
        print(result)
        return jsonify({'prediction': result})
    else:
        return '<p>Wokring</p>'