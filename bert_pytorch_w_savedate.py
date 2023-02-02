import pandas as pd
import torch
from transformers import BertTokenizer, BertModel
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import timeit
from tqdm import tqdm

torch.cuda.empty_cache()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
print(torch.cuda.get_device_name())
print(torch.__version__)
print(torch.version.cuda)

tensor_input_ids = torch.load('tensor_input_ids_800.pt').to(device)
tensor_attention_mask = torch.load('tensor_attention_mask_800.pt').to(device)
tensor_labels = torch.load('tensor_labels_800.pt').to(device)

class BinaryClassification(torch.nn.Module):
    def __init__(self, num_labels, model):
        super(BinaryClassification, self).__init__()
        self.num_labels = num_labels
        self.bert = model
        self.intermediate_layer1 = torch.nn.Linear(768, 512)
        self.intermediate_layer2 = torch.nn.Linear(512, 256)
        self.dropout = torch.nn.Dropout(0.1)
        self.classifier = torch.nn.Linear(256, num_labels)

    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids, attention_mask)
        output = self.intermediate_layer1(output.pooler_output)
        output = self.intermediate_layer2(output)
        output = self.dropout(output)
        output = self.classifier(output)
        return output

num_labels = 2
model = BertModel.from_pretrained('bert-base-uncased').to(device)
binary_classification_model = BinaryClassification(num_labels, model).to(device)

data = TensorDataset(tensor_input_ids, tensor_attention_mask, tensor_labels)

dataloader = DataLoader(data, batch_size=2, shuffle=True)

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(binary_classification_model.parameters(), lr=2e-5)

num_steps = 4
for step in range(num_steps):
    binary_classification_model.train()
    total_loss = 0
    total_acc = 0
    start_time = timeit.default_timer()
    for i_ids, a_mask, t_labels in tqdm(dataloader):
        logits = binary_classification_model(i_ids, a_mask)
        optimizer.zero_grad()
        loss = loss_fn(logits, t_labels)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        # calculate accuracy
        with torch.no_grad():
            train_preds = torch.argmax(logits, dim=1)
            acc = (train_preds == t_labels).float().mean()
            total_acc += acc.item()
    # calculate average accuracy
    avg_acc = total_acc / len(dataloader)
    end_time = timeit.default_timer()
    step_duration = end_time - start_time
    print("Step: {} Loss: {} Accuracy: {} Time: {}".format(step, total_loss/len(dataloader), avg_acc, step_duration))

torch.save({
            'model_state_dict': binary_classification_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
            }, 'pybert_model_7')