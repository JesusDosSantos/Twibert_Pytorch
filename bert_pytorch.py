import pandas as pd
import torch
from transformers import BertTokenizer, BertModel
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import timeit

torch.cuda.empty_cache()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
print(torch.cuda.get_device_name())
print(torch.__version__)
print(torch.version.cuda)

print('Loading train.json')
df_train = pd.read_json('./Twibot-20/train.json')
print('Loading test.json')
df_test = pd.read_json('./Twibot-20/test.json')
print('Loading support.json')
df_dev = pd.read_json('./Twibot-20/dev.json')
print('Finished')
df_train = df_train.iloc[:, [0, 1, 2, 3, 5]]
df_test = df_test.iloc[:, [0, 1, 2, 3, 5]]
df_dev = df_dev.iloc[:, [0, 1, 2, 3, 5]]

df_data_labeled = pd.concat([df_train, df_dev, df_test], ignore_index=True)

df_data_labeled = df_data_labeled.dropna()

df_tweets_labeled = df_data_labeled.explode('tweet')

df_tweets_labeled = df_tweets_labeled.sample(n=200000)

df_data_labeled = df_tweets_labeled.dropna()

df_tweets_labeled = df_data_labeled.explode('tweet')
df_tweets_labeled = df_tweets_labeled.reset_index()

df_tweets_labeled = df_tweets_labeled[['tweet','label']]

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

input_ids = []
attention_masks = []
labels = []


df_tweets_labeled = df_tweets_labeled.values.tolist()
for text, label in df_tweets_labeled:
    encoded_text = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=256,
        return_token_type_ids=False,
        pad_to_max_length=True,
        return_attention_mask=True
    )
    input_ids.append(encoded_text['input_ids'])
    attention_masks.append(encoded_text["attention_mask"])
    labels.append(label)


tensor_input_ids = torch.tensor(input_ids, device=device)
tensor_attention_mask = torch.tensor(attention_masks, device=device)
tensor_labels = torch.tensor(labels, device=device)

from sklearn.model_selection import train_test_split

pad_input_ids, test_input_ids, train_labels, test_labels = train_test_split(tensor_input_ids, tensor_labels, test_size=0.2, shuffle=False, stratify=None)
pad_attention_mask, test_attention_mask, _, _ = train_test_split(tensor_attention_mask, tensor_labels,test_size=0.2, shuffle=False, stratify=None)

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

data = TensorDataset(pad_input_ids, pad_attention_mask, train_labels)

dataloader = DataLoader(data, batch_size=10, shuffle=True)

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(binary_classification_model.parameters(), lr=2e-5)

num_steps = 4
from tqdm import tqdm
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

