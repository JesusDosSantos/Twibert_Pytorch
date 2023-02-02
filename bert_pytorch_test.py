import pandas as pd
import torch
from transformers import BertTokenizer, BertModel
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import timeit


torch.cuda.empty_cache()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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

df_tweets_labeled = df_tweets_labeled.sample(n=100000)

df_data_labeled = df_tweets_labeled.dropna()

df_tweets_labeled = df_data_labeled.explode('tweet')
df_tweets_labeled = df_tweets_labeled.reset_index()


df_tweets_labeled = df_tweets_labeled[['tweet','label']]

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

input_ids = []
attention_masks = []
labels = []


df_tweets_labeled = df_tweets_labeled.values.tolist()
for text, label in tqdm(df_tweets_labeled):
    encoded_text = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=500,
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

# Load the saved model


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
optimizer = torch.optim.Adam(binary_classification_model.parameters(), lr=2e-5)
checkpoint = torch.load('pybert_model_7')
binary_classification_model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

loss_fn = torch.nn.CrossEntropyLoss()

# Evaluation
binary_classification_model.eval()
test_dataset = TensorDataset(test_input_ids, test_attention_mask, test_labels)
test_dataloader = DataLoader(test_dataset, batch_size=10, shuffle=False)

test_loss = 0
test_acc = 0
test_num_steps = 0

from sklearn.metrics import accuracy_score, classification_report
with torch.no_grad():
    test_preds = []
    test_labels = []
    for i_ids, a_mask, t_labels in tqdm(test_dataloader):
        i_ids, a_mask, t_labels = i_ids.to(device), a_mask.to(device), t_labels.to(device)
        test_logits = binary_classification_model(i_ids, a_mask)
        test_loss += loss_fn(test_logits, t_labels)
        preds = torch.argmax(test_logits, dim=1)
        test_preds += preds.tolist()
        test_labels += t_labels.tolist()
        test_num_steps += 1
    test_loss = test_loss / test_num_steps
    test_acc = accuracy_score(test_labels, test_preds)
    print("Test Accuracy: {} Test Loss: {}".format(test_acc, test_loss.item()))
    print(classification_report(test_labels, test_preds))