import os
import pandas as pd
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import Trainer
from transformers import TrainingArguments
from datasets import load_dataset
from sklearn import preprocessing

os.environ['WANDB_DISABLED'] = 'true'

print('Loading dataset')
raw_dataset = load_dataset('csv', data_files='data/intent_train_data.csv')

print('Formatting data')
le = preprocessing.LabelEncoder()
le.fit(raw_dataset['train']['intent'])

tokenizer = AutoTokenizer.from_pretrained('distilbert-base-cased')

def text_tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True)
def intent_format_function(examples):
    return {'label': le.transform(examples['intent'])}

tokenized_dataset = raw_dataset.map(text_tokenize_function, batched=True)
tokenized_dataset = tokenized_dataset.map(intent_format_function, batched=True)

train_dataset = tokenized_dataset['train']

print('Loading model')
model = AutoModelForSequenceClassification.from_pretrained(
    'distilbert-base-cased',
    num_labels = 2,
    cache_dir = './models').cuda()

print('Setting up training')
training_args = TrainingArguments(
    output_dir = './models/intent_modeling',
    gradient_accumulation_steps = 1,
    # learning_rate = 2e-5,
    per_device_train_batch_size = 1,
    per_device_eval_batch_size = 1,
    label_names = ['label'],
    report_to = None)

trainer = Trainer(
    model = model,
    args = training_args,
    train_dataset = train_dataset)

print('Starting training')
trainer.train()

print('Saving model')
model.save_pretrained('./models/intent_modeling')

print('Save complete')