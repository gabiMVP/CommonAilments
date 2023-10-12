# Based on :
# https://colab.research.google.com/github/snapthat/TF-T5-text-to-text/blob/master/snapthatT5/notebooks/TF-T5-Datasets%20Training.ipynb#scrollTo=rA9OVe28DXOq
#

import datetime
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import json
import numpy as np
import pandas as pd
import tensorflow as tf
import trax
from datasets import Dataset
from transformers import TFT5ForConditionalGeneration, T5TokenizerFast, KerasMetricCallback
from CommonAilmentsT5 import CommonAilmentsT5
import keras_nlp
from transformers.keras_callbacks import KerasMetricCallback
import re

rouge_l = keras_nlp.metrics.RougeL()

MODEL_NAME = "t5-base"
data_dir = "../data"
log_dir = f"{data_dir}/experiments/t5/logs"
save_path = f"{data_dir}/experiments/t5/models"
cache_path_train = f"{data_dir}/cache/t5.train"
cache_path_test = f"{data_dir}/cache/t5.test"
model = TFT5ForConditionalGeneration.from_pretrained(MODEL_NAME, return_dict=True)
tokenizer = T5TokenizerFast.from_pretrained(MODEL_NAME)
encoder_max_len = 512
decoder_max_len = 512
tf.config.run_functions_eagerly(True)


def jsonToStringwithComa(entry):
    m1 = json.loads(entry)
    m2 = [list(x.values()) for x in m1]
    m3 = ''.join([item + ',' for sublist in m2 for item in sublist])
    return m3
def wikitext_detokenizer(string):
    # contractions
    string = string.replace("s '", "s'")
    string = re.sub(r"/' [0-9]/", r"/'[0-9]/", string)
    string = re.sub("\s\s+", " ", string)
    # number separators
    string = string.replace(" @-@ ", "-")
    string = string.replace(" @,@ ", ",")
    string = string.replace(" @.@ ", ".")
    # punctuation
    string = string.replace(" : ", ": ")
    string = string.replace(" ; ", "; ")
    string = string.replace(" . ", ". ")
    string = string.replace(" ! ", "! ")
    string = string.replace(" ? ", "? ")
    string = string.replace(" , ", ", ")
    # double brackets
    string = re.sub(r"\(\s*([^\)]*?)\s*\)", r"(\1)", string)
    string = re.sub(r"\[\s*([^\]]*?)\s*\]", r"[\1]", string)
    string = re.sub(r"{\s*([^}]*?)\s*}", r"{\1}", string)
    string = re.sub(r"\"\s*([^\"]*?)\s*\"", r'"\1"', string)
    string = re.sub(r"'\s*([^']*?)\s*'", r"'\1'", string)
    # miscellaneous
    string = string.replace("= = = =", "====")
    string = string.replace("= = =", "===")
    string = string.replace("= =", "==")
    string = string.replace(" " + chr(176) + " ", chr(176))
    string = string.replace(" \n", "\n")
    string = string.replace("\n ", "\n")
    string = string.replace(" N ", " 1 ")
    string = string.replace(" 's", "'s")

    return string

def encode(example,
           encoder_max_len=encoder_max_len, decoder_max_len=decoder_max_len):
    question = example['question']
    answer = example['answers_text']

    encoder_inputs = tokenizer(question, truncation=True,
                               return_tensors='tf', max_length=encoder_max_len,
                               pad_to_max_length=True
                               )

    decoder_inputs = tokenizer(answer, truncation=True,
                               return_tensors='tf', max_length=decoder_max_len,
                               pad_to_max_length=True
                               )

    input_ids = encoder_inputs['input_ids'][0]
    input_attention = encoder_inputs['attention_mask'][0]
    # we set the targets to -100 where padding as in documentation
    target_ids = decoder_inputs['input_ids'][0]
    condition = tf.equal(target_ids, 0)
    target_ids = tf.where(condition, -100, target_ids)

    target_attention = decoder_inputs['attention_mask'][0]

    outputs = {'input_ids': input_ids, 'attention_mask': input_attention,
               'labels': target_ids, 'decoder_attention_mask': target_attention}
    return outputs


def main_T5_Native_Tensforflow():
    dataPath = './data/dataDeasease.csv'
    df = pd.read_csv(dataPath, delimiter=';')

    df = df.dropna()
    """
    question1 : what disease has these simptoms: [simptomsDesc]
    Answer: [name]
    """

    q1 = 'what desease has these simptoms:' + df['Sympoms desc'] + '?'
    q1 = q1.str.replace('The symptoms that are highly suggestive of', '')
    a1 = df['name']

    """
    question2 : who Is At Risk for [name]  
    Answer: [whoIsAtRiskDesc]
    """
    q2 = 'who Is At Risk for ' + df['name'] + '?'
    a2 = df['whoIsAtRiskDesc']
    """
    question3 : what are the most comon test and procedures for [name] ? 
    Answer: [commonTestsAndProcedures] after deJson
    """
    q3 = 'what are the most comon test and procedures for ' + df['name'] + '?'
    a3 = df['commonTestsAndProcedures'].map(jsonToStringwithComa)

    """
    question4 : what are drugs for [name] ? 
     Answer: [medications1] 
    """
    q4 = ' what are drugs for  ' + df['name'] + '?'
    a4 = df['medications1']

    q = pd.concat([q1, q2, q3, q4])
    a = pd.concat([a1, a2, a3, a4])

    q = q.map(wikitext_detokenizer)
    a = a.map(wikitext_detokenizer)
    qaPairs = pd.concat((q, a), axis=1)
    qaPairs.columns = ["question", "answers_text"]

    # just used to see how the model behaves, what are the expected inputs and outputs
    examine_model = False
    if (examine_model):
        examine_pretrained_model(a, q, qaPairs)

    # use the huggingface Dataset object
    dataset = Dataset.from_list(qaPairs.to_dict('records'))
    print(next(iter(dataset)))

    # use the huggingface Dataset object
    dataset = dataset.train_test_split(test_size=0.1)

    train_ds = dataset["train"]

    test_ds = dataset["test"]

    warmup_steps = 1e4
    batch_size = 4
    buffer_size = 1000
    ntrain = len(train_ds)
    nvalid = len(test_ds)
    steps = int(np.ceil(ntrain / batch_size))
    valid_steps = int(np.ceil(nvalid / batch_size))

    print("Total Steps: ", steps)
    print("Total Validation Steps: ", valid_steps)

    # print(dataset1)
    train_dataset = train_ds.map(encode)
    valid_dataset = test_ds.map(encode)
    ex = next(iter(train_dataset))
    print("Example data from the mapped dataset: \n", ex)
    """
    example fine tune by the ones who made T5   : https://colab.research.google.com/github/google-research/text-to-text-transfer-transformer/blob/master/notebooks/t5-trivia.ipynb
    """

    def to_tf_dataset(dataset):
        columns = ['input_ids', 'attention_mask', 'labels', 'decoder_attention_mask']
        dataset.set_format(type='tensorflow', columns=columns)
        return_types = {'input_ids': tf.int32, 'attention_mask': tf.int32,
                        'labels': tf.int32, 'decoder_attention_mask': tf.int32, }
        return_shapes = {'input_ids': tf.TensorShape([None]), 'attention_mask': tf.TensorShape([None]),
                         'labels': tf.TensorShape([None]), 'decoder_attention_mask': tf.TensorShape([None])}
        ds = tf.data.Dataset.from_generator(lambda: dataset, return_types, return_shapes)
        return ds

    tf_train_ds = to_tf_dataset(train_dataset)
    tf_valid_ds = to_tf_dataset(valid_dataset)

    def create_dataset(dataset, cache_path=None, batch_size=4,
                       buffer_size=1000, shuffling=True):
        if cache_path is not None:
            dataset = dataset.cache(cache_path)
        if shuffling:
            dataset = dataset.shuffle(buffer_size)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(1)
        return dataset

    tf_train_ds = create_dataset(tf_train_ds, batch_size=batch_size,
                                 shuffling=True, cache_path=None)
    tf_valid_ds = create_dataset(tf_valid_ds, batch_size=batch_size,
                                 shuffling=False, cache_path=None)

    steps_per_epoch = tf_train_ds.cardinality().numpy()
    valid_steps_per_epoch = tf_valid_ds.cardinality().numpy()
    print(steps_per_epoch)
    print(valid_steps_per_epoch)

    start_profile_batch = steps + 10
    stop_profile_batch = start_profile_batch + 100
    profile_range = f"{start_profile_batch},{stop_profile_batch}"

    log_path = log_dir + "/" + datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_path, histogram_freq=1,
                                                          update_freq=20, profile_batch=profile_range)

    checkpoint_filepath = save_path + "/" + "T5-{epoch:04d}-{val_loss:.4f}.ckpt"

    def metric_fn(eval_predictions):
        predictions, labels = eval_predictions
        decoded_predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        for label in labels:
            label[label < 0] = tokenizer.pad_token_id  # Replace masked label tokens
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        result = rouge_l(decoded_labels, decoded_predictions)
        # We will print only the F1 score, you can use other aggregation metrics as well
        result = {"RougeL": result["f1_score"]}

        return result

    metric_callback = KerasMetricCallback(
        metric_fn, eval_dataset=tf_valid_ds, predict_with_generate=True
    )

    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=False,
        monitor='val_loss',
        mode='min',
        save_best_only=True)

    callbacks = [tensorboard_callback, model_checkpoint_callback, metric_callback]
    metrics = [tf.keras.metrics.SparseTopKCategoricalAccuracy(name='accuracy')]

    # learning_rate = CustomSchedule()
    learning_rate = 3e-4  # Instead set a static learning rate
    optimizer = tf.keras.optimizers.Adam(learning_rate)

    model = CommonAilmentsT5.from_pretrained(MODEL_NAME)
    model.compile(optimizer=optimizer, metrics=metrics)

    model.fit(tf_train_ds, epochs=5, callbacks=callbacks,
              validation_data=tf_valid_ds)

    input_text = 'what desease has these simptoms:  vocal cord polyp are hoarse voice, difficulty speaking, throat swelling, and lump in throat, although you may still have vocal cord polyp without those symptoms.               ?'
    encoded_query = tokenizer(input_text,
                              return_tensors='tf', pad_to_max_length=True, truncation=True, max_length=encoder_max_len)
    input_ids = encoded_query["input_ids"]
    attention_mask = encoded_query["attention_mask"]
    generated_answer = model.generate(input_ids, attention_mask=attention_mask,
                                      max_length=decoder_max_len, top_p=0.95, top_k=50)
    decoded_answer = tokenizer.decode(generated_answer.numpy()[0])
    print("Answer: ", decoded_answer)

    "Expected Answer:  <pad> Vocal cord polyp</s>"

    def printQAPairs_vs_model_prediction_plus_rougeF1(number):
        inputText1 = qaPairs.iloc[number].values.tolist()[0]
        print("Question: ", inputText1)
        encoded_query = tokenizer(inputText1,
                                  return_tensors='tf', pad_to_max_length=True, truncation=True,
                                  max_length=256)
        input_ids = encoded_query["input_ids"]
        attention_mask = encoded_query["attention_mask"]
        generated_answer = model.generate(input_ids, attention_mask=attention_mask,
                                          max_length=256, top_p=0.95, top_k=50)
        decoded_answer = tokenizer.decode(generated_answer.numpy()[0])
        print("Answer: ", decoded_answer)
        print("Expected Answer: ", qaPairs.iloc[number].values.tolist()[1])
        resultq1 = rouge_l(qaPairs.iloc[number].values.tolist()[1], decoded_answer)
        print("Rouge: ", resultq1["f1_score"])

    """
    we take a qa pair for each category and see the results visually + F1 score from Rouge Score 

    QA pairs will be like  : 

    [Q: 'who Is At Risk for Spondylitis?',
     A : 'Groups of people at highest risk for spondylitis include age 45-59 years. On the other hand, age 1-4 years and age < 1 years almost never get spondylitis.
     ,Within all the people who go to their doctor with spondylitis, 88% report having back pain, 83% report having low back pain, and 47% report having leg pain. ']

    """
    indexQaPairs = [18, 800, 1600, 2400]
    for i in indexQaPairs:
        printQAPairs_vs_model_prediction_plus_rougeF1(i)


def examine_pretrained_model(a, q, qaPairs):
    """
    here we just see how the tokenizer tokenizez and how the model behaves
    """
    input_ids = tokenizer(
        "translate English to German:how are you?", return_tensors="tf").input_ids
    input_ids2 = tokenizer(
        "translate English to German:how are you?", return_tensors="tf")
    print(input_ids2)
    generated_ids = model.generate(input_ids=input_ids)
    preds = [tokenizer.decode(gen_id, skip_special_tokens=True)
             for gen_id in generated_ids
             ]
    print("".join(preds))
    qaPairs.head(10)
    q.head(10)
    input_ids = tokenizer(
        "translate English to German:how are you?", return_tensors="tf").input_ids
    input_ids2 = tokenizer(
        "translate English to German:how are you?", return_tensors="tf")
    print(input_ids2)
    generated_ids = model.generate(input_ids=input_ids)
    preds = [tokenizer.decode(gen_id, skip_special_tokens=True)
             for gen_id in generated_ids
             ]
    print("".join(preds))
    qaPairs.head(10)
    q.head(10)
    sampleQA = qaPairs.iloc[2]
    # print(sampleQA)
    print(sampleQA['question'])
    print(sampleQA['answers_text'])
    sampleQ = q.iloc[2]
    sampleA = a.iloc[2]
    print(sampleQ)
    print(sampleA)
    sampleQA_Encoded = tokenizer(
        sampleQA['question'],
        sampleQA['answers_text'],
        max_length=300,
        padding='max_length',
        return_tensors='tf')
    sampleQ_Encoded = tokenizer(
        sampleQ,
        max_length=300,
        padding='max_length',
        return_tensors='tf')
    sampleA_Encoded = tokenizer(
        sampleA,
        max_length=300,
        padding='max_length',
        return_tensors='tf')
    sampleQA_Encoded.keys()
    print(tokenizer.special_tokens_map)
    print("sample qa encoded \n")
    print(sampleQA_Encoded['input_ids'])
    print(sampleQA_Encoded['attention_mask'])
    print(tokenizer.decode(tf.squeeze(sampleQA_Encoded['input_ids'])))
    print("sample q encoded \n")
    print(sampleQ_Encoded['input_ids'])
    print(sampleQ_Encoded['attention_mask'])
    print(tokenizer.decode(tf.squeeze(sampleQ_Encoded['input_ids'])))
    print("sample a encoded \n")
    print(sampleA_Encoded['input_ids'])
    print(sampleA_Encoded['attention_mask'])
    print(tokenizer.decode(tf.squeeze(sampleA_Encoded['input_ids'])))
    label = sampleA_Encoded['input_ids']
    # print(label)
    condition = tf.equal(label, 0)
    label = tf.where(condition, -100, label)
    print(label)
    loss = model(input_ids=sampleQ_Encoded['input_ids'], labels=label)
    print(loss['loss'])
    # ds = tf.data.TextLineDataset(qaPairs)
    # dataset = Dataset.from_list(qaPairs.to_dict('records'))
    # dataset1 = Dataset.from_pandas(qaPairs)
    # dataset1 = dataset1.remove_columns('__index_level_0__')
    # x = qaPairs.to_dict('records')
    # y = [str(m).replace(")", "").replace("{", "") for m in x]
    # dataset1 = tf.data.Dataset.from_tensor_slices((q, a))
    # cut_off = int(len(x) * .05)
    # train_ds = dataset1.take(cut_off)
    # test_ds = dataset1.skip(cut_off)


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
if __name__ == '__main__':
    main_T5_Native_Tensforflow()
