# based on https://keras.io/examples/nlp/t5_hf_summarization/

import json
import pandas as pd
from datasets import Dataset
import keras_nlp
import tensorflow as tf
import numpy as np
from transformers import T5TokenizerFast, \
    TFAutoModelForSeq2SeqLM
from transformers.keras_callbacks import KerasMetricCallback
import re

data_dir = './data'
log_dir = f"{data_dir}/experiments/t5/logs"
save_path = f"{data_dir}/experiments/t5/models"
cache_path_train = f"{data_dir}/cache/t5.train"
cache_path_test = f"{data_dir}/cache/t5.test"
MODEL_NAME = "t5-base"
TRAIN_TEST_SPLIT = 0.05
MAX_INPUT_LENGTH = 512
MAX_TARGET_LENGTH = 512
BATCH_SIZE = 8
LEARNING_RATE = 3e-4
MAX_EPOCHS = 10

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
def jsonToStringwithComa(entry):
    m1 = json.loads(entry)
    m2 = [list(x.values()) for x in m1]
    m3 = ''.join([item + ',' for sublist in m2 for item in sublist])
    return m3


def main_T5_HuggingFace_Keras():
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
    ds = tf.data.TextLineDataset(qaPairs)

    # Pandas Dataset
    dataset = Dataset.from_list(qaPairs.to_dict('records'))
    raw_datasets = dataset.train_test_split(
        train_size=1-TRAIN_TEST_SPLIT, test_size=TRAIN_TEST_SPLIT
    )

    def preprocess_function(examples):
        inputs = [doc for doc in examples["question"]]
        model_inputs = tokenizer(inputs, max_length=MAX_INPUT_LENGTH, truncation=True, padding="max_length",
                                 add_special_tokens=True)
        # model_inputs = tokenizer(inputs, max_length=MAX_INPUT_LENGTH, truncation=True)
        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                examples["answers_text"], max_length=MAX_TARGET_LENGTH, truncation=True, padding="max_length",
                add_special_tokens=True
                # examples["answers_text"], max_length=MAX_TARGET_LENGTH, truncation=True
            )
        ids = np.array(labels["input_ids"])
        # to use in case DataCollator does not work ,must set the padding to labels -100 so the loss function ignores the padding
        # ids[ids == 0] = -100
        # ids= ids.tolist()

        model_inputs["labels"] = ids

        return model_inputs

    tokenizer = T5TokenizerFast.from_pretrained(MODEL_NAME)
    tokenized_datasets = raw_datasets.map(preprocess_function, batched=True)
    model = TFAutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    from transformers import DataCollatorForSeq2Seq
    # label_pad_token_id
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, return_tensors="tf")

    train_dataset = tokenized_datasets["train"].to_tf_dataset(
        batch_size=BATCH_SIZE,
        columns=["input_ids", "attention_mask", "labels"],
        shuffle=True,
        collate_fn=data_collator,
    )
    test_dataset = tokenized_datasets["test"].to_tf_dataset(
        batch_size=BATCH_SIZE,
        columns=["input_ids", "attention_mask", "labels"],
        shuffle=False,
        collate_fn=data_collator,
    )
    generation_dataset = (
        tokenized_datasets["test"]
            .shuffle()
            .select(list(range(BATCH_SIZE * 20)))
            .to_tf_dataset(
            batch_size=BATCH_SIZE,
            columns=["input_ids", "attention_mask", "labels"],
            shuffle=False,
            collate_fn=data_collator,
        )
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(optimizer=optimizer)

    rouge_l = keras_nlp.metrics.RougeL()

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
        metric_fn, eval_dataset=generation_dataset, predict_with_generate=True
    )

    callbacks = [metric_callback]

    # For now we will use our test set as our validation_data
    model.fit(
        train_dataset, validation_data=test_dataset, epochs=MAX_EPOCHS, callbacks=callbacks
    )
    model.save_pretrained(save_path)
    input_text = 'what desease has these simptoms:  vocal cord polyp are hoarse voice, difficulty speaking, throat swelling, and lump in throat, although you may still have vocal cord polyp without those symptoms.               ?'
    encoded_query = tokenizer(input_text,
                              return_tensors='tf', pad_to_max_length=True, truncation=True, max_length=MAX_INPUT_LENGTH)
    input_ids = encoded_query["input_ids"]
    attention_mask = encoded_query["attention_mask"]
    generated_answer = model.generate(input_ids, attention_mask=attention_mask,
                                      max_length=MAX_TARGET_LENGTH, top_p=0.95, top_k=50)
    decoded_answer = tokenizer.decode(generated_answer.numpy()[0])
    print("Answer: ", decoded_answer)

    def printQAPairs_vs_model_prediction_plus_rougeF1(number):
        inputText1 = qaPairs.iloc[number].values.tolist()[0]
        print("Question: ", inputText1)
        encoded_query = tokenizer(inputText1,
                                  return_tensors='tf', pad_to_max_length=True, truncation=True,
                                  max_length=MAX_INPUT_LENGTH)
        input_ids = encoded_query["input_ids"]
        attention_mask = encoded_query["attention_mask"]
        generated_answer = model.generate(input_ids, attention_mask=attention_mask,
                                          max_length=MAX_TARGET_LENGTH, top_p=0.95, top_k=50)
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
    indexQaPairs = [18,800,1600,2400]
    for i in indexQaPairs:
        printQAPairs_vs_model_prediction_plus_rougeF1(i)

if __name__ == '__main__':
    main_T5_HuggingFace_Keras()
