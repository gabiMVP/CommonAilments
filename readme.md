# Fine-tune the Pretrained T5 Model from Huggingface

The purpose is to do Closed Book Question-Answering on a given Dataset 

Here I did fine-tuning using the standard T5 Base model  with 220 million  parameters 

https://huggingface.co/docs/transformers/model_doc/t5
 

Original Paper of model :
https://arxiv.org/pdf/1910.10683.pdf

Original Training steps by the Creators :
https://colab.research.google.com/github/google-research/text-to-text-transfer-transformer/blob/master/notebooks/t5-trivia.ipynb#scrollTo=yGQ-zpgy3raf

### Dataset

We have a Dataset in the form of an cvs file 

Here we have 796 rows having data about 796 common health ailments

From these ailments we invision a practical model that can answer 4 types of questions:
- question1 : what disease has these simptoms: {simptoms entered by user}?
- question2 : who Is At Risk for {ailment}?
- question3 : what are the most common test and procedures for {ailment}? 
- question4 : which are the drugs to treat {ailment}? 

From 796 rows we can thus make 4 times as many question and answer pair


### Implementation notes:

We use pandas to create the question answer pairs 

One important note is that as per the documentation in the model https://huggingface.co/transformers/v2.10.0/model_doc/t5.html
the padding on the labes has to be with value -100 so the loss function of the model knows to ignore it for the loss calculation 

I tried 2 implementations
- using Tensorflow + the HuggingFace T5Model
- using Tensorflow + the HuggingFace T5Model + more Objects from the HuggingFace Library 

The learning rate is 3e-4 as in the documentation of the model 

The evaluation metric is Rouge Score L  and was implemented using the RougeL class from Keras
which focuses on  Longest Common Subsequence predicted by the model
 
good explanation here : https://medium.com/mlearning-ai/text-summarization-84ada711c49c
 
### Results:
 
Best results with 5 training epochs on T5-base Model:

loss: 0.3785 - val_loss: 0.5736 - RougeL: 0.4860

### Some Examples of output 

I take a question answer pair from each of the 4 categories to compare the model's answer vs the target answer:

```diff
#Question:  what desease has these simptoms: glaucoma are symptoms of eye, spots or clouds in vision, blindness, and itchy eyelid, although you may still have glaucoma without those symptoms. ?

- Answer:  <pad> Glaucoma</s>

+Expected Answer:  Glaucoma

#Question:  who Is At Risk for Spondylitis?

- Answer:  <pad> Groups of people at highest risk for spondylitis include age 75+ years age 60-74 years. On the other hand, age 1-4 years and age <unk> 1 years almost never get spondylitis.,Within all the people who go to their doctor with spondylitis, 84% report having back pain, 56% report having neck pain, and 56% report having neck pain.</s>

+Expected Answer:  Groups of people at highest risk for spondylitis include age 45-59 years. On the other hand, age 1-4 years and age < 1 years almost never get spondylitis.,Within all the people who go to their doctor with spondylitis, 88% report having back pain, 83% report having low back pain, and 47% report having leg pain.

#Question:  what are the most comon test and procedures for Indigestion?

- Answer:  <pad> Hematologic tests (Blood test),Complete blood count (Cbc),Radiographic imaging procedure,Urinalysis,Glucose measurement (Glucose level),Electrolytes panel,Kidney function tests (Kidney function test),Electrocardiogram,</s>

+Expected Answer:  Hematologic tests (Blood test),Complete blood count (Cbc),Urinalysis,Glucose measurement (Glucose level),Electrolytes panel,Kidney function tests (Kidney function test),Electrocardiogram,Intravenous fluid replacement,

#Question:   what are drugs for Mononucleosis?

- Answer:  <pad> The most commonly prescribed drugs for patients with mononucleosis include clotrimazole topical, tetrahydrozoline ophthalmic, phenylephrine (duramax), cyclosporine ophthalmic, phenylephrine (duramax), cyclosporine ophthalmic, phenylephrine (duramax), cyclosporine ophthalmic, phenylephrine (duramax), timolol, acetaminophen ophthalmic and timolol.</s>

+Expected Answer:  The most commonly prescribed drugs for patients with mononucleosis include dexamethasone topical product, drospirenone / ethinyl estradiol, adapalene topical, phentermine, nalbuphine (nubain), cat's claw preparation, tromethamine, cladribine, intramuscular immunoglobulin (baygam), lipase, trandolapril / verapamil, desoximetasone topical and ramelteon (rozerem) .

```
### Notes
The expected answer is expected to be very long for question categories 3 and 4 respectively: 
- what are drugs for {ailment}
- what are the most comon test and procedures for{ailment}

and we see the model having a little difficulty at some point in the long sequences, 
should be tried with a larger variant of the model  + more training ephocs 

