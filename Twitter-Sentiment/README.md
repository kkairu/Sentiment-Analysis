
# Kenya Airways Twitter Sentiment Analysis Model using fastText
- FastText is an NLP library developed by the Facebook AI. It's easy to use and fast to train.
- The core of FastText relies on the Continuous Bag of Words (CBOW) model for word representation and a hierarchical classifier to speed up training.
- fastText replaces the softmax over labels with a hierarchical softmax.

## Training Data Source - Twitter US Airline Sentiment Data on Kaggle/CrowdFlower
- https://www.kaggle.com/jagannathrk/twitter-us-airline-sentiment-analysis/data

Twitter data was scraped from February of 2015 and contributors were asked to first classify positive, negative, and neutral tweets, followed by categorizing negative reasons (such as "late flight" or "rude service").


```python
import sys, os, csv, datetime, json

import fasttext, nltk
#nltk.download('punkt')


from bs4 import BeautifulSoup
import re
import itertools
import emoji

import pandas as pd 

from sklearn.model_selection import train_test_split

```


```python
# Load smileys dictionary
smileys_dict = {}
with open("data\smileys.txt") as f:
    for line in f:
       (key, val) = line.split()
       smileys_dict[key] = val
        
#print(smileys_dict)
print('Smileys Dictionary Loaded...')
```

    Smileys Dictionary Loaded...
    


```python
# Load Contractions dictionary - takes care of slang contractions as well
# Consider contractions package ****

contractions_dict = {}
with open("data\contractions.txt") as f:
    for line in f:
       (key, val) = line.split(': ')
       contractions_dict[key] = val.strip('\n')
        
#print(contractions_dict)
print('Contractions Dictionary Loaded...')
```

    Contractions Dictionary Loaded...
    

## Cleaning tweets

Contractions/slang cleaning, Fix misspelled word, Escaping HTML, Removal of hashtags/accounts, Removal of punctuation, Emojis/Smileys etc.


```python
def clean_tweet(tweet):    
    
    #Escaping HTML characters
    tweet = BeautifulSoup(tweet,'lxml').get_text()
   
    #Special case not handled previously.
    tweet = tweet.replace('\x92',"'")
    
    #Removal of hastags/account
    tweet = ' '.join(re.sub("(@[A-Za-z0-9]+)|(#[A-Za-z0-9]+)", " ", tweet).split())
    
    #Removal of address
    tweet = ' '.join(re.sub("(\w+:\/\/\S+)", " ", tweet).split())
    
    #Removal of Punctuation
    tweet = ' '.join(re.sub("[\.\,\!\?\:\;\-\=]", " ", tweet).split())
    
    #Lower case
    tweet = tweet.lower()
    
    #CONTRACTIONS source: https://en.wikipedia.org/wiki/Contraction_%28grammar%29
    CONTRACTIONS = contractions_dict #contractions_dict()
    tweet = tweet.replace("’","'")
    words = tweet.split()
    reformed = [CONTRACTIONS[word] if word in CONTRACTIONS else word for word in words]
    tweet = " ".join(reformed)
    
    # Standardizing words
    tweet = ''.join(''.join(s)[:2] for _, s in itertools.groupby(tweet))
    
    #Deal with emoticons source: https://en.wikipedia.org/wiki/List_of_emoticons
    SMILEY = smileys_dict #smileys_dict()  
    words = tweet.split()
    reformed = [SMILEY[word] if word in SMILEY else word for word in words]
    tweet = " ".join(reformed)
    
    #Deal with emojis
    tweet = emoji.demojize(tweet)

    tweet = tweet.replace(":"," ")
    tweet = ' '.join(tweet.split())

    return tweet
```

## Format the data


```python
def transform_instance(row):
    cur_row = []
    #Prefix the index-ed label with __label__
    label = "__label__" + row[5]  
    cur_row.append(label)
    cur_row.extend(nltk.word_tokenize(clean_tweet(row[14].lower())))
    return cur_row


```


```python
#data = pd.read_csv('data\AirlineTwitterSentiment.csv', encoding = 'unicode_escape') 

#train, test = train_test_split(data, test_size=0.25)

#train.to_csv (r'data\AirlineTwitterSentiment_TRAIN.csv', index = None, header=True)
#test.to_csv (r'data\AirlineTwitterSentiment_TEST.csv', index = None, header=True)
```


```python
def data_preprocess(input_file, output_file, keep=1):
    i=0
    with open(output_file, 'w', encoding='utf-8') as csvoutfile:
        csv_writer = csv.writer(csvoutfile, delimiter=' ', lineterminator='\n')
        with open(input_file, 'r', newline='', encoding='utf-8') as csvinfile: #,encoding='latin1'
            csv_reader = csv.reader(csvinfile, delimiter=',', quotechar='"')
            for row in csv_reader:
                if row[4]!="MIXED" and row[5].upper() in ['POSITIVE','NEGATIVE','NEUTRAL'] and row[14]!='':
                    row_output = transform_instance(row)
                    csv_writer.writerow(row_output)
                    # print(row_output)
                #i=i+1
                #if i%10000 ==0:
                #    print(i)
            
# Preparing the training dataset        
data_preprocess('data\AirlineTwitterSentiment_TRAIN.csv', 'tweets.train')

# Preparing the validation dataset        
data_preprocess('data\AirlineTwitterSentiment_TEST.csv', 'tweets.validation')

print('Data Preprocess Done...')
```

    Data Preprocess Done...
    

## Upsampling to offset categories imbalance.
Category imbalance problem occurs when one label appears more often than others. In such a situation, classifiers tend to be overwhelmed by the large classes and ignore the small ones.


```python
def upsampling(input_file, output_file, ratio_upsampling=1):
    # Create a file with equal number of tweets for each label
    #    input_file: path to file
    #    output_file: path to the output file
    #    ratio_upsampling: ratio of each minority classes vs majority one. 
    #    *** 1 means there will be as much of each class than there is for the majority class 
    
    i=0
    counts = {}
    dict_data_by_label = {}

    # GET LABEL LIST AND GET DATA PER LABEL
    with open(input_file, 'r', newline='', encoding='utf-8') as csvinfile: 
        csv_reader = csv.reader(csvinfile, delimiter=',', quotechar='"')
        for row in csv_reader:
            counts[row[0].split()[0]] = counts.get(row[0].split()[0], 0) + 1
            if not row[0].split()[0] in dict_data_by_label:
                dict_data_by_label[row[0].split()[0]]=[row[0]]
            else:
                dict_data_by_label[row[0].split()[0]].append(row[0])
            #i=i+1
            #if i%10000 ==0:
            #    print("read" + str(i))

    # FIND MAJORITY CLASS
    majority_class=""
    count_majority_class=0
    for item in dict_data_by_label:
        if len(dict_data_by_label[item])>count_majority_class:
            majority_class= item
            count_majority_class=len(dict_data_by_label[item])  
    
    # UPSAMPLE MINORITY CLASS
    data_upsampled=[]
    for item in dict_data_by_label:
        data_upsampled.extend(dict_data_by_label[item])
        if item != majority_class:
            items_added=0
            items_to_add = count_majority_class - len(dict_data_by_label[item])
            while items_added<items_to_add:
                data_upsampled.extend(dict_data_by_label[item][:max(0,min(items_to_add-items_added,len(dict_data_by_label[item])))])
                items_added = items_added + max(0,min(items_to_add-items_added,len(dict_data_by_label[item])))

    # WRITE ALL
    #i=0

    with open(output_file, 'w', encoding='utf-8') as txtoutfile:
        for row in data_upsampled:
            txtoutfile.write(row+ '\n' )
            #i=i+1
            #if i%10000 ==0:
            #    print("writer" + str(i))


upsampling( 'tweets.train','uptweets.train')
# No need to upsample for the validation set. As it does not matter what validation set contains.

print('Upsampling done...')

```

    Upsampling done...
    

## Training the Model
- https://github.com/facebookresearch/fastText/tree/master/python

## Baseline 80-85% - Current Model Prediction Accuracy == 79.21%

Research shows that human analysts tend to agree around 80-85% of the time.


```python
training_data_path ='uptweets.train' 
validation_data_path ='tweets.validation'
model_path =''
model_name="model_KQ_Sentiment.ftz"

def train():
    print(str(datetime.datetime.now())+' == Training START')
    try:
        hyper_params = {"lr": 0.01,
                        "epoch": 50,
                        "wordNgrams": 3,
                        "dim": 20}     
                               
        #print( + ' START=>')

        # Train the model.
        model = fasttext.train_supervised(input=training_data_path, **hyper_params)
        print("\t Model trained with the hyperparameter {}".format(hyper_params))

        # CHECK PERFORMANCE
        print(str(datetime.datetime.now()) + ' == Training COMPLETE.')
        
        model_acc_training_set = model.test(training_data_path)
        model_acc_validation_set = model.test(validation_data_path)
        
        # DISPLAY ACCURACY OF TRAINED MODEL
        text_line = "\nAccuracy:" + str(model_acc_training_set[1])  + "\nValidation:" + str(model_acc_validation_set[1]) + '\n' 
        print(text_line)
        
        #quantize a model to reduce the memory usage
        model.quantize(input=training_data_path, qnorm=True, retrain=True, cutoff=100000)
        
        
        model.save_model(os.path.join(model_path,model_name))    
        
        print("Model is quantized and Saved!")
    
    except Exception as e:
        print('Exception during training: ' + str(e) )


# Train your model.
train()
```

    2019-12-29 23:22:24.279551 == Training START
    	 Model trained with the hyperparameter {'lr': 0.01, 'epoch': 50, 'wordNgrams': 3, 'dim': 20}
    2019-12-29 23:22:28.767462 == Training COMPLETE.
    
    Accuracy:0.9655621473014036
    Validation:0.7898907103825137
    
    Model is quantized and Saved!
    


```python
model = fasttext.load_model(model_name)
```

    
    


```python
new_tweet = "KQ250 is a shit plane to Seychelles . You pay an absurd amount for a ticket for lousy entertainment , food , service and comfort . @KenyaAirways . When the heck are you going to provide a better plane to paradise ."

model.predict([new_tweet],k=3)
```




    ([['__label__negative', '__label__neutral', '__label__positive']],
     array([[9.96885896e-01, 2.63305381e-03, 5.11026010e-04]]))




```python
model.predict([new_tweet],k=1)
```




    ([['__label__negative']], array([[0.9968859]]))


