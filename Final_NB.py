import os
import re
from collections import Counter
import numpy as np
from nltk.corpus import stopwords
import pandas as pd
from sklearn.metrics import accuracy_score
from nltk.stem import WordNetLemmatizer
  
lemmatizer = WordNetLemmatizer()
#Additional stopwords from meta data and some commonly appearing words from all class files
extended_list = ['path','from','newsgroups','edu','apr','make','think','know','would','likes','like','lines','subject','message','date','article','references','sender','organization','cantaloupe','line','news','line','gmt']
stop_words=stopwords.words('english')
stop_words.extend(extended_list)



# Folder Path
path = "D:\Spring 2023\Machine Learning\Assignments and Projects\project2\data\/20_newsgroups\/"

# Change the directory
os.chdir(path)

#array of strings 
arrau_of_20_str = [] # this will store all the text file data for training set , it will be list of strings , each string represents one class
class_names = [] # this will store 20 class names
prob_of_20_str = [] # this will be list of python dictonary {class:Probability} for all 20 classes for every calculated during training
global_counter_20_str = [] # this is a list of counters , each dict value will contain word count in that class



def preprocessing_training_set():
     
     for file in os.listdir(path):
        # Check whether file is in text format or not
        class_names.append(file)
        file_path = f"{path}\{file}"
        #with open(file_path, 'r') as f:
        res = ""
        for files in os.listdir(file_path)[:500]:
            print("Reading file ",files," for training data")
            filename =  f"{file_path}\{files}"
            with open(filename,'r') as f:
                content = f.read()
                content = ''.join(content.splitlines(keepends=True)[16:])
                
                res+=content
        arrau_of_20_str.append(res)
     return arrau_of_20_str


def remove_new_lines_spaces(file_content):
    #file_content = file_content.replace("","")
    #file_content = file_content.replace(".","")

    stripped_str = file_content.split()
    stripped_str = ' '.join(map(str,stripped_str))
    return stripped_str



def extract_just_words(content):
    # remove punctuation
    
    # below will remove all the punctuations
    for character in '''!()-[]{/};:'"+=_-)*&^%$#@!~><,.?/|\,<>./|?@#$%^&*_~''':
        content = content.replace(character, ' ')   

    
    #removing all digits
    digits = r'[0-9]'
    without_digits = re.sub(digits,'',content)
    #print(without_digits)

    # removing non ascii characters
    formatted_clean_str = without_digits.encode("ascii", errors="ignore").decode()
    #print(formatted_clean_str)

    
    
    return formatted_clean_str




def remove_stop_words(text):
    text = remove_new_lines_spaces(text)
    word_tokens = text.split(' ')
    filtered_sentence = [w for w in word_tokens if not w.lower() in stop_words]
    return ' '.join(filtered_sentence)
    
def lemmatize_words(text):
    temp_list = []
    for words in text:
        temp_list.append(lemmatizer.lemmatize(words))

    return temp_list


def to_lower_case(text):
    text=text.split(' ')
    new_text = []
    for word in text:
        new_text.append(word.lower())

    for word in new_text:
        if len(word)<4:
            new_text.remove(word)
    

    return ' '.join(new_text)
np.seterr(divide = 'ignore') 

def create_wordCount_list(global_20_str):
    i=0
    for each_str in global_20_str:
        print("preprocessing and creating word count for class",i)
        each_str = remove_new_lines_spaces(each_str)
        each_str = extract_just_words(each_str)
        
        each_str = to_lower_case(each_str)
        each_str = remove_stop_words(each_str)
        #print(each_str)
        each_str = lemmatize_words(each_str.split(' '))

        counter_of_each_str = Counter(each_str)
        counter_of_each_str = dict(counter_of_each_str.most_common(30))
        
        #print(counter_of_each_str)
        global_counter_20_str.append(counter_of_each_str)
        i=i+1
    
    print("Calculating probability of words for each class")
    #call calculate probability function for all 20 classes and store the probability
    for counter_dict in global_counter_20_str:
        prob_of_20_str.append(calculate_probabilty(counter_dict))
    
    return prob_of_20_str

        
def training_calculation():
    text = preprocessing_training_set()
    create_wordCount_list(text)


#calculate and store probabilty of each word in particular class in a dictonary
#(number of occurances of a particular word / total number of words in that class)

def calculate_probabilty(counter_dic):
    temp_counter = {}
    for word in counter_dic:
        temp_counter[word] = counter_dic[word]/sum(counter_dic.values())
    
    return temp_counter




def calculate_probabilty_against_each_class(prob_input):
    prob = {}
    i=0
    
    for label in prob_of_20_str:
        for word in prob_input.keys():
            if word in label.keys():
                if i in prob.keys():
                    prob[i] +=np.log(pow(label[word],prob_input[word]))
                else:
                    prob[i]=np.log(pow(label[word],prob_input[word]))
        if i in prob.keys():
            prob[i]+=np.log((1/20))
            #print(prob[i])
        i=i+1
    #print(prob)
    return prob
 
 
training_calculation()


def testing():
    # Set the number of files to select
    data = []
    for file in os.listdir(path):
        
    # Check whether file is in text format or not
        class_names.append(file)
        file_path = f"{path}\{file}"
        #with open(file_path, 'r') as f:
        
        for files in os.listdir(file_path)[501:]:
            print("reading file ",files," for testing")
            filename =  f"{file_path}\{files}"
            with open(filename,'r') as f:
                contents = f.read()
                #print(contents)
                label = f"{file}"
                label = class_names.index(label)
                data.append({"label": label, "content": contents})

    # Create a pandas dataframe from the data list
    df = pd.DataFrame(data)
    predicted = []
    for each_row in df.iterrows():
        prob = {}
        input_text = each_row[1]['content']
        each_str = remove_new_lines_spaces(input_text)
        each_str = extract_just_words(each_str)
        each_str = to_lower_case(each_str)
        each_str = ' '.join(lemmatize_words(each_str.split(' ')))
        each_str = remove_stop_words(each_str)
        input_file_counter = Counter(each_str.split(' '))
        input_file_counter = dict(input_file_counter.most_common(10))
        for k, v in input_file_counter.copy().items():
            if len(k) < 2:
                del input_file_counter[k]
        
        prob = calculate_probabilty_against_each_class(input_file_counter)
        max_v=-100
        if prob:
            max_v = max(prob,key=prob.get,)

        predicted.append(max_v)
        

    
    df['predicted'] = predicted
    accuracy = accuracy_score(df['label'], df['predicted'])*100
    print("The Accuracy is "+ str(accuracy))
              

testing()