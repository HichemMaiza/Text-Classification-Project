import os
import glob
# text processing imports
#import spacy
import sklearn
import string
import re
import pandas as pd

# path to file and list of files names in train 
path_to_txt_train = '/home/maiza/Bureau/Projets/dlil_project/txttoutf8/*'
path_labels_train = r'/home/maiza/Bureau/Projets/dlil_project/release1/labels/train.2015.lst' 

# path to file and list of files names in dev 

path_to_txt_dev = '/home/maiza/Bureau/Projets/dlil_project/txttoutf8_dev/*'
path_labels_dev = r'/home/maiza/Bureau/Projets/dlil_project/release1/labels/dev.2015.lst' 

# path to file and list of files names in test 

path_to_txt_test = '/home/maiza/Bureau/Projets/dlil_project/txttoutf8_test/*'
path_labels_test= r'/home/maiza/Bureau/Projets/dlil_project/release1/labels/test.2015.lst' 


list_file_train = glob.glob(path_to_txt_train)
list_file_dev = glob.glob(path_to_txt_dev) 
list_file_test = glob.glob(path_to_txt_test) 



# pre-traitement  

def pretraitement(list_file, path_labels, p) : 
    
    ponctuation = string.punctuation
    
    # a dictionary to store data  
    
    train_data = {

        'file_id' : [], 
        'text' : [],
        'file_label' : []
    }
    
    # a dictionary to store labels with respect to there ids

    train_id_labels= {
    
    'file_id': [], 
    'file_label': []
    }

    for file in list_file: 

        # extract_id 

        path = file[0:len(p)-1]
        file_name = file[len(p)-1:].split('_')
        file_id =str(file_name[0]) + '_' + str(file_name[1]) + '_' + str(file_name[2]) + '_' + str(file_name[3])

        first_list_line = []
        second_list_line = []

        with open(file) as f:

            lines = f.readlines()
            for line in lines :
                line = line.replace("\n",' ')
                first_list_line.append(line)

            for index, value in enumerate(first_list_line):
                if (value != ' '):
                    second_list_line.append(value)

            text = str(second_list_line)

            for i in text:
                if i in ponctuation:
                    text=text.replace(i,'')

            train_data['text'].append(text)
            train_data['file_id'].append(file_id)
    
    # sort labels 
    # affect class labels to every text with respect to the id 
    # first extract id and labels         
    with open(path_labels) as labels:
        lines = labels.readlines()
        for l in lines:
            l = l.replace('\n', '').split('_')
            file_id =str(l[0]) + '_' + str(l[1]) + '_' + str(l[2]) + '_' + str(l[3])
            train_id_labels['file_id'].append(file_id)
            train_id_labels['file_label'].append(l[4])
        
    # construct global data frame
    for file_id_index, file_id_value in enumerate(train_data['file_id']):
        temp = []
        for sub_file_id_index, sub_file_id_value in enumerate (train_id_labels['file_id']):
            if (file_id_value == sub_file_id_value):                        
                temp = train_id_labels['file_label'][sub_file_id_index]       
                train_data['file_label'].append(temp) 
    data_frame = pd.DataFrame(train_data) 
    data_frame.head()
    return data_frame


data_train = pretraitement(list_file_train, path_labels_train, path_to_txt_train)
data_dev = pretraitement(list_file_dev, path_labels_dev, path_to_txt_dev)
data_test = pretraitement(list_file_test, path_labels_test, path_to_txt_test)


from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

vectorizer = TfidfVectorizer(token_pattern=r'\b\w+\b')
train_matrix = vectorizer.fit_transform(data_train['text'])
print(train_matrix.toarray())
# Second, convert the test data into a sparse matrix, using the same word-column mapping
dev_matrix = vectorizer.transform(data_dev['text']) 
test_matrix = vectorizer.transform(data_test['text']) 



from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(data_train['file_label']) 
labels_train = le.transform(data_train['file_label']) 
labels_test = le.transform(data_test['file_label'])


# Entrainement 
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
naive_bayes_model = gnb.fit(train_matrix.toarray(), labels_train)
pred = naive_bayes_model.predict(dev_matrix.toarray())
pred_test = naive_bayes_model.predict(test_matrix.toarray()) 


from sklearn.metrics import precision_score
pmacro = precision_score(labels_test, pred_test, average='macro') 
from sklearn.metrics import recall_score 
rmacro = recall_score(labels_test, pred_test, average='macro')   
print(" pmicro ",pmicro)
print(" rmicro",rmicro)

print("pmacro", pmacro) 
print("rmacro ", rmacro)




# Parti2 


pathToFile = '/home/maiza/Bureau/Projets/dlil_project/release2/trans-auto/utf.trans-asr-decoda.ctm'
fileId = [] 
texte = [] 
data = {}
with open(pathToFile) as transauto: 

    lines = transauto.readlines()
    for line in lines:
        line = line.strip('\n').split(' ')
        fileId.append(line[0])
        texte.append(line[4]) 
    uniquefileId = np.unique(fileId)
    
    data['fileId'] = fileId
    data['texte'] = texte
    
trans_auto = pd.DataFrame(data)


temp = []
completeTexte = []

j = 0
a = 0

for i in range(len(uniquefileId)):
    
    temp = [] 
    a = a + 1 
    

    while j < a and j < len(fileId):

        if uniquefileId[i] == fileId[j]:
            
            temp.append(texte[j])
            a +=1
            j +=1
            
        else : 
            
            j += 1
            
    completeTexte.append(temp) 



dataFrame = {
    'texte' : completeTexte, 
    'file_id': uniquefileId
}


trans_auto = pd.DataFrame(dataFrame)



test_auto_file = [] 
test_label_auto = []
test_text_auto = []

train_dev_auto_file = [] 
train_dev_auto = []
train_dev_text_auto = [] 

for index_manu, file_id_manu in enumerate(data_train['file_id']):
    for index_auto, file_id_auto in enumerate(trans_auto ['file_id']):
        if file_id_manu == file_id_auto: 
            test_auto_file.append(file_id_auto)
            test_label_auto.append(data_train['file_label'][index_manu])
            test_text_auto.append(data_train['text'][index_manu])
            
data_train_auto = {
    
    'file_id' : test_auto_file,
    'file_label' : test_label_auto,
    'text' : test_text_auto
}

data_train_auto = pd.DataFrame(data_train_auto) 


test_auto_file = [] 
test_label_auto = []
test_text_auto = []

train_dev_auto_file = [] 
train_dev_auto = []
train_dev_text_auto = [] 

for index_manu, file_id_manu in enumerate(data_test['file_id']):
    for index_auto, file_id_auto in enumerate(trans_auto ['file_id']):
        if file_id_manu == file_id_auto: 
            test_auto_file.append(file_id_auto)
            test_label_auto.append(data_test['file_label'][index_manu])
            test_text_auto.append(data_test['text'][index_manu])
            
data_test_auto = {
    
    'file_id' : test_auto_file,
    'file_label' : test_label_auto,
    'text' : test_text_auto
}

data_test_auto= pd.DataFrame(data_test_auto)







from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

vectorizer = TfidfVectorizer(token_pattern=r'\b\w+\b')
train_matrix = vectorizer.fit_transform(data_train_auto['text'])
# Second, convert the test data into a sparse matrix, using the same word-column mapping
#dev_matrix = vectorizer.transform(data_dev['text']) 
test_matrix = vectorizer.transform(data_test_auto['text']) 



from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(data_train_auto['file_label']) 

labels_train = le.transform(data_train_auto['file_label']) 
labels_test = le.transform (data_test_auto['file_label'])


from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
naive_bayes_model = gnb.fit(train_matrix.toarray(), labels_train)
#pred = naive_bayes_model.predict(test_matrix.toarray())


from sklearn.metrics import precision_score
pmicro = precision_score(labels_test, pred_test, average='micro') 

from sklearn.metrics import recall_score 
rmicro = recall_score(labels_test, pred_test, average='micro') 

from sklearn.metrics import precision_score
pmacro = precision_score(labels_test, pred_test, average='macro') 
from sklearn.metrics import recall_score 
rmacro = recall_score(labels_test, pred_test, average='macro')   


print(" pmicro ",pmicro)
print(" rmicro",rmicro)

print("pmacro", pmacro) 
print("rmacro ", rmacro)


