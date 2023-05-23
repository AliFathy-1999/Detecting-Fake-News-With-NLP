import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk # Natural Language toolkit
nltk.download("stopwords")  #downloading stopwords
nltk.download('punkt')
from nltk import word_tokenize,sent_tokenize
nltk.download('wordnet')
import nltk as nlp
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re

#%matplotlib inline
df1 = pd.read_csv('True.csv')
df2 = pd.read_csv('Fake.csv')
df1['Target']=1
df2['Target']=0
df=pd.concat([df1,df2],axis=0)
sns.countplot("Target",data = df) # 1= True 0=Fake

plt.figure(figsize=(10,5))
sns.countplot("subject",hue="Target" , data=df)
plt.xticks(rotation=90)

year=[]

for i in df.date:
    if '2017' in i:
        year.append("2017")
    elif '2016' in i:
        year.append("2016")
    elif '2015' in i:
        year.append("2015")
    else:
        year.append("2015")
    
len(year)
len(df)
df["Year"]=year

plt.figure(figsize=(10,5))
sns.countplot("Year",hue="Target" , data=df)
plt.xticks(rotation=90)

plt.figure(figsize=(10,5))
sns.countplot("subject",hue="Year" , data=df)
plt.xticks(rotation=90) 



fig,ax=plt.subplots(1,2,figsize=(20,5))
sns.countplot("subject" ,data=df2,ax=ax[0])
ax[0].set_title('Subjects for Fake News')

sns.countplot("subject",data=df1,ax=ax[1])
ax[1].set_title('Subjects for True News')
fig.show()

#Natural language processing

X=df.title.copy()
y=df.Target.copy()
X.head()
len(X)
y.head()
from nltk.corpus import stopwords
stop_words = stopwords.words('english')

X_list=[]

for i in X:
    i=re.sub("[^a-zA-z]"," ",i) # removing expressions that are not word
    i=i.lower() # make all lower case letters
    i = i.split()# ['letter1', 'letter2', 'letter3']
    i=" ".join([word for word in i if not word in stop_words]) #removing unused words
    X_list.append(i)
    
X_list[:5]
df["Cleaned"]=X_list
df.head()

from wordcloud import WordCloud 
import matplotlib.pyplot as plt
plt.subplots(figsize=(12,12))
wordcloud=WordCloud(background_color="white",width=1024,height=768).generate(" ".join(df.Cleaned))
plt.imshow(wordcloud)
plt.axis("off")
plt.show()

df.title[0]
df.Cleaned[0]
X=df.Cleaned

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42,shuffle=True)

print(len(X_train)) # 31428 sentences
print(len(y_train)) #31428 Labels
print(len(X_test))  # 13470 sentences
print(len(y_test)) #13470 Labels

max_lenght=100

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
tokenizer = Tokenizer() 
tokenizer.fit_on_texts(X_train)

word_index = tokenizer.word_index # creating word dict for words in training
sequences = tokenizer.texts_to_sequences(X_train)  # replacing words with the number corresponding to them in the dictionary(word_index)
X_train_padded = pad_sequences(sequences, padding='post',maxlen=max_lenght) # padding words

print(len(word_index))
print(word_index)

print("Original Version:",X_train[13970])
print("---------------------------------")
print("Padded version",X_train_padded[0]) 
print("---------------------------------")
print("Tokenized version:",sequences[0])
print("---------------------------------")
print("Shape after the padding:",X_train_padded.shape)  

X_test_sequences = tokenizer.texts_to_sequences(X_test)
X_test_padded = pad_sequences(X_test_sequences,padding="post",maxlen=max_lenght)

print("Original Version:",X_test[226])
print("---------------------------------")
print("Padded version",X_test_padded[0]) 
print("---------------------------------")
print("Tokenized version:",X_test_sequences[0])
print("---------------------------------")
print("Shape after the padding:",X_test_padded.shape)  

#improve a model and make it more generilize and prevent overfitting
import tensorflow as tf

vocab_size = len(tokenizer.word_index)+1
embedding_dim=16

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=100),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='c'),
    tf.keras.layers.Dropout(0.50),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()

num_epochs = 4
history=model.fit(X_train_padded,y_train, epochs=num_epochs, validation_data=(X_test_padded,y_test))

import matplotlib.pyplot as plt


plt.plot(history.history["accuracy"],color="green")
plt.plot(history.history["loss"],color="red")
plt.title("Train accuracy and Train loss")
plt.grid()

plt.plot(history.history["val_accuracy"],color="blue")
plt.plot(history.history["val_loss"],color="orange")
plt.title("Test accuracy and Test loss")
plt.grid()

print("Accuracy of the model on Training Data is - " , model.evaluate(X_train_padded,y_train)[1]*100 , "%")
print("Accuracy of the model on Testing Data is - " , model.evaluate(X_test_padded,y_test)[1]*100 , "%")

pred = model.predict_classes(X_test_padded)
pred[:5]
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test,pred)
sns.heatmap(cm,annot=True,linecolor="white",fmt='' , xticklabels = ['Fake','True'] , yticklabels = ['Fake','True'])


