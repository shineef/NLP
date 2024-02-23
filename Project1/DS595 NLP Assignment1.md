# DS595 NLP Assignment1

> datasets from [Fake News Detection | Kaggle](https://www.kaggle.com/datasets/bhavikjikadara/fake-news-detection?select=fake.csv)

### Task 1. Explore Data and Preprocess

1. The most commonly used words:
   
   For this section, we can explore the news text and title separately cause we don't know whether the performance of the text and title is consistent.
   
   Samples of the texts TOP10:
   
   | Real_News_Word | Real_News_Frequency | Fake_News_Word | Fake_News_Frequency |
   | -------------- | ------------------- | -------------- | ------------------- |
   | said           | 99021               | Trump          | 73361               |
   | Trump          | 54213               | said           | 31025               |
   | would          | 31483               | people         | 24844               |
   | Reuters        | 28363               | would          | 23218               |
   | government     | 18715               | one            | 21010               |
   | year           | 18639               | Clinton        | 17965               |
   | President      | 17409               | Obama          | 17771               |
   | state          | 16318               | like           | 17576               |
   | also           | 15744               | Donald         | 17180               |
   | House          | 15369               | President      | 16495               |
   
   Samples of the titles TOP10:
   
   Version A
   
   | Real_News_Word | Real_News_Frequency | Fake_News_Word | Fake_News_Frequency |
   | -------------- | ------------------- | -------------- | ------------------- |
   | Trump          | 5399                | Trump          | 7672                |
   | say            | 3222                | VIDEO          | 5354                |
   | House          | 1417                | Video          | 3015                |
   | Russia         | 912                 | Obama          | 1607                |
   | North          | 908                 | WATCH          | 1603                |
   | Korea          | 881                 | Hillary        | 1495                |
   | White          | 790                 | TRUMP          | 1019                |
   | China          | 766                 | OBAMA          | 869                 |
   | Senate         | 753                 | President      | 865                 |
   | bill           | 691                 | BREAKING       | 852                 |
   
   Unlike the fixed pattern of article content, some title letters are all capitalized and some are only capitalized. So we can see the condition that "Trump" and "TRUMP" have different frequencies. And since this has no assistance in telling the difference between real and fake news according to our observation, we have the following version B(apply this version instead of version A) to combine the frequency of the same terms by converting all of them to lowercase.
   
   Version B (applied)
   
   | Real_News_Word | Real_News_Frequency | Fake_News_Word | Fake_News_Frequency |
   | -------------- | ------------------- | -------------- | ------------------- |
   | trump          | 5406                | trump          | 8705                |
   | say            | 3223                | video          | 8449                |
   | house          | 1445                | obama          | 2476                |
   | republican     | 973                 | hillary        | 2189                |
   | north          | 925                 | watch          | 1905                |
   | russia         | 912                 | clinton        | 1136                |
   | korea          | 881                 | president      | 1108                |
   | new            | 862                 | get            | 936                 |
   | white          | 807                 | black          | 933                 |
   | state          | 803                 | new            | 895                 |
   
   The word cloud for real news
   
   (text)
   
   ![real_news_wordcloud.png](C:\Users\xy2fo\Desktop\DS595NLP\real_news_wordcloud.png)
   
   (title)
   
   ![real_news_title_wordcloud_B.png](C:\Users\xy2fo\Desktop\DS595NLP\real_news_title_wordcloud_B.png)
   
   The word cloud for fake news:
   
   (text)
   
   ![fake_news_wordcloud.png](C:\Users\xy2fo\Desktop\DS595NLP\fake_news_wordcloud.png)
   
   (title)
   
   ![fake_news_title_wordcloud_B.png](C:\Users\xy2fo\Desktop\DS595NLP\fake_news_title_wordcloud_B.png)

2. The difference between real news and fake news 
   
   Let us try to analyze the preprocessed textual data to see if we can tell the difference directly without using other advanced methods.
   
   <u>Finding 1</u>
   
   From the top 100 most commonly used words for real and fake news, the performance of the text and title is consistent. So we are going to discuss them separately.
   
   <u>Finding 2</u> (text)
   
   The real and fake news both have a high frequency of function words like "said" and "would". But the real one has a higher frequency.
   
   <u>Finding 3</u> (text)
   
   The fake news contains a higher frequency of the term "people" and also the name of public people like "Trump", "Clinton" and "Obama". While the real news contains a higher frequency of public media like "Reuters"(this is because news from Reuters will have a (Reuters) signal in front of the text) and also more formal terms like "government" and "President", in addition to public people name like "Trump".
   
   <u>Finding 4</u> (title)
   
   Similar to finding 3, the fake news titles contain a higher frequency of the names of public people. While the real news titles contain a higher frequency of more formal terms like "republican" and also countries like "russia", "korea" and "china".
   
   <u>Conclusion</u>
   
   So according to these two findings, news with a title like "Former Obama Photographer Takes Trolling Trump To A Whole New Level (TWEETS)" has a higher possibility of being a piece of fake news because it contains "Obama" and "Trump". 
   
   Likewise, the message "Instead, the government has been operating on a series of temporary measures" in a text has a higher possibility of being a piece of real news because it contains "government".
   
   But these are not necessarily true and merely a guess since these terms appear in both real and fake news. We can compare the differences in datasets but we are not easily sure whether a certain piece of news is real or not. That's one of the reasons why we need machine learning methods.

3. The strongest feature set
   
   We can use strong features like
   
   * TF: count the number of times each word appears in the document
   
   * TF-IDF: reflect the importance of words in a corpus
   
   * Sentiment scores: gauge the tone of the text
   
   * Part-of-speech tags: understand the grammatical structure
   
   * Word embeddings: such as Word2Vec or GloVe, provide rich semantic representation

### Task 2. Build Machine Learning Models

We use <u>TF-IDF(1)</u> combined with n-grams, <u>sentiment scores(2)</u> and <u>POS tags(3)</u> as our features and don't use word embeddings because features like this are too large. 

Also after testing on the first model Logistic Regression, the confusion matrix changes from [[6855 141] [ 109 6365]] to [[6846 150] [ 130 6344]], which means the performance degrades slightly. Besides, program execution time increases from 870s to 995s. So we drop TF(4) when we use other models.

And there's a little difference for RNNs and LSTM. We are supposed to use word embeddings, but this is too large for my gpu resource. So we just use the sequence of word indexes when we use them.

| Model               | Feature | Precision | Recall | Accuracy | Time(s) |
| ------------------- | ------- | --------- | ------ | -------- | ------- |
| Logistic Regression | 1234    | 0.98      | 0.98   | 0.98     | 995     |
| Logistic Regression | 123     | 0.98      | 0.98   | 0.98     | 870     |
| Naive Bayes         | 1       | 0.97      | 0.94   | 0.96     | 20      |
| Naive Bayes         | 123     | 0.99      | 0.34   | 0.68     | 1294    |
| SVM                 | 12      | 0.94      | 0.92   | 0.94     | 32418   |
| SVM                 | 123     | 0.94      | 0.92   | 0.94     | 33906   |
| CNNs(gpu)           | 12      | 0.98      | 0.98   | 0.98     | 230     |
| CNNs(gpu)           | 123     | 0.97      | 0.99   | 0.98     | 548     |
| RNNs(gpu)           | index   | 0.81      | 0.02   | 0.52     | 236     |
| LSTM(gpu)           | index   | 0.999     | 0.99   | 0.9965   | 594     |

LSTM wins. 

The Confusion Matrix: 

[[4648    2]
 [  29 4301]]

Besides, I believe the performance would be better if using word embeddings as features.

### Task 3. Enhanced NLP Features

We already applied POS tags in task 2. We can see some of the models didn't improve a lot. In this section, we will explore the influence of different POS information filters further.

For the reason of program execution time, we only use Logistic Regression and LSTM for the test.

| Model               | Feature  | Filter     | Precision | Recall | Accuracy |
| ------------------- | -------- | ---------- | --------- | ------ | -------- |
| LSTM                | index, 3 | N, J       | 0.99      | 0.98   | 0.992    |
| LSTM                | index, 3 | N, J, V    | 0.99      | 0.98   | 0.989    |
| LSTM                | index, 3 | N, J, V, R | 0.98      | 0.99   | 0.989    |
| LSTM                | index, 3 | V          | 0.91      | 0.84   | 0.88     |
| Logistic Regression | 123      | N, J       | 0.98      | 0.98   | 0.98     |
| Logistic Regression | 123      | N, J, V    | 0.98      | 0.98   | 0.98     |
| Logistic Regression | 123      | N, J, V, R | 0.98      | 0.98   | 0.98     |

As a result, there is not any significant improvement after applying POS tagging. This is might because all kinds of words in the collection matter to the classification in terms of this dataset.

### Task 4. Future Work

In the future, I would like to apply word embeddings to LSTM because of its outstanding performance. Due to the limit of my gpu resources, I didn't apply this in this assignment. Also, the result of the RNN model should be better. This model needs to be fine-tuned later.

It's impossible so far to use GPT api to assist with this task since the job of GPT is to generate text not to provide a cloud server for computing. But we can use GPT api to do other jobs. For example, I used to use GPT api to reply to messages automatically on WeChat.
