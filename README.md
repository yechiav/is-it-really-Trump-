# is it really Trump??
Trump tweets detection and other NLP goodies

![alt text](https://foter.com/photos/398/people-holding-trump.jpg)

Photo on [Foter.com](https://foter.com/)

see deatiled report:

[Report](https://github.com/yechiav/is-it-really-Trump-/blob/master/Iis_It_Real_Trump-report.pdf "Report")

## Task:detection of Real trump user by tweets content
in this task we proved that in simple NLP tools it is possible to detect Trump by his tweets contant, 
the dataset is couple of thousands tweets from Trump’s account posted between early
2015 and mid 2017.

#### process:

![alt text](https://github.com/yechiav/is-it-really-Trump-/blob/master/Process.JPG)

before the resutls, it can be quite easily seen that even with only looking at the time of the tweets, its pretty clear who is the real trump...
![alt text](https://github.com/yechiav/is-it-really-Trump-/blob/master/Time.png)

i am expecting good results

### Results
and they are, all models and configuration yileded very promising resutls
![alt text](https://github.com/yechiav/is-it-really-Trump-/blob/master/results.JPG)

## Task: clustering and detection of artifical BOT genereted FCC comments
Background: the Federal Communication Commision is the federal agency in charge of
regulating interstate communication channels such as radio, TV, cable and the internet. Citizens
can sign petitions and post comments in support/against proposed regulations. Net Neutrality is
a major regulatory issue that will be decided this December. It was ​recently claimed
(recommended reading!) that many of the comments opposing net neutrality that were
submitted to the FCC are not authentic and were submitted by bots that used simple linguistic
manipulations in order to appear authentic

in this part we will cluster the comments and suggest clusters of non authentic comments

#### process:
data cleaening: stop wrods removal, and stemming
feature extraction: TF-IDF 
Topic modeling: clustering using Latent Dirichlet allocatio
clsutering: taking the affiliation of each message to a the topics and clustering them

### Results
![alt text](https://github.com/yechiav/is-it-really-Trump-/blob/master/Topic_modeling.JPG)

when clutering the topics and reviewing teh resutls we can see a good clsutering suggesting a limited number of topic the messages concerns

and we can easliy detect that one of the topic is clearly contains too similiar commnets, suggesting it was computer generated

![alt text](https://github.com/yechiav/is-it-really-Trump-/blob/master/non_authentic.JPG)


### prerequisites
- seaborn 
- Pandas
- Numpy
- WordCloud
- nltk
