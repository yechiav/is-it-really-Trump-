
from collections import Counter

import seaborn as sns
import time
import string
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
import timeit
from datetime import datetime
from sklearn import metrics
import matplotlib.pyplot as plt
import matplotlib
from nltk.corpus import stopwords
from nltk.stem.porter import *
from sklearn.manifold import TSNE
import csv
from numpy  import vstack,array
from wordcloud import WordCloud
from scipy.sparse import csr_matrix


Task = ["TaskA","TaskB"] #this are the optionss
Task = "TaskA"

Task2_b_datatoload = ["tweets","FCC"] #this are the optionss
Task2_b_datatoload = "tweets"

Print_TSNE = False
Print_Files = False
Print_heatmap = False
print_CloudWord = False




devices_list = ["android","iphone"]
user_list = ["realDonaldTrump"]
none_feat = ['UID','User','Message','Time','Device','OriginalM','weekday']

none_feat_cat = ['UID','User','Message','Time','Device','OriginalM','weekday','weekend','hour_Bin']


get_dummies_feat = ['weekend','hour_Bin']

file = "tweets.tsv"

cut_points = [6,12,17]
labels = ["night","morning","afternoon","evening"]

font = {'size'   : 10}

matplotlib.rc('font', **font)



feat_cols = lambda x: [item_x for item_x in list(x) if item_x not in none_feat]

feat_cols_no_cat = lambda x: [item_x for item_x in list(x) if item_x not in none_feat_cat]

mytokenize = lambda doc: doc.lower().split(" ")

def weekend(day):
    if day>4:
        return "c_weekend"
    else:
        return "c_dayweek"

def print_df_feature(df,feature=0,file_name="output_feature.xls"):
    df2=df
    if(feature!=0):
        df2=df[feat_cols(df)]

    writer = pd.ExcelWriter(file_name)
    df2.to_excel(writer, 'Tweets')
    writer.save()

def renmove_stop_words(string1):
    tokens= mytokenize(string1)
    stopWords= stopwords.words('english')
    filtered = [w for w in tokens if not w in stopWords]
    string1 = " ".join(filtered)
    return string1

#Binning:
def binning(col, cut_points, labels=None):
  #Define min and max values:
  minval = col.min()
  maxval = col.max()

  #create list by adding min and max to cut_points
  break_points = [minval] + cut_points + [maxval]

  #if no labels provided, use default labels 0 ... (n-1)
  if not labels:
    labels = range(len(cut_points)+1)

  #Binning using cut function of pandas
  colBin = pd.cut(col,bins=break_points,labels=labels,include_lowest=True)
  return colBin


def PrintTsne(df):

    df=df.reindex()
    time_start = time.time()
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)



    tsne_results = tsne.fit_transform(df[feat_cols_no_cat(df)])

    print ('t-SNE done! Time elapsed: {} seconds'.format(time.time() - time_start))

    # df_tsne = df.loc[rndperm[:n_sne], :].copy()
    df_tsne=pd.DataFrame()
    df_tsne['x-tsne'] = tsne_results[:, 0]
    df_tsne['y-tsne'] = tsne_results[:, 1]




    plt.rcParams['figure.figsize'] = (20.0, 5.0)
    sns.set(style="ticks")
    sns.lmplot(x="x-tsne", y="y-tsne", data=df_tsne, fit_reg=False)
    plt.title("TSNE map")
    plt.show()
    return df_tsne

def feature_extract_before_clean_(df):
    df['OriginalM'] = df['Message']


    df["hour"] = tweets["Time"].apply(lambda x: x.hour)
    df["day"] = tweets["Time"].apply(lambda x: x.weekday())
    df["dayofmonth"] = tweets["Time"].apply(lambda x: x.day)
    df["month"] = tweets["Time"].apply(lambda x: x.month)

    df["hour_Bin"] = binning(df["hour"], cut_points, labels)


    df['capital_letters'] = df['Message'].apply(lambda x: countCaptial(x))
    df['capital_percent'] = df['Message'].apply(lambda x: PrecentCapital(x))
    df['weekday'] = df['Time'].apply(lambda x: x.weekday())
    df['weekend'] = df['weekday'].apply(lambda x: weekend(x))
    df['links'] = df['Message'].apply(lambda x: countlink(x))
    df['length'] = df['Message'].apply(lambda x: len(x))
    df['num_hashtags'] = df['Message'].apply(lambda x: num_hashtags(x))
    df['num_mentions'] = df['Message'].apply(lambda x: num_mentions(x))


    return df

def stemming(sentance):
    stemmer = PorterStemmer()
    tokens  = mytokenize(sentance)
    sentance2 = " ".join([stemmer.stem(token) for token in tokens])
    return sentance2

def feature_extract_after_clean_TaskA(df):

    df['Message'] = df['Message'].apply(lambda x: remove_links(x))
    df['Message'] = df['Message'].apply(lambda x: str(x).lower())
    # df['Message'] = df['Message'].apply(lambda x: re.sub(r'@\S+', ' ', x)) #renove user mentino
    # # df['Message'] = df['Message'].apply(lambda x: re.sub(r'#\S+', ' ', x)) #remove hashtags
    df['Message'] = df['Message'].apply(lambda x: x.translate(None, string.punctuation)) #remove puntication
    df['Message'] = df['Message'].apply(lambda x: renmove_stop_words(x)) #remove stop words
    df['Message'] = df['Message'].apply(lambda x: ''.join([i for i in x if not i.isdigit()])) #remove numbers
    df['Message'] = df['Message'].apply(lambda x: stemming(x))



    minmax = MinMaxScaler()

    list1=feat_cols_no_cat(df)
    df_minmanx = minmax.fit_transform(df[list1])
    df_minmanx = pd.DataFrame(df_minmanx)

    df[list1]=df_minmanx

    for i in get_dummies_feat:
        df1=pd.get_dummies(df[i])
        df = pd.concat([df, df1], axis=1)


    TF_IDFDENSE = tfidf(df)
    df = pd.concat([df, TF_IDFDENSE], axis=1)


    return df

def feature_extract_after_clean_TaskB(df):

    df['Message'] = df['Message'].apply(lambda x: remove_links(x))
    df['Message'] = df['Message'].apply(lambda x: str(x).lower())
    df['Message'] = df['Message'].apply(lambda x: re.sub(r'@\S+', ' ', x)) #renove user mentino
    # df['Message'] = df['Message'].apply(lambda x: re.sub(r'#\S+', ' ', x)) #remove hashtags
    #df['Message'] = df['Message'].apply(lambda x: x.translate(None, string.punctuation)) #remove puntication
    df['Message'] = df['Message'].apply(lambda x: renmove_stop_words(x)) #remove stop words
    df['Message'] = df['Message'].apply(lambda x: ''.join([i for i in x if not i.isdigit()])) #remove numbers
    df['Message'] = df['Message'].apply(lambda x: stemming(x))
    #TOCO add back the trasnalete part

    return df


def uniquevalues(df):
    for column in df.columns:
        print(column)
        print("column name: {0} and diffrent values {1} and dtype:{2}".format(column, len(df[column].unique()) ,type(df[column])))

def uniquevalues_count(df,unique=5):
    above=[]
    bellow=[]
    for column in df.columns:
        uniq=len(df[column].unique())
        if uniq>unique:
            above.append(column)
        else:
            bellow.append(column)
    print("Above:")
    print(above)
    print("Bellow")
    print(bellow)

def RemvoeNoneUniqueCol(df, unique=5):
    above = []
    bellow = []
    for column in df.columns:
        uniq = len(df[column].unique())
        if uniq > unique:
            above.append(column)
        else:
            bellow.append(column)
    df=df[above]
    # uniquevalues(df)
    return df

def tfidf(df):
    no_features=1000
    tfidf_tweets = df['Message'].values


    tfidf_vectorizer = TfidfVectorizer()
    tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, max_features=no_features, stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(tfidf_tweets)

    # Convert our matrix to a dense matrix and convert to a DataFrame, adding the actual column names
    dense = pd.DataFrame(tfidf_matrix.todense(), columns=tfidf_vectorizer.get_feature_names())
    # dense = pd.DataFrame(tfidf_matrix, columns=tfidf_vectorizer.get_feature_names())


    dense.columns = ["l_"+str(col)  for col in dense.columns]

    dense=RemvoeNoneUniqueCol(dense,5)


    # output_tweets = pd.merge(df,dense)
    output_tweets = pd.concat([df, dense], axis=1)

    print("df rows: {0} and dense rows {1} and output_tweets rows: {2}".format(df.shape[0],dense.shape[0],output_tweets.shape[0]))

    return dense

def countCaptial(string1):
    str1 = str(string1)
    if type(string1) is str:
        return len(re.findall(r'[A-Z]',str1))
    else:
        return 0

#TODO: understand why i should turn it into string
def PrecentCapital(string1):
    str1 = string1
    if type(string1) is str:
        capital = len(re.findall(r'[A-Z]',str1))
        lower = len(re.findall(r'[a-z]',str1))
        return capital / float(capital + lower)
    else:
        return 0

def countlink(string1):
    str1 = str(string1)
    urls = re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', str1)
    return len(urls)

def remove_links(string1):
    str1 = str(string1)
    str1 = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ' ', str1)
    return str1

def CleanUser(df):
    list1 = list(df['User'].values)
    set1 = set(list1)
    dict1 = Counter(df['User'].values)
    print(dict1)

def num_hashtags(s):
    return sum(1 for c in s if c == "#")

def num_mentions(s):
    return sum(1 for c in s if c == "@")



def printHeatMap(df):
    # Heatmap of Trump Tweets
    a = []
    values = set(df["Device"].values)
    # f, (a[1], a[2]) = plt.subplots(2,1, sharey=True)

    i_1=0
    for S_values in values:
        i_1+=1
        value_matrix = df[df["Device"] == S_values][["day", "hour", "UID"]]
        value_matrix = value_matrix.groupby(["day", "hour"]).count().reset_index()
        value_matrix = value_matrix.pivot("day", "hour", "UID")
        value_matrix = value_matrix.fillna(0)

        for i in set(np.arange(0, 24)) - set(value_matrix.columns):  # Fill in missing hours
            value_matrix[i] = 0

        value_matrix = value_matrix.sort_index(axis=1)
        value_matrix

        plt.subplot(plt.subplot(2, 1, i_1))
        plt.rcParams['figure.figsize'] = (20.0, 5.0)
        sns.heatmap(value_matrix, cbar_kws={"orientation": "horizontal"});
        plt.title("Distribution of Number of Tweets, Device = "+S_values);


    plt.show()



def CleanDF(df):

    df = df.drop(df[~df['Device'].isin(devices_list)].index)


    df =    df.drop(df[~df['User'].isin(user_list)].index)

    df = df.reset_index(drop=True)



    # TODO: Remove this part of length
    return df



def FeatSelect(df,numFeatures=20):
    # load the iris datasets
    dataset = df[feat_cols_no_cat(df)]

    model = LogisticRegression()
    # create the RFE model and select 3 attributes
    rfe = RFE(model, numFeatures)
    rfe = rfe.fit(dataset, df['Device'])


    filtered = np.array(list(dataset.columns))[np.array(rfe.support_)]


    return filtered



def findNan(df):
    # nan = df.isnull().any
    # print(nan)
    for column in df:
        i=0
        print((column,df[column].hasnans) )

def model_tunning(X_train, X_test, y_train, y_test,model,params,features_count,name):

    df_scores=pd.DataFrame()
    df_scores_inline=pd.DataFrame()
    # scores = ['precision', 'recall']

    scores = ['precision']

    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        print()

        clf = GridSearchCV(model, params, cv=5,
                           scoring='%s_macro' % score)
        clf.fit(X_train, y_train)

        print("Best parameters set found on development set:")
        print()
        print(clf.best_params_)
        print()
        print("Grid scores on development set:")
        print()
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        i=0
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            i+=1
            #TODO return print
            # print("%0.3f (+/-%0.03f) for %r"
            #       % (mean, std * 2, params))
            its_dict = {"model":name,"features":features_count,"mean":mean,"std":std * 2}
            its_dict = dict(its_dict, **params)
            its_padnas = pd.DataFrame(its_dict, index=[i])
            df_scores_inline = df_scores_inline.append(its_padnas.copy(),ignore_index =True)
            # print(df_scores_inline)
            #TODO fin why scores includes only last scores
        print(df_scores)
        print()

        print("Detailed classification report:")
        print()
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
        print()
        y_true, y_pred = y_test, clf.predict(X_test)
        print(classification_report(y_true, y_pred))
        print()

        y_pred = pd.Series(y_pred, index=y_true.index)
        test_answer = y_pred== y_true
        test_answer=test_answer.apply(lambda x: "Correct" if x  == True  else "Wrong" )



    return clf.best_params_, test_answer , df_scores_inline

def model_tunning_helper(df):

    global Print_Files
    number_of_features = [5,10,20,40,70]

    SVC_params = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                         'C': [1, 10, 100, 1000]},
                        {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

    LogRegParams = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}

    df_answers=pd.DataFrame()



    df_scores_all=pd.DataFrame()

    models_tuple=[]  #[name,model,param]
    names  = ["SVC","LogisticRegression"]
    models=[SVC(),LogisticRegression()]
    Params=[SVC_params,LogRegParams]

    models_tuple = zip(names,models,Params)


    X = tweets
    y = tweets["Device"]
    X_train_orig, X_test_orig, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    feature_dict ={}
    for features_count in number_of_features:

        features_selected = FeatSelect(tweets,features_count)
        print(features_selected)
        feature_dict[features_count] = features_selected
        X_train =   X_train_orig[features_selected]
        X_test =    X_test_orig[features_selected]


        print("------------------------------------------------------------------")
        print("--------feautres=---{0}-------------------------------------------".format(features_count))
        print("------------------------------------------------------------------")

        i=10
        for pos, value in enumerate(y_test):
           print("{} , {}".format(pos,value))
           i-=1
           if i==0:
            break



        for item in models_tuple:
            name=item[0]
            model=item[1]
            params=item[2]
            print("------------------------------------------------------------------")
            print("--------name=-------------{0}---------------------------------".format(name))
            print("------------------------------------------------------------------")

            params, ps_answer, df_scores= model_tunning(X_train, X_test, y_train, y_test,model,params,features_count,name)
            title = str(features_count)+" " +name + ' ' + str(params)
            ps_answer.rename(title)
            df_answers[title] = ps_answer

            print("here")
            print(title)
            print(df_scores)

            df_scores_all = df_scores_all.append(df_scores,ignore_index=True)

    #print features

    if Print_Files:
        df_scores_all.to_csv("Scores.csv")

    df_features_selections = pd.DataFrame.from_dict(feature_dict,orient='index').T
    if Print_Files:
        df_features_selections.to_csv("featureselect.csv")
    return df_answers



#--------------------------------------------------------------------------
# Task 2 functions
def display_topics(model_component, feature_names, no_top_words,mode="toScreen"):
    df=pd.DataFrame()
    for topic_idx, topic in enumerate(model_component):
        if(mode=="toScreen"):
            print( "Topic %d:" % (topic_idx))
            print( " ".join([feature_names[i]
                            for i in topic.argsort()[:-no_top_words - 1:-1]]))
        else:
            Topic    =     "Topic %d:" % (topic_idx)
            topwrods =      [feature_names[i]
                            for i in topic.argsort()[:-no_top_words - 1:-1]]
            df = df.append({'Topic': Topic,
                       'topwrods': topwrods}, ignore_index=True)
    if(mode!="toScreen"):
        # df.to_csv("displaytopics.csv")
        return df

def display_topics_wordcloud(model_component, feature_names, no_top_words):
    df=pd.DataFrame()
    zipped = []
    i_1=1
    for topic_idx, topic in enumerate(model_component):
            Topic    =     "Topic %d:" % (topic_idx)
            topwrods =      [feature_names[i]
                            for i in topic.argsort()[:-no_top_words - 1:-1]]
            frequencies = [topic[i]
                            for i in topic.argsort()[:-no_top_words - 1:-1]]
            # frequencies = frequencies.apply(lambda  x: int(round(x*1000)))
            frequencies = [int(round(x*1000)) for x in frequencies]
            zipped = zip(topwrods,frequencies)

            zip_dic = dict(zipped)


            wordcloud = WordCloud(width=1000, height=500).generate_from_frequencies(zip_dic)

            plt.subplot(3, 2, i_1)
            plt.imshow(wordcloud)
            plt.axis("off")
            plt.title(Topic)

            if(i_1==6):
                break

            i_1+=1

    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    plt.savefig("wordcloud.png", bbox_inches='tight',dpi=200)
    plt.show()


    return df

def loadData(mode):

    documents = "x"
    if(mode=="tweets"):
        file_tweet = "tweets.tsv"



        list_of_all_tokens = re.split(r'[\t\n]+', open(file_tweet, 'r').read())
        nparray_alltokens = np.array(list_of_all_tokens)
        if nparray_alltokens[-1] == '':
            nparray_alltokens = nparray_alltokens[:-1]
        tweets = pd.DataFrame(nparray_alltokens.reshape(-1, 5),
                              columns=['UID', 'User', 'Message', 'Time', 'Device'])


        tweets = CleanDF(tweets)
        tweets = feature_extract_after_clean_TaskB(tweets)

        documents =tweets['Message'].tolist()
        device =[]
        device =tweets['Device'].tolist()
        return documents, device
    else:
        data_arr = []
        print("f")
        file_tweet = "proc_17_108_unique_comments_text_dupe_count.csv"
        LoadPrecentage = 0.3

        with open(file_tweet, 'r', encoding="utf8") as fin:
            c=0
            rows_count=0
            reader = csv.reader(fin, delimiter=',', quotechar='"')
            for row in reader:
                c += 1
                if(rows_count<LoadPrecentage/100*c):
                    data_arr.append([x for x in row])
                    rows_count += 1
            print ("number of row:{0}".format(c))
            print("loading {0}% of data...".format(LoadPrecentage))
            print("{0} rows loaded".format(rows_count))
            data = vstack(data_arr)
            documents = data[:, 1]
        return  documents


def save_sparse_csr(filename,array):
    np.savez(filename,data = array.data ,indices=array.indices,
             indptr =array.indptr, shape=array.shape )

def load_sparse_csr(filename):
    loader = np.load(filename)
    return csr_matrix((  loader['data'], loader['indices'], loader['indptr']),
                         shape = loader['shape'])

def scatterPlot2(df,list_clf,title , silhouetteScroes):

    i=1
    for item in list_clf:
        model_name = item[1]
        plt.subplot(3,2,i)
        plt.scatter(df['X_TSNE'], df['Y_TSNE'], c=df[model_name], cmap='rainbow',alpha=0.5 ,s=1)
        silute_score= str(round(silhouetteScroes[model_name],4))
        title1  = model_name + "\nsilhouetteScroe: " + silute_score
        plt.title(title1)
        plt.xlabel("TSNE_X")
        plt.ylabel("TSNE_Y")

        i+=1

    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    filename=title+"_TSNE.png"
    plt.savefig(filename, bbox_inches='tight', dpi=200)
    # plt.show()


    return 0

def returnTSNE(df):

    X_embedded = TSNE(n_components=2).fit_transform(df)
    df= pd.DataFrame(X_embedded,columns=["X_TSNE","Y_TSNE"])

    return df

def cluster_data(df,model,name):
    clustering = model.fit_predict(df)
    df_clusteing = pd.DataFrame({name: clustering})

    return df_clusteing, clustering




if __name__ == "__main__":

    # Task = ["TaskA", "TaskB"]

    global Task
    global Print_TSNE
    global Print_Files
    global print_CloudWord

    if Task == "TaskA":
        tweets = pd.read_table(file,names=['UID','User','Message','Time','Device'],sep='\t',parse_dates=True,header=None)

        # Read Tweets
        list_of_all_tokens = re.split(r'[\t\n]+', open(file, 'r').read())
        nparray_alltokens = np.array(list_of_all_tokens)
        if nparray_alltokens[-1] == '':
            nparray_alltokens = nparray_alltokens[:-1]
        tweets = pd.DataFrame(nparray_alltokens.reshape(-1, 5),
                              columns=['UID','User','Message','Time','Device'])
        tweets["Time"] = pd.to_datetime(tweets["Time"], format='%Y-%m-%d %H:%M:%S')



        tweets = feature_extract_before_clean_(tweets)
        tweets = CleanDF(tweets)

        if Print_heatmap:
            printHeatMap(tweets)

        tweets = feature_extract_after_clean_TaskA(tweets)



        if Print_TSNE:
            df_tsne = PrintTsne(tweets)
            tweets = pd.concat([tweets,df_tsne], axis=1)

        df_answers = model_tunning_helper(tweets)

        tweets = pd.concat([tweets,df_answers],axis=1)



        if Print_TSNE:
            tweets = pd.concat([tweets, df_tsne], axis=1)

        if Print_Files:
          tweets.to_csv('after_fet.csv')


    if Task == "TaskB":

        startTime = datetime.now()

        mode = "tweets"
        # mode = "else"

        load_mode = "FromFreash"
        # load_mode="FromStorage"

        no_features = 1000
        no_topics = 6
        no_top_words = 15

        data_arr = []
        filename = "CSR_matrix.npz"
        tf_feature_names_file = "tf_feature_names.npy"
        tsne_pickle = 'tsne.pickle'
        model_component_file = "mode_component.npy"
        tf_trsansfromred_file = "tf_trsansfromred_file.npy"

        tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=no_features, stop_words='english')

        start = timeit.timeit()
        print("loadMode: {0}".format(load_mode))

        if load_mode == "FromFreash":
            documents, device = loadData(mode)

            df = pd.DataFrame(documents)
            df_device = pd.DataFrame(device)
            df = pd.concat([df, df_device], axis=1)
            tf = tf_vectorizer.fit_transform(documents)

            lda = LatentDirichletAllocation(n_topics=no_topics, max_iter=5, learning_method='online',
                                            learning_offset=50.,
                                            random_state=0).fit(tf)

            tf_transformed = lda.transform(tf)

            model_component = lda.components_

            np.save(tf_trsansfromred_file, tf_transformed)

            np.save(model_component_file, model_component)  # save LDA component
            saved_data = save_sparse_csr(filename, tf)  # save TF output data

            del tf
            del model_component
            del tf_transformed

            tf = load_sparse_csr(filename)
            model_component = np.load(model_component_file)
            tf_transformed = np.load(tf_trsansfromred_file)

            tf_feature_names = tf_vectorizer.get_feature_names()
            np.save(tf_feature_names_file, tf_feature_names)
            del tf_feature_names
            tf_feature_names = np.load(tf_feature_names_file)
            print('Time elpased \tTF LDA \t  (hh:mm:ss.ms) {}'.format(datetime.now() - startTime))

        else:
            df = pd.DataFrame()
            tf = load_sparse_csr(filename)
            tf_feature_names = np.load(tf_feature_names_file)
            model_component = np.load(model_component_file)
            tf_transformed = np.load(tf_trsansfromred_file)
            print('Time elpased \tloading data from HD\t  (hh:mm:ss.ms) {}'.format(datetime.now() - startTime))

        transfromed_tf = tf.todense()

        dataTocluter = transfromed_tf
        # dataTocluter = model_component



        list_clf = []
        Kmeans_cluster = [2, 6, 8]
        DBscan_eps = [0.3, 0.4, 0.5]
        DBscan_eps = [0.2]
        DBscan_min_sample = [10, 20, 30]
        DBscan_min_sample = [10, 20, 30]

        DBscan_min_sample = [i * 0.5 for i in DBscan_min_sample]

        for cluster in Kmeans_cluster:
            model = KMeans(n_clusters=cluster, random_state=42)
            name = "kMeans_" + str(cluster)
            list_clf.append((model, name))
        for eps in DBscan_eps:
            for minsample in DBscan_min_sample:
                model = DBSCAN(eps=eps, min_samples=minsample)
                name = "DBSCan_" + str(eps).replace('.', '') + "_" + str(minsample)
                list_clf.append((model, name))

        if load_mode == "FromFreash":
            df_xy = returnTSNE(dataTocluter)
            df_xy.to_pickle(tsne_pickle)

            # test
            del df_xy
            df_xy = pd.read_pickle(tsne_pickle)
        else:
            df_xy = pd.read_pickle(tsne_pickle)

        df = pd.concat([df, df_xy], axis=1)

        print('Time elpased \tTSNE\t  (hh:mm:ss.ms) {}'.format(datetime.now() - startTime))
        silhouetteScroe_s = dict()

        for clf in list_clf:
            model = clf[0]
            model_name = clf[1]
            df1, y = cluster_data(dataTocluter, model, model_name)
            print(
            'Time elpased \tClustering Model:\t {}\t  (hh:mm:ss.ms) {}'.format(model_name, datetime.now() - startTime))

            silhouetteScroe = metrics.silhouette_score(dataTocluter, y)
            silhouetteScroe_s[str(model_name)] = silhouetteScroe
            print("Silhouette Coefficient: %0.3f"
                  % silhouetteScroe)

            print(
            'Time elpased \tSilhouette Model:\t {}\t  (hh:mm:ss.ms) {}'.format(model_name, datetime.now() - startTime))
            df = pd.concat([df, df1], axis=1)
        #
        filename_plot = "test_data"

        print('Time elpased \ttotal clustering clustering\t  (hh:mm:ss.ms) {}'.format(datetime.now() - startTime))

        print('Time elpased \tclustering\t  (hh:mm:ss.ms) {}'.format(datetime.now() - startTime))

        scatterPlot2(df, list_clf, filename_plot, silhouetteScroe_s)

        topic_df = display_topics(model_component, tf_feature_names, no_top_words, mode="toDF")
        if print_CloudWord:
            display_topics_wordcloud(model_component, tf_feature_names, no_top_words)

        df.to_csv("results_data.csv")
        topic_df.to_csv("topics.csv")
        print('Time elpased \tAll App\t  (hh:mm:ss.ms) {}'.format(datetime.now() - startTime))