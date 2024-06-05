import os
import re
import json
import pandas as pd
import numpy as np
import statistics
import matplotlib.pyplot as plt
import seaborn as sns
import plotly
import plotly.express as px
from warnings import simplefilter
from wordcloud import WordCloud
from cleantext import clean
from tqdm import tqdm
from functools import wraps
from transformers import AutoTokenizer, AutoModelForSequenceClassification #Tokenizer and roBERTa
from scipy.special import softmax
from pprint import pprint
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
nltk.downloader.download('vader_lexicon')
nltk.download('stopwords')
nltk.download('punkt')
stopwordsList = set(stopwords.words("english"))

simplefilter("ignore")

pd.set_option("display.max_columns", None)
viridisColorScale = sns.color_palette("viridis")
magmaColorScale = sns.color_palette("magma")
crestColorScale = sns.color_palette("crest")
flareColorScale = sns.color_palette("flare")
pastelColorScale = sns.color_palette("pastel")


#Defining a decorator to save plots
def savePlots(plotFunction):

    def checkPlots(plotNames, plots):
        if isinstance(plotNames, list) and isinstance(plots, list):
            return True
        else:
            #print("\033[91mCheckPlots: object obtained are not lists\033[0m")
            return False

    def checkPlotsTypeAndSave(plotName, plots, filePath):
        if isinstance(plots, (plt.Figure, plt.Axes, sns.axisgrid.FacetGrid, sns.axisgrid.PairGrid, list)):
            plt.savefig(f"{filePath}{plotName}.png", dpi=300)
            print(f"{plotName} Exported Correctly")

        elif isinstance(plots, plotly.graph_objs._figure.Figure):
            plots.write_html(f"{filePath}{plotName}.html")
            print(f"{plotName} Exported Correctly")

        else:
            try:
                plt.savefig(f"{filePath}{plotName}.png", dpi=300)
                print(f"{plotName} Exported Correctly")
            except:
                print("\033[91mExporting the plots wasn't possible\033[0m")

        return None

    @wraps(plotFunction)
    def wrapper(*args, **kwargs):

        plotsNames, generatedPlots, filePath = plotFunction(*args, **kwargs)
        #print("File path: " + filePath)

        if checkPlots(plotsNames, generatedPlots) is True:

            for plotName, plot in zip(plotsNames, generatedPlots):
                checkPlotsTypeAndSave(plotName, plot, filePath)

        elif checkPlots(plotsNames, generatedPlots) is False:
            #print("Saving Single Graph...")
            checkPlotsTypeAndSave(plotsNames, generatedPlots, filePath)

        else:
            print(f"\033[91mExporting the plots wasn't possible, here's the data types obtained by the decorator: PlotNames: {type(plotsNames)}, Generated Plots (could be a list of plots): {type(generatedPlots)}, File Path: {type(filePath)}\033[0m")

        return None

    return wrapper


class dataManager:

    def __init__(self, ep, filename):
        self.episode = ep
        self.filename = filename
        self.plotsPath = f"./OutputData/HPE{ep}OutputData/{ep}-Plots/"
        self.charactersName = {
            "1": "Ahn Jeong-won",
            "2": "Kim Jun-wan",
            "3": "Lee Ik-joon",
            "4": "Yang Seok-hyung",
            "5": "Chae Song-hwa",
            "0": "Secondary"
        }

    def __repr__(self) -> str:
        return f"Data Available for Episode: {self.episode}\n"


    def createDirs(self) -> None:
        os.makedirs(f"./OutputData/HPE{self.episode}OutputData/{self.episode}-Plots/", exist_ok=True)
        os.makedirs(f"./OutputData/HPE{self.episode}OutputData/{self.episode}-Data/", exist_ok=True)
        return None

    @staticmethod
    def removeStopwords(text: str) -> str:

        tokenizedText = word_tokenize(text)
        cleanText = [w for w in tokenizedText if w not in stopwordsList]
        cleanText = " ".join(cleanText)
        
        return cleanText
    

    @staticmethod
    def wordFrequencyAnalysis(df: pd.DataFrame) -> pd.DataFrame:

        allSentences = df["Text"] #All Sentences DF

        fullText = " ".join(allSentences)

        pattern = r"\w+"

        f = open("stopwordsEng.txt", encoding='utf8')
        stopwords = f.readlines()
        stopwords = [w.strip() for w in stopwords]
        f.close()

        words = re.findall(pattern, fullText)
        
        cleanedWords = [w for w in words if w not in stopwords]

        wordsDict = {}

        for i in cleanedWords:
            wordsDict[i] = cleanedWords.count(i)

        wordsDF = pd.DataFrame.from_dict(wordsDict, orient='index', columns=['wordsCount'])
        wordsDF = wordsDF.sort_values(['wordsCount'], ascending=False)

        #print(wordsDF)

        return wordsDF


    @savePlots
    def generateWordcloud(self, df: pd.DataFrame): #Words DataFrame

        wcloud = WordCloud(background_color="white", max_words=50, width=1600, height=900)
        wcloud.generate_from_frequencies(df.to_dict()["wordsCount"])

        plt.figure(figsize=(16, 9))
        plt.imshow(wcloud)
        plt.axis("off")
        plt.title(f"Wordcloud for Episode {self.episode}", fontdict=dict(weight="bold", size=20))

        return f"EP{self.episode}WordCloud", plt, self.plotsPath


    def readData(self) -> pd.DataFrame:

        data = pd.read_csv(self.filename, encoding="utf-8", index_col=False)

        #print("\nData Summary: \n", data.describe(), "\n")
        print("\nCharacter Column's NaN Sum (Secondary Characters Sentences): ", data["Character"].isna().sum(), "\n")

        data["Character"] = data["Character"].fillna(0) #All the sentences that aren't said by the 5 main characters are classified as from "secondary" characters

        data["Character"] = data["Character"].astype("int16", errors="raise")
        data["Character"] = data["Character"].astype(str, errors="raise")

        print(data["Character"].unique())

        print("\nColumns Data Types:")
        print(data.dtypes, "\n\n")

        #Defining characters names from numbers
        data["Character"] = data["Character"].apply(lambda x: self.charactersName[x]) #Setting the names (in the tv series) for each character

        data["Text"] = data["Text"].apply(lambda x: x.replace("-", "")) #Removing dashes from sentences
        data["Text"] = data["Text"].apply(lambda x: clean(x.lower(), no_emoji=True, to_ascii=True)) #Cleaning text removing emojis and converting every character into the closest ascii one
        data["Text"] = data["Text"].apply(lambda x: self.removeStopwords(x))

        FrequentWords = self.wordFrequencyAnalysis(data) #Gets only the first 10 elements of the ordered dictionary obtained by the self.wordFrequencyAnalysis function

        print("TOP 10 Most Frequent Words: \n", FrequentWords.head(10), "\n")

        self.generateWordcloud(FrequentWords)


        return data



class sentimentAnalyzer:

    def __init__(self, ep, data):
        self.data = data
        self.episode = ep
        self.plotsPath = f"./OutputData/HPE{ep}OutputData/{ep}-Plots/"
        self.charactersName = {
            "1": "Ahn Jeong-won",
            "2": "Kim Jun-wan",
            "3": "Lee Ik-joon",
            "4": "Yang Seok-hyung",
            "5": "Chae Song-hwa",
            "0": "Secondary"
        }
        self.episodesInfo = {
            "1": {"Description": "Five friends whose friendship goes back to their days in med school are brought back together when a phone call interrupts each of their busy lives.", "Duration": "4812"},
            "2": {"Description": "For the first time in 20 years, everyone's finally working under the same roof. A patient with a familiar name finds Chae Song Hwa.", "Duration": "4829"},
            "3": {"Description": "The hospital's med students are inspired by the doctors, who each look after their patients in their own ways. Lee Ik Joon gets a welcome visitor.", "Duration": "5177"},
        }

    def __repr__(self) -> str:
        return f"Data Available for Episode: {self.episode}"


    @staticmethod
    def vaderAnalyze(data: pd.DataFrame) -> pd.DataFrame:
        sia = SentimentIntensityAnalyzer()

        scores = {}
        for i, row in tqdm(data.iterrows(), total=len(data)):
            text = row["Text"]
            revIndex = row["SentenceIndex"] #Unique index of the sentence

            scores[revIndex] = sia.polarity_scores(text)


        vaderScores = pd.DataFrame(scores).T
        vaderScores = vaderScores.reset_index().rename(columns={"index": "SentenceIndex", "neg": "vaderNeg", "neu": "vaderNeu", "pos": "vaderPos", "compound": "vaderCompound"})

        return vaderScores
    

    @staticmethod
    def robertaAnalyze(data: pd.DataFrame) -> pd.DataFrame:
        pretrainedModel = f"cardiffnlp/twitter-roberta-base-sentiment"
        tokenizer = AutoTokenizer.from_pretrained(pretrainedModel)
        model = AutoModelForSequenceClassification.from_pretrained(pretrainedModel)

        def roBERTaPolarityScores(textInput):

            encodedText = tokenizer(textInput, return_tensors='pt') #"pt" stands for PyTorch which is the type of tensors returned
            modelOutput = model(**encodedText)
            outputScores = modelOutput[0][0].detach().numpy()

            sfScores = softmax(outputScores) #Softmaxed scores

            #print("\n", sfScores)

            scoresDict = {
                'roBERTaNeg': sfScores[0],
                'roBERTaNeu': sfScores[1],
                'roBERTaPos': sfScores[2]
            }

            return scoresDict

        scores = {}
        for i, row in tqdm(data.iterrows(), total=len(data)):
            try:
                text = row["Text"]
                revIndex = row["SentenceIndex"]

                scores[revIndex] = roBERTaPolarityScores(text)

            except RuntimeError:
                print("\nRuntimeError On Row: ", i, " Probably Text Too Long")
            except IndexError:
                print("\nIndexError On Row: ", i)

        roBERTaScores = pd.DataFrame(scores).T
        roBERTaScores = roBERTaScores.reset_index().rename(columns={"index": "SentenceIndex"})

        return roBERTaScores
    

    def descriptiveAnalysis(self, fullData: pd.DataFrame) -> None:

        print("\nFull Data DataFrame Columns: \n", fullData.columns.values, "\n")
        print("\nFull Data DataFrame Summary: \n", fullData.describe(), "\n\n")
        
        #MAJOR SENTIMENT PERCENTAGE ACROSS THE ith EPISODE

        totalRows, _ = fullData.shape

        roBERTaSentimentPercentages = {
            "Positive": len(fullData.query("roBERTaMajorSentiment == 'Positive'")) / totalRows,
            "Neutral": len(fullData.query("roBERTaMajorSentiment == 'Neutral'")) / totalRows,
            "Negative": len(fullData.query("roBERTaMajorSentiment == 'Negative'")) / totalRows
        }

        VADERSentimentPercentages = {
            "Positive": len(fullData.query("vaderMajorSentiment == 'Positive'")) / totalRows,
            "Neutral": len(fullData.query("vaderMajorSentiment == 'Neutral'")) / totalRows,
            "Negative": len(fullData.query("vaderMajorSentiment == 'Negative'")) / totalRows
        }

        print(f"Percentages of roBERTa Major Sentiment Relative to Total for Episode {self.episode}: ")
        pprint(roBERTaSentimentPercentages, indent=4, width=1)

        print()

        print(f"Percentages of VADER Major Sentiment Relative to Total for Episode {self.episode}: ")
        pprint(VADERSentimentPercentages, indent=4, width=1)

        print("\n")

        roBERTaModalMajorSentiment = statistics.mode(fullData["roBERTaMajorSentiment"])
        VADERModalMajorSentiment = statistics.mode(fullData["vaderMajorSentiment"])

        print("Modal roBERTa Major Sentiment: ", roBERTaModalMajorSentiment)
        print("Modal VADER Major Sentiment: ", VADERModalMajorSentiment)

        print()

        roBERTaTotalPositiveSentences = len(fullData.query("roBERTaMajorSentiment == 'Positive'"))
        roBERTaTotalNeutralSentences = len(fullData.query("roBERTaMajorSentiment == 'Neutral'"))
        roBERTaTotalNegativeSentences = len(fullData.query("roBERTaMajorSentiment == 'Negative'"))

        VADERTotalPositiveSentences = len(fullData.query("vaderMajorSentiment == 'Positive'"))
        VADERTotalNeutralSentences = len(fullData.query("vaderMajorSentiment == 'Neutral'"))
        VADERTotalNegativeSentences = len(fullData.query("vaderMajorSentiment == 'Negative'"))


        print("roBERTa Total Positive Sentences: ", roBERTaTotalPositiveSentences)
        print("roBERTa Total Neutral Sentences: ", roBERTaTotalNeutralSentences)
        print("roBERTa Total Negative Sentences: ", roBERTaTotalNegativeSentences)

        print("VADER Positive Sentences: ", VADERTotalPositiveSentences)
        print("VADER Neutral Sentences: ", VADERTotalNeutralSentences)
        print("VADER Negative Sentences: ", VADERTotalNegativeSentences)

        print("\n")

        sentencesCountByCharacter = {}

        for p in list(self.charactersName.values()):
            sentencesCountByCharacter[p] = len(fullData.query(f"Character == '{p}'"))

        print("Sentences Count by Character: ", sentencesCountByCharacter, "\n")

        # roBERTa Grouping ----------------------------------------------------------------------------------------------------

        roBERTaPositiveGrouping = fullData[fullData["roBERTaMajorSentiment"] == "Positive"].groupby("Character", as_index=False)["roBERTaPos"]
        roBERTaNeutralGrouping = fullData[fullData["roBERTaMajorSentiment"] == "Neutral"].groupby("Character", as_index=False)["roBERTaNeu"]
        roBERTaNegativeGrouping = fullData[fullData["roBERTaMajorSentiment"] == "Negative"].groupby("Character", as_index=False)["roBERTaNeg"]


        # ----------------------------------------------------------------------------------------------------

        roBERTaCntPosSentiment = roBERTaPositiveGrouping.count()
        roBERTaCntNeuSentiment = roBERTaNeutralGrouping.count()
        roBERTaCntNegSentiment = roBERTaNegativeGrouping.count()

        #print("roBERTaPositive Sentiment Sentences Count By Character: \n", roBERTaCntPosSentiment, "\n")  #Counting positive sentiment sentences (meaning when positive is the major sentiment)
        #print("roBERTaNeutral Sentiment Sentences Count By Character: \n", roBERTaCntNeuSentiment, "\n")  #Counting neutral sentiment sentences (meaning when neutral is the major sentiment)
        #print("roBERTaNegative Sentiment Sentences Count By Character: \n", roBERTaCntNegSentiment, "\n")  #Counting negative sentiment sentences (meaning when negative is the major sentiment)

        #print()

        # ----------------------------------------------------------------------------------------------------

        #Checking how positive, neutral or negative every sentiment is where it's the prevailing one
        roBERTaMedianPosSentiment = roBERTaPositiveGrouping.median()
        roBERTaMedianNeuSentiment = roBERTaNeutralGrouping.median()
        roBERTaMedianNegSentiment = roBERTaNegativeGrouping.median()

        #print("roBERTaMedian Positive Sentiment By Character: \n", roBERTaMedianPosSentiment, "\n")
        #print("roBERTaMedian Neutral Sentiment By Character: \n", roBERTaMedianNeuSentiment, "\n")
        #print("roBERTaMedian Negative Sentiment By Character: \n", roBERTaMedianNegSentiment, "\n")

        #print()

        #----------------------------------------------------------------------------------------------------

        roBERTaAvgPosSentiment = roBERTaPositiveGrouping.mean() #Calculating the average positive sentiment in the sentences where that's the predominant one
        roBERTaAvgNeuSentiment = roBERTaNeutralGrouping.mean() #Calculating the average neutral sentiment in the sentences where that's the predominant one
        roBERTaAvgNegSentiment = roBERTaNegativeGrouping.mean() #Calculating the average negative sentiment in the sentences where that's the predominant one

        #print("roBERTaAverage Positive Sentiment For Positive Sentences By Character: \n", roBERTaAvgPosSentiment, "\n")
        #print("roBERTaAverage Neutral Sentiment For Neutral Sentences By Character: \n", roBERTaAvgNeuSentiment, "\n")
        #print("roBERTaAverage Negative Sentiment For Negative Sentences By Character: \n", roBERTaAvgNegSentiment, "\n")


        #print()

        #----------------------------------------------------------------------------------------------------

        roBERTaStdPosSentiment = roBERTaPositiveGrouping.std()
        roBERTaStdNeuSentiment = roBERTaNeutralGrouping.std()
        roBERTaStdNegSentiment = roBERTaNegativeGrouping.std()

        #print("roBERTaPositive Sentiment Standard Deviation For Positive Sentences By Character: \n", roBERTaStdPosSentiment, "\n")
        #print("roBERTaNeutral Sentiment Standard Deviation For Neutral Sentences By Character: \n", roBERTaStdNeuSentiment, "\n")
        #print("roBERTaNegative Sentiment Standard Deviation For Negative Sentences By Character: \n", roBERTaStdNegSentiment, "\n")


        #print()


        # VADER Grouping ----------------------------------------------------------------------------------------------------


        vaderPositiveGrouping = fullData[fullData["vaderMajorSentiment"] == "Positive"].groupby("Character", as_index=False)["vaderPos"]
        vaderNeutralGrouping = fullData[fullData["vaderMajorSentiment"] == "Neutral"].groupby("Character", as_index=False)["vaderNeu"]
        vaderNegativeGrouping = fullData[fullData["vaderMajorSentiment"] == "Negative"].groupby("Character", as_index=False)["vaderNeg"]

        # ----------------------------------------------------------------------------------------------------

        vaderCntPosSentiment = vaderPositiveGrouping.count()
        vaderCntNeuSentiment = vaderNeutralGrouping.count()
        vaderCntNegSentiment = vaderNegativeGrouping.count()

        #print("VADER Positive Sentiment Sentences Count By Character: \n", vaderCntPosSentiment, "\n")  # Counting positive sentiment sentences (meaning when positive is the major sentiment)
        #print("VADER Neutral Sentiment Sentences Count By Character: \n", vaderCntNeuSentiment, "\n")  # Counting neutral sentiment sentences (meaning when neutral is the major sentiment)
        #print("VADER Negative Sentiment Sentences Count By Character: \n", vaderCntNegSentiment, "\n")  # Counting negative sentiment sentences (meaning when negative is the major sentiment)

        #print()

        # ----------------------------------------------------------------------------------------------------

        # Checking how positive, neutral or negative every sentiment is where it's the prevailing one
        vaderMedianPosSentiment = vaderPositiveGrouping.median()
        vaderMedianNeuSentiment = vaderNeutralGrouping.median()
        vaderMedianNegSentiment = vaderNegativeGrouping.median()

        #print("VADER Median Positive Sentiment By Character: \n", vaderMedianPosSentiment, "\n")
        #print("VADER Median Neutral Sentiment By Character: \n", vaderMedianNeuSentiment, "\n")
        #print("VADER Median Negative Sentiment By Character: \n", vaderMedianNegSentiment, "\n")

        #print()

        # ----------------------------------------------------------------------------------------------------

        vaderAvgPosSentiment = vaderPositiveGrouping.mean()  # Calculating the average positive sentiment in the sentences where that's the predominant one
        vaderAvgNeuSentiment = vaderNeutralGrouping.mean()  # Calculating the average neutral sentiment in the sentences where that's the predominant one
        vaderAvgNegSentiment = vaderNegativeGrouping.mean()  # Calculating the average negative sentiment in the sentences where that's the predominant one

        #print("VADER Average Positive Sentiment For Positive Sentences By Character: \n", vaderAvgPosSentiment, "\n")
        #print("VADER Average Neutral Sentiment For Neutral Sentences By Character: \n", vaderAvgNeuSentiment, "\n")
        #print("VADER Average Negative Sentiment For Negative Sentences By Character: \n", vaderAvgNegSentiment, "\n")

        #print()

        # ----------------------------------------------------------------------------------------------------

        vaderStdPosSentiment = vaderPositiveGrouping.std()
        vaderStdNeuSentiment = vaderNeutralGrouping.std()
        vaderStdNegSentiment = vaderNegativeGrouping.std()

        #print("VADER Positive Sentiment Standard Deviation For Positive Sentences By Character: \n", vaderStdPosSentiment, "\n")
        #print("VADER Neutral Sentiment Standard Deviation For Neutral Sentences By Character: \n", vaderStdNeuSentiment, "\n")
        #print("VADER Negative Sentiment Standard Deviation For Negative Sentences By Character: \n", vaderStdNegSentiment, "\n")

        print()


        sentimentData = {}

        #Adding a key value pair to the sentimentData dictionary where each key is a character name and the value is a dictionary containing various data about the sentiment of the character itself
        for i in list(self.charactersName.values()):
            sentimentData.update({i: {"vaderAveragePositive": vaderAvgPosSentiment.loc[vaderAvgPosSentiment["Character"] == i, "vaderPos"].values[0],
                                      "vaderAverageNeutral": vaderAvgNeuSentiment.loc[vaderAvgNeuSentiment["Character"] == i, "vaderNeu"].values[0],
                                      "vaderAverageNegative": vaderAvgNegSentiment.loc[vaderAvgNegSentiment["Character"] == i, "vaderNeg"].values[0],
                                      "vaderMedianPositive": vaderMedianPosSentiment.loc[vaderMedianPosSentiment["Character"] == i, "vaderPos"].values[0],
                                      "vaderMedianNeutral": vaderMedianNeuSentiment.loc[vaderMedianNeuSentiment["Character"] == i, "vaderNeu"].values[0],
                                      "vaderMedianNegative": vaderMedianNegSentiment.loc[vaderMedianNegSentiment["Character"] == i, "vaderNeg"].values[0],
                                      "vaderStdPositive": vaderStdPosSentiment.loc[vaderStdPosSentiment["Character"] == i, "vaderPos"].values[0],
                                      "vaderStdNeutral": vaderStdNeuSentiment.loc[vaderStdNeuSentiment["Character"] == i, "vaderNeu"].values[0],
                                      "vaderStdNegative": vaderStdNegSentiment.loc[vaderStdNegSentiment["Character"] == i, "vaderNeg"].values[0],
                                      "vaderPositiveSentencesCount": vaderCntPosSentiment.loc[vaderCntPosSentiment["Character"] == i, "vaderPos"].values[0],
                                      "vaderNeutralSentencesCount": vaderCntNeuSentiment.loc[vaderCntNeuSentiment["Character"] == i, "vaderNeu"].values[0],
                                      "vaderNegativeSentencesCount": vaderCntNegSentiment.loc[vaderCntNegSentiment["Character"] == i, "vaderNeg"].values[0],
                                      "roBERTaAveragePositive": roBERTaAvgPosSentiment.loc[roBERTaAvgPosSentiment["Character"] == i, "roBERTaPos"].values[0],
                                      "roBERTaAverageNeutral": roBERTaAvgNeuSentiment.loc[roBERTaAvgNeuSentiment["Character"] == i, "roBERTaNeu"].values[0],
                                      "roBERTaAverageNegative": roBERTaAvgNegSentiment.loc[roBERTaAvgNegSentiment["Character"] == i, "roBERTaNeg"].values[0],
                                      "roBERTaMedianPositive": roBERTaMedianPosSentiment.loc[roBERTaMedianPosSentiment["Character"] == i, "roBERTaPos"].values[0],
                                      "roBERTaMedianNeutral": roBERTaMedianNeuSentiment.loc[roBERTaMedianNeuSentiment["Character"] == i, "roBERTaNeu"].values[0],
                                      "roBERTaMedianNegative": roBERTaMedianNegSentiment.loc[roBERTaMedianNegSentiment["Character"] == i, "roBERTaNeg"].values[0],
                                      "roBERTaStdPositive": roBERTaStdPosSentiment.loc[roBERTaStdPosSentiment["Character"] == i, "roBERTaPos"].values[0],
                                      "roBERTaStdNeutral": roBERTaStdNeuSentiment.loc[roBERTaStdNeuSentiment["Character"] == i, "roBERTaNeu"].values[0],
                                      "roBERTaStdNegative": roBERTaStdNegSentiment.loc[roBERTaStdNegSentiment["Character"] == i, "roBERTaNeg"].values[0],
                                      "roBERTaPositiveSentencesCount": roBERTaCntPosSentiment.loc[roBERTaCntPosSentiment["Character"] == i, "roBERTaPos"].values[0],
                                      "roBERTaNeutralSentencesCount": roBERTaCntNeuSentiment.loc[roBERTaCntNeuSentiment["Character"] == i, "roBERTaNeu"].values[0],
                                      "roBERTaNegativeSentencesCount": roBERTaCntNegSentiment.loc[roBERTaCntNegSentiment["Character"] == i, "roBERTaNeg"].values[0],
                                      "TotalSentencesCount": sentencesCountByCharacter[i]}})


        sentimentDataDF = pd.DataFrame(sentimentData).T
        print(sentimentDataDF, "\n")
        print("Sentiment Scores DataFrame Shape: ", sentimentDataDF.shape, "\n")


        #Converting every float number into string to store everything into a json file
        for c, dct in sentimentData.items():
            for k, v in dct.items():
                dct[k] = str(v)


        #Some useful information that goes at the end of the json file
        jsonAppendice = {
                        "Episode": str(self.episode),
                        "Details": self.episodesInfo[self.episode],
                        "UsefulStatistics": {
                                "roBERTaModalMajorSentiment": roBERTaModalMajorSentiment,
                                "VADERModalMajorSentiment": VADERModalMajorSentiment,
                                "roBERTaTotalPositiveSentences": roBERTaTotalPositiveSentences,
                                "roBERTaTotalNeutralSentences": roBERTaTotalNeutralSentences,
                                "roBERTaTotalNegativeSentences": roBERTaTotalNegativeSentences,
                                "VADERTotalPositiveSentences": VADERTotalPositiveSentences,
                                "VADERTotalNeutralSentences": VADERTotalNeutralSentences,
                                "VADERTotalNegativeSentences": VADERTotalNegativeSentences
                                }
                        }

        sentimentData.update(jsonAppendice)


        #Exporting the whole data into a json file
        with open(f"./OutputData/HPE{self.episode}OutputData/{self.episode}-Data/EP{self.episode}SentimentStatistics.json", "w") as jsonFile:
            json.dump(sentimentData, jsonFile, indent=4)


        def sentimentPlots():

            @savePlots
            def majorSentimentPercentagesPlot():

                plt.figure(figsize=(16, 9))
                plt.pie(list(roBERTaSentimentPercentages.values()), labels=list(roBERTaSentimentPercentages.keys()), colors=viridisColorScale, autopct='%1.1f%%', textprops={'color': "w"})
                plt.title(f"Sentiments Percentages Plot For Episode {self.episode}", fontdict=dict(weight="bold"))
                plt.legend(loc="upper right", prop=dict(size="medium"))

                return f"EP{self.episode}majorSentimentPercentagesPlot", plt, self.plotsPath
            
            @savePlots
            def averageSentimentByCharacter():

                medianPosSentimentCharacterNames = roBERTaMedianPosSentiment["Character"]
                medianNeuSentimentCharacterNames = roBERTaMedianNeuSentiment["Character"]
                medianNegSentimentCharacterNames = roBERTaMedianNegSentiment["Character"]

                figPos = px.bar(x=medianPosSentimentCharacterNames, y=roBERTaMedianPosSentiment["roBERTaPos"], color=medianPosSentimentCharacterNames)
                figPos.update_layout(title=f"EP{self.episode} - Median Positive Sentiment By Character", xaxis_title="Character", yaxis_title="Median Positive Sentiment", legend_title="Characters", font=dict(size=20))

                figNeu = px.bar(x=medianNeuSentimentCharacterNames, y=roBERTaMedianNeuSentiment["roBERTaNeu"], color=medianNeuSentimentCharacterNames)
                figNeu.update_layout(title=f"EP{self.episode} - Median Neutral Sentiment By Character", xaxis_title="Character", yaxis_title="Median Neutral Sentiment", legend_title="Characters", font=dict(size=20))

                figNeg = px.bar(x=medianNegSentimentCharacterNames, y=roBERTaMedianNegSentiment["roBERTaNeg"], color=medianNegSentimentCharacterNames)
                figNeg.update_layout(title=f"EP{self.episode} - Median Negative Sentiment By Character", xaxis_title="Character", yaxis_title="Median Negative Sentiment", legend_title="Characters", font=dict(size=20))
                
                return [f"EP{self.episode}medianPositiveSentimentByCharacter", f"EP{self.episode}medianNeutralSentimentByCharacter", f"EP{self.episode}medianNegativeSentimentByCharacter"], [figPos, figNeu, figNeg], self.plotsPath
            
            @savePlots
            def sentimentStdPlot():

                charactersLabels = sentimentDataDF.index.values.tolist()

                xLabelsLoc = np.arange(len(sentimentDataDF)) #Specifying label locations
                barsWidth = 0.25 #Bars' width
                multiplier = 0 #Used to specify the offset between the bars

                fig, ax = plt.subplots(figsize=(16, 9))

                sentimentsSeries = [sentimentDataDF["roBERTaStdPositive"], sentimentDataDF["roBERTaStdNeutral"], sentimentDataDF["roBERTaStdNegative"]]
                sentimentsNames = ["roBERTaStdPositive", "roBERTaStdNeutral", "roBERTaStdNegative"]

                for name, measurement in zip(sentimentsNames, sentimentsSeries):
                    offset = barsWidth * multiplier
                    bars = ax.bar(xLabelsLoc + offset, round(measurement, ndigits=3), barsWidth, label=name)
                    ax.bar_label(bars, padding=3)
                    multiplier += 1

                # Add some text for labels, title and custom x-axis tick labels, etc.
                ax.set_ylabel('roBERTa Standard Deviation', fontdict=dict(weight="bold", size=15))
                ax.set_xlabel("Characters", fontdict=dict(weight="bold", size=15))
                ax.set_title(f"EP{self.episode} - roBERTa Sentiments' Standard Deviation by Characters", fontdict=dict(weight="bold", size=25))
                ax.set_xticks(xLabelsLoc + barsWidth, charactersLabels, rotation=0, fontdict=dict(weight="bold", size=10))
                ax.legend(loc='upper left', ncols=3)
                fig.tight_layout()

                return f"EP{self.episode}roBERTaSentimentsStdByCharacters", fig, self.plotsPath

            @savePlots
            def sentimentMedianPlot():

                charactersLabels = sentimentDataDF.index.values.tolist()

                xLabelsLoc = np.arange(len(sentimentDataDF)) #Specifying label locations
                barsWidth = 0.25 #Bars' width
                multiplier = 0 #Used to specify the offset between the bars

                fig, ax = plt.subplots(figsize=(16, 9))

                sentimentsSeries = [sentimentDataDF["roBERTaMedianPositive"], sentimentDataDF["roBERTaMedianNeutral"], sentimentDataDF["roBERTaMedianNegative"]]
                sentimentsNames = ["roBERTaMedianPositive", "roBERTaMedianNeutral", "roBERTaMedianNegative"]

                for name, measurement in zip(sentimentsNames, sentimentsSeries):
                    offset = barsWidth * multiplier
                    bars = ax.bar(xLabelsLoc + offset, round(measurement, ndigits=3), barsWidth, label=name)
                    ax.bar_label(bars, padding=3)
                    multiplier += 1

                # Add some text for labels, title and custom x-axis tick labels, etc.
                ax.set_ylabel('roBERTa Median Sentiment Level', fontdict=dict(weight="bold", size=15))
                ax.set_xlabel("Characters", fontdict=dict(weight="bold", size=15))
                ax.set_title(f"EP{self.episode} - roBERTa Sentiments' Median Level by Characters", fontdict=dict(weight="bold", size=25))
                ax.set_xticks(xLabelsLoc + barsWidth, charactersLabels, rotation=0, fontdict=dict(weight="bold", size=10))
                ax.legend(loc='upper left', ncols=3)
                fig.tight_layout()

                return f"EP{self.episode}roBERTaSentimentsMedianByCharacters", fig, self.plotsPath


            plotsList = [majorSentimentPercentagesPlot, averageSentimentByCharacter, sentimentStdPlot, sentimentMedianPlot]
            all((i(), plt.clf()) for i in plotsList)

            return None

        sentimentPlots()
        
        return None
    

    @staticmethod
    def exportFullData(episode, fullData: pd.DataFrame) -> None:

        fullData.to_csv(f"./OutputData/HPE{episode}OutputData/{episode}-Data/HPE{episode}FullData.csv", encoding="utf-8", index=False)
        fullData.to_json(f"./OutputData/HPE{episode}OutputData/{episode}-Data/HPE{episode}FullData.json", index=False)

        print(f"\nExported Full DataFrame Shape: {fullData.shape}")
        print("Full DataFrame Columns: \n", fullData.columns.values)

        return None


    def executeAnalysis(self):

        roBERTaSentimentNamesDict = {"roBERTaPos": "Positive",
                                     "roBERTaNeu": "Neutral",
                                     "roBERTaNeg": "Negative"}

        vaderSentimentNamesDict = {"vaderPos": "Positive",
                                   "vaderNeu": "Neutral",
                                   "vaderNeg": "Negative"}

        vaderData = self.vaderAnalyze(self.data)
        roBERTaData = self.robertaAnalyze(self.data)


        fullData = pd.concat([self.data, vaderData, roBERTaData], axis=1, join="inner")
        fullData = fullData.loc[:, ~fullData.columns.duplicated()] #Removingss duplicated columns


        fullData["roBERTaMajorSentiment"] = fullData[["roBERTaPos", "roBERTaNeu", "roBERTaNeg"]].idxmax(axis=1) #Identifying the prevailing sentiment of the sentences through roBERTa scores
        fullData["roBERTaMajorSentiment"] = fullData["roBERTaMajorSentiment"].apply(lambda x: roBERTaSentimentNamesDict[x])

        fullData["vaderMajorSentiment"] = fullData[["vaderPos", "vaderNeu", "vaderNeg"]].idxmax(axis=1) #Identifying the prevailing sentiment of the sentences through VADER scores
        fullData["vaderMajorSentiment"] = fullData["vaderMajorSentiment"].apply(lambda x: vaderSentimentNamesDict[x])


        #print(fullData)

        self.descriptiveAnalysis(fullData) #Executing a descriptive analysis

        self.exportFullData(self.episode, fullData)

        return None




















































