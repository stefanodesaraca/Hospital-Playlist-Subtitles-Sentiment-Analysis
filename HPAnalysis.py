import os
import re
import json
import pandas as pd
import numpy as np
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
            "2": "Kim Jun-Wan",
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
        print("\nNaN Sum: ", data["Character"].isna().sum(), "\n")

        data["Character"] = data["Character"].fillna(0) #All the sentences that aren't said by the 5 main characters are classified as from "secondary" characters

        data["Character"] = data["Character"].astype("int16", errors="raise")
        data["Character"] = data["Character"].astype(str, errors="raise")

        print(data["Character"].unique())

        print("\nColumns Data Types: \n", data.dtypes, "\n\n")

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
            "2": "Kim Jun-Wan",
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

        print("\nFull Data DataFrame Columns: \n", fullData.columns, "\n")
        print("\nFull Data DataFrame Summary: \n", fullData.describe(), "\n\n")
        
        #MAJOR SENTIMENT PERCENTAGE ACROSS THE ith EPISODE

        totalRows, _ = fullData.shape

        sentimentPercentages = {
            "Positive": len(fullData.query("MajorSentiment == 'Positive'")) / totalRows,
            "Neutral": len(fullData.query("MajorSentiment == 'Neutral'")) / totalRows,
            "Negative": len(fullData.query("MajorSentiment == 'Negative'")) / totalRows
        }

        print(f"Percentages of Major Sentiment Relative to Total for Episode {self.episode}: ")
        pprint(sentimentPercentages, indent=4, width=1)

        print("\n")


        sentencesCountByCharacter = {}

        for p in list(self.charactersName.values()):
            sentencesCountByCharacter[p] = len(fullData.query(f"Character == '{p}'"))

        print("Sentences Count by Character: \n", sentencesCountByCharacter, "\n")


        #Checking how positive, neutral or negative every sentiment is where it's the prevailing one
        medianPositiveSentimentByCharacter = fullData[fullData["MajorSentiment"]=="Positive"].groupby("Character", as_index=False)["roBERTaPos"].median()
        medianNeutralSentimentByCharacter = fullData[fullData["MajorSentiment"] == "Neutral"].groupby("Character", as_index=False)["roBERTaNeu"].median()
        medianNegativeSentimentByCharacter = fullData[fullData["MajorSentiment"] == "Negative"].groupby("Character", as_index=False)["roBERTaNeg"].median()

        print("Median Positive Sentiment By Character: \n", medianPositiveSentimentByCharacter, "\n")
        print("Median Neutral Sentiment By Character: \n", medianNeutralSentimentByCharacter, "\n")
        print("Median Negative Sentiment By Character: \n", medianNegativeSentimentByCharacter, "\n")

        print()

        #----------------------------------------------------------------------------------------------------

        
        cntPosSentiment = {}
        cntNeuSentiment = {}
        cntNegSentiment = {}


        for x in list(self.charactersName.values()):
            cntPos = len(fullData.query(f"Character == '{x}' & MajorSentiment == 'Positive'")) #Counting positive sentiment sentences (meaning when positive is the major sentiment)
            cntPosSentiment[x] = cntPos

        print(f"Positive Sentiment Sentences Count for Every Character in Episode: {self.episode}")
        print(cntPosSentiment, "\n")

        for y in list(self.charactersName.values()):
            cntNeu = len(fullData.query(f"Character == '{y}' & MajorSentiment == 'Neutral'")) #Counting neutral sentiment sentences (meaning when positive is the major sentiment)
            cntNeuSentiment[y] = cntNeu

        print(f"Neutral Sentiment Sentences Count for Every Character in Episode: {self.episode}")
        print(cntNeuSentiment, "\n")

        for z in list(self.charactersName.values()):
            cntNeg = len(fullData.query(f"Character == '{z}' & MajorSentiment == 'Negative'")) #Counting negative sentiment sentences (meaning when positive is the major sentiment)
            cntNegSentiment[z] = cntNeg

        print(f"Negative Sentiment Sentences Count for Every Character in Episode: {self.episode}")
        print(cntNegSentiment, "\n")


        print("\n")


        avgPosSentiment = {}
        avgNeuSentiment = {}
        avgNegSentiment = {}


        for xy in list(self.charactersName.values()):
            characterSentences = fullData.query(f"Character == '{xy}' & MajorSentiment == 'Positive'")
            avgPos = np.mean(characterSentences["roBERTaPos"]) #Calculating the average positive sentiment in the sentences where that's the predominant one

            avgPosSentiment[xy] = avgPos

        print(f"Average Positive Sentiment for Every Character in Episode: {self.episode}")
        print(avgPosSentiment, "\n")

        for yz in list(self.charactersName.values()):
            characterSentences = fullData.query(f"Character == '{yz}' & MajorSentiment == 'Neutral'")
            avgNeu = np.mean(characterSentences["roBERTaNeu"]) #Calculating the average neutral sentiment in the sentences where that's the predominant one

            avgNeuSentiment[yz] = avgNeu

        print(f"Average Neutral Sentiment for Every Character in Episode: {self.episode}")
        print(avgNeuSentiment, "\n")

        for zw in list(self.charactersName.values()):
            characterSentences = fullData.query(f"Character == '{zw}' & MajorSentiment == 'Negative'")
            avgNeg = np.mean(characterSentences["roBERTaNeg"]) #Calculating the average negative sentiment in the sentences where that's the predominant one

            avgNegSentiment[zw] = avgNeg

        print(f"Average Negative Sentiment for Every Character in Episode: {self.episode}")
        print(avgNegSentiment, "\n")


        print("\n")


        stdPosSentiment = {}
        stdNeuSentiment = {}
        stdNegSentiment = {}


        for xyz in list(self.charactersName.values()):
            characterSentences = fullData.query(f"Character == '{xyz}' & MajorSentiment == 'Positive'")
            stdPos = np.std(characterSentences["roBERTaPos"]) #Calculating the standard deviation of positive sentiment in the sentences where that's the predominant one

            stdPosSentiment[xyz] = stdPos

        print(f"Standard Deviation of Positive Sentiment for Every Character in Episode: {self.episode}")
        print(stdPosSentiment, "\n")

        for yzw in list(self.charactersName.values()):
            characterSentences = fullData.query(f"Character == '{yzw}' & MajorSentiment == 'Neutral'")
            stdNeu = np.std(characterSentences["roBERTaNeu"]) #Calculating the standard deviation of neutral sentiment in the sentences where that's the predominant one

            stdNeuSentiment[yzw] = stdNeu

        print(f"Standard Deviation of Neutral Sentiment for Every Character in Episode: {self.episode}")
        print(stdNeuSentiment, "\n")

        for zwx in list(self.charactersName.values()):
            characterSentences = fullData.query(f"Character == '{zwx}' & MajorSentiment == 'Negative'")
            stdNeg = np.std(characterSentences["roBERTaNeg"]) #Calculating the standard deviation of negative sentiment in the sentences where that's the predominant one

            stdNegSentiment[zwx] = stdNeg

        print(f"Standard Deviation of Negative Sentiment for Every Character in Episode: {self.episode}")
        print(stdNegSentiment, "\n")


        print("\n")


        sentimentData = {}

        #Adding a key value pair to the sentimentData dictionary where each key is a character name and the value is a dictionary containing various data about the sentiment of the character itself
        for i in list(self.charactersName.values()):
            sentimentData.update({i: {"AveragePositive": avgPosSentiment[i],
                                      "AverageNeutral": avgNeuSentiment[i],
                                      "AverageNegative": avgNegSentiment[i],
                                      "StdPositive": stdPosSentiment[i],
                                      "StdNeutral": stdNeuSentiment[i],
                                      "StdNegative": stdNegSentiment[i],
                                      "PositiveSentencesCount": cntPosSentiment[i],
                                      "NeutralSentencesCount": cntNeuSentiment[i],
                                      "NegativeSentencesCount": cntNegSentiment[i],
                                      "TotalSentencesCount": sentencesCountByCharacter[i]}})


        sentimentDataDF = pd.DataFrame(sentimentData).T
        print(sentimentDataDF, "\n")


        #Converting every float number into string to store everything into a json file
        for c, dct in sentimentData.items():
            for k, v in dct.items():
                dct[k] = str(v)


        #Some useful information that goes at the end of the json file
        jsonAppendice = {
                        "Episode": str(self.episode),
                        "Details": self.episodesInfo[self.episode]
                        }

        sentimentData.update(jsonAppendice)


        #Exporting the whole data into a json file
        with open(f"./OutputData/HPE{self.episode}OutputData/{self.episode}-Data/EP{self.episode}SentimentStatistics.json", "w") as jsonFile:
            json.dump(sentimentData, jsonFile, indent=4)


        def sentimentPlots():

            @savePlots
            def majorSentimentPercentagesPlot():

                plt.figure(figsize=(16, 9))
                plt.pie(list(sentimentPercentages.values()), labels=list(sentimentPercentages.keys()), colors=viridisColorScale)
                plt.title(f"Sentiments Percentages Plot For Episode {self.episode}", fontdict=dict(weight="bold"))
                plt.legend(loc="upper right", prop=dict(size="medium"))

                return "majorSentimentPercentagesPlot", plt, self.plotsPath
            
            @savePlots
            def sentimentByCharacter():

                figPos = px.bar(x=list(avgPosSentiment.keys()), y=list(avgPosSentiment.values()), color=list(avgPosSentiment.keys()))
                figPos.update_layout(title="Average Positive Sentiment By Character", xaxis_title="Character", yaxis_title="Average Positive Sentiment", legend_title="Characters", font=dict(size=20))

                figNeu = px.bar(x=list(avgNeuSentiment.keys()), y=list(avgNeuSentiment.values()), color=list(avgNeuSentiment.keys()))
                figNeu.update_layout(title="Average Neutral Sentiment By Character", xaxis_title="Character", yaxis_title="Average Neutral Sentiment", legend_title="Characters", font=dict(size=20))

                figNeg = px.bar(x=list(avgNegSentiment.keys()), y=list(avgNegSentiment.values()), color=list(avgNegSentiment.keys()))
                figNeg.update_layout(title="Average Negative Sentiment By Character", xaxis_title="Character", yaxis_title="Average Negative Sentiment", legend_title="Characters", font=dict(size=20))
                
                return ["averagePositiveSentimentByCharacter", "averageNeutralSentimentByCharacter", "averageNegativeSentimentByCharacter"], [figPos, figNeu, figNeg], self.plotsPath
            
            @savePlots
            def sentimentStdPlot():

                charactersLabels = sentimentDataDF.index.values.tolist()

                xLabelsLoc = np.arange(len(sentimentDataDF)) #Specifying label locations
                barsWidth = 0.25 #Bars' width
                multiplier = 0 #Used to specify the offset between the bars

                fig, ax = plt.subplots(figsize=(16, 9))

                sentimentsSeries = [sentimentDataDF["StdPositive"], sentimentDataDF["StdNeutral"], sentimentDataDF["StdNegative"]]
                sentimentsNames = ["StdPositive", "StdNeutral", "StdNegative"]

                for name, measurement in zip(sentimentsNames, sentimentsSeries):
                    offset = barsWidth * multiplier
                    bars = ax.bar(xLabelsLoc + offset, round(measurement, ndigits=3), barsWidth, label=name)
                    ax.bar_label(bars, padding=3)
                    multiplier += 1

                # Add some text for labels, title and custom x-axis tick labels, etc.
                ax.set_ylabel('Standard Deviation', fontdict=dict(weight="bold", size=15))
                ax.set_xlabel("Characters", fontdict=dict(weight="bold", size=15))
                ax.set_title("Sentiments' Standard Deviation by Characters", fontdict=dict(weight="bold", size=25))
                ax.set_xticks(xLabelsLoc + barsWidth, charactersLabels, rotation=0, fontdict=dict(weight="bold", size=10))
                ax.legend(loc='upper left', ncols=3)
                fig.tight_layout()

                return "SentimentsStdByCharacters", fig, self.plotsPath

            @savePlots
            def sentimentAvgPlot():

                charactersLabels = sentimentDataDF.index.values.tolist()

                xLabelsLoc = np.arange(len(sentimentDataDF)) #Specifying label locations
                barsWidth = 0.25 #Bars' width
                multiplier = 0 #Used to specify the offset between the bars

                fig, ax = plt.subplots(figsize=(16, 9))

                sentimentsSeries = [sentimentDataDF["AveragePositive"], sentimentDataDF["AverageNeutral"], sentimentDataDF["AverageNegative"]]
                sentimentsNames = ["AveragePositive", "AverageNeutral", "AverageNegative"]

                for name, measurement in zip(sentimentsNames, sentimentsSeries):
                    offset = barsWidth * multiplier
                    bars = ax.bar(xLabelsLoc + offset, round(measurement, ndigits=3), barsWidth, label=name)
                    ax.bar_label(bars, padding=3)
                    multiplier += 1

                # Add some text for labels, title and custom x-axis tick labels, etc.
                ax.set_ylabel('Average Sentiment Level', fontdict=dict(weight="bold", size=15))
                ax.set_xlabel("Characters", fontdict=dict(weight="bold", size=15))
                ax.set_title("Sentiments' Average Level by Characters", fontdict=dict(weight="bold", size=25))
                ax.set_xticks(xLabelsLoc + barsWidth, charactersLabels, rotation=0, fontdict=dict(weight="bold", size=10))
                ax.legend(loc='upper left', ncols=3)
                fig.tight_layout()

                return "SentimentsAvgByCharacters", fig, self.plotsPath


            plotsList = [majorSentimentPercentagesPlot, sentimentByCharacter, sentimentStdPlot, sentimentAvgPlot]
            all((i(), plt.clf()) for i in plotsList)

            return None

        sentimentPlots()
        
        return None
    

    @staticmethod
    def exportFullData(episode, fullData: pd.DataFrame) -> None:

        fullData.to_csv(f"./OutputData/HPE{episode}OutputData/{episode}-Data/HPE{episode}FullData.csv", encoding="utf-8", index=False)
        fullData.to_json(f"./OutputData/HPE{episode}OutputData/{episode}-Data/HPE{episode}FullData.json", index=False)

        return None


    def executeAnalysis(self):

        sentimentNamesDict = {"roBERTaPos": "Positive",
                              "roBERTaNeu": "Neutral",
                              "roBERTaNeg": "Negative"}

        vaderData = self.vaderAnalyze(self.data)
        roBERTaData = self.robertaAnalyze(self.data)


        fullData = pd.concat([self.data, vaderData, roBERTaData], axis=1, join="inner")
        fullData = fullData.loc[:, ~fullData.columns.duplicated()] #Removingss duplicated columns


        fullData["MajorSentiment"] = fullData[["roBERTaPos", "roBERTaNeu", "roBERTaNeg"]].idxmax(axis=1) #Finding the major sentiment for each row. The sentiment scores to find it are the ones gotten from roBERTa model considering it more precise (on average) than VADER
        fullData["MajorSentiment"] = fullData["MajorSentiment"].apply(lambda x: sentimentNamesDict[x]) #Correcting the major sentiment of the sentence name


        #print(fullData)

        self.descriptiveAnalysis(fullData) #Executing a descriptive analysis

        self.exportFullData(self.episode, fullData)

        return None




















































