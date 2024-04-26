from HPAnalysis import dataManager, sentimentAnalyzer



if __name__ == "__main__":

    episodes = ["1", "2", "3"]

    for e in episodes:

        episodeFileName = "./HPDatasets/HPE" + e + ".csv"

        manager = dataManager(e, episodeFileName)
        manager.createDirs()
        episodeData = manager.readData()

        print(repr(manager))

        analyzer = sentimentAnalyzer(e, episodeData)
        analyzer.executeAnalysis()















