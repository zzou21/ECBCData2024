# Early Consumption Before Capitalism (ECBC) 2024 Data+ 

This repository stores the final version of the code used by Duke University's 2024 Data+ "Ethical Consumption before Capitalism" research team. The team's goal is to innovatively combine historical and computational methodologies to study the linguistic choices made by writers within or related to the Virginia Company of London. We especially focus on the historical context of English discussions around the religious conversion of the Native American communities. This project is built on top of the important work done by past years' Data+ and Bass Connections teams.

2024 project website: [https://sites.duke.edu/ecbc2024/](https://sites.duke.edu/ecbc2024/)

# Folder Documentation:

## AuxiliaryPrograms:

This folder contains useful preparatory and experimental codes. These programs are the selected few that are helpful to preserve for reference.
* ### PtMac_Bible
  * #### data
    * _Bible_full_text.txt_: contains the entirety of the Geneva Bible in TXT format.
    * _genevaBible.csv_: contains the entirety of the Geneva Bible labeled with Book and Verse numbers.
    * _CSscore.ipynb_: uses the fine-tuned / default MacBERTh to compute the cosine similarity between two words.
    * _TextifyBible.ipynb_: turns the Bible in CSV format into a TXT file.
    * _pretrainMac.ipynb_: pretrains MacBERTh on Geneva Bible.
  * #### TFIDFKMeans: this folder contains practice code that performs TF-IDF to create word clusters. This folder was used in the early stage of this project to experiment with TF-IDF and K-Means methods.
    * _KMeansResults.json_: output of TFIDFmethod1WithClustering.ipynb
    * _TFIDFmethod1WithClustering.ipynb_: using TfidfVectorizer to create the top TF-IDF results for each file in a folder.
  * #### TextProcessing:
    * _PracticeCosineSimilarity.ipynb_: this program performs cosine similarity calculation based on TF-IDF. This program intended for testing and experimentation purposes is an early-stage program using cosine similarity. It assumes the TF-IDF results as the keywords to perform cosine similarity calculation.
    * _AllTestFunctions.py_: this file functions as a notepad where we store the smaller pieces of auxiliary programs. For example, the program in it selects the EEBO texts published between 1606 and 1625.

## BaseWordSelection:
* _accessEmbeddingStorageCalculation.py_: This program accesses a JSON that stores the embedding coordinates of individual words in a document (intended to be the concatenation of all documents in a corpus). Such a JSON makes the cosine similarity and other calculation processes faster, as the program doesn’t have to embed all words every time the program is executed.
* _BaseGen.py_: generates base words based on a set of categories given; did not end up using but upon proper change, can be very useful.
* _TopBasewordForCate.py_: generates base words based on a particular category. This has been useful.
* _standardizedwords.json_: The 2023 Data+ Ethical Consumption Before Capitalism team built this list. This is a list of unstandardized word spellings paired with their standardized spelling. This file is used during text cleaning.
* _StoreEmbedding.py_: performs embedding for the words in documents and stores the embedding of all tokens except special tokens

## Bias Identification: This is the most important folder of the Bias Axis Projection workflow.
* _Bias_Identification_Base.py_: digitize the bias axes, and project a particular keyword on a set of bias axes predetermined via close reading; compute this for all files in a particular folder.
* _BiasList_Interface.ipynb_: an interface that updates a set of categorized words.
* _Project_All_Words.py_: attempts to project all words of a document onto a particular bias axis and use the median of all projections to represent the entire text; did not end up using but this could be a meaningful breakthrough to study.

## CleanAndPrepareProjectionResults:
* _projectionTXTtoCSV.py_: this tool transforms the Bias Axes Projection outputs (in the form of a TXT file) into a CSV file for easy processing. The TXT file is the output of Bias_Identification_Base.py, whose output format can be found in the TXT files in the data folder in the root directory.

## FindCosineSimilarityWords:
* IFIDFWordsCosSim: this sub-folder contains the cosine similarity word search program using TF-IDF outputs from the 2023 Data+ team. The main program is stored in
  * _TFIDFCosSimSearch.py_, and TFIDFWordBank.json contains the TF-IDF words from 2023.
* decodeCosSimOutput: This sub-folder contains the customizable process to analyze the cosine similarity outputs.
  * _RollingUpdatesOutputs.json_: this contains some cosine similarity outputs from individual documents.
  * _decodeCosSimResults.py_: this is an important functionality-based Python class object that could: 1) perform standard decoding to turn cosine similarity outputs into more readable formats; 2) perform network analysis-intended decoding that finds the frequency and location (or the cosine similarity category and text) of appearance for each word in the corpus; 3) provide customizable decode methods that users could build themselves to better fit their programs. This file also contains a workflow control mechanism to prevent using up too much device memory and a text metadata search mechanism.
  * _outputCosSimReadable.json_: this JSON is the output of the standard decoding function in the Python class object in the decodeCosSimResults.py file.
* _600Files.json_: this contains the file serial number of all EEBO-TCP texts published between 1525-1639.
* _80VAFiles.json_: this contains the Virginia or colonization and conversion-related EEBO-TCP files identified by the 2023 Data+ team.
* _AccessMetadata.py_: this program contains a Python class object that finds the metadata of a text using its serial number in the format of “A00001.”
* _ClusterUsingCosineSimilarity.py_: this is the main cosine similarity word search algorithm. It takes in a directory of files and searches for the top N most similar words in each file according to the “keywordJSONPath,” a JSON that stores the category terms and their respective context sentences. Then, the result is outputted into a JSON that should then be decoded and analyzed using the decodeCosSimResults.py program under the decodeCosSimOutput sub-folder.
* _OneLevelKeywordSentence.json_: this JSON stores the category terms used to find cosine similarity words from as well as each term’s respective context sentences. This JSON is accessed by the ClusterUsingCosineSimilarity.py file.
* _combined2023csvAndTFIDFFileNames.json_: this JSON stores all EEBO-TCP file serial numbers identified by the 2023 Data+ team that are related to the Colony of Virginia.
* _filtered600Files.json_: this JSON stores the EEBO text serial numbers that were narrowed from our previous 600-file JSON.

## data:
* _categorized_words_GX.json_ (where X=1, 2, 3): 3 groups of categories to be used to construct bias axes. Each group contains 6 categories and base word and context sentences associated with the base words.
* _projection_result_GX_AB.txt_ [where X=1, 2, 3; AB = NT (native) or VA (Virginia)]: this is the output of Bias\ Identification/Bias_Identification_Base.py
* _projectionResultTable_GX_AB.csv_ [where X=1, 2, 3; AB = NT (native) or VA (Virginia)]: this is the output of CleanAndPrepareProjectionResults/projectionTXTtoCSV.py
* _Line_graph.*_: the R documents used to plot the time series analysis line graph
* _Visualization.*_: the R documents used to perform k-means clustering and distribution plot

## PunctuationRestoration:
* _puncRestore.py_: takes in the EP (Early Print) documents stored on the DCC folder called dedication_text_EPcorpus_1590-1639 and adds punctuation to them

## XMLProcessingAndTraining:
This folder contains the code that traverses EEBO-TCP to handle document metadata and build the relevant Python architecture and fine-tune MacBERTh for the EEBO-Geneva Bible connection.
### AuxiliaryXMLProcessing
* _ReadXMLwithoutTags.py_: this program reads the XML files that EEBO-TCP raw texts were originally stored in.
### ManuscriptMetadata
* _DocumentMetadata.json_: all metadata of EEBO-TCP texts (author, title, publication year, publisher, publication city) stored as uncleaned information in a JSON.
* _FindSpecificMetadata.py_: this program finds the metadata of EEBO-TCP texts as specified by the user. The result (or the metadata of the selected texts) is stored in a JSON file.
* _FindingDocumentMetadata.py_: this program traverses through the file names of EEBO-TCP files (where file names are all labeled using serial numbers) and finds the metadata of each text. This is designed to match serial numbers with metadata.
* _OperateMetadata.py_: this is an extensive Python class object that cleans the DocumentMetadata.json’s uncleaned metadata, resulting in cleanedDocumentMetadata.json. This program makes the document metadata easier to traverse and implement in users’ own programs.
* _SpecificMetadata.json_: storage JSON from FindSpecificMetadata.py.
* _cleanedDocumentMetadata.json_: this JSON is the cleaned version of DocumentMetadata.json
* _turnFileNamesIntoList.py_: this auxiliary program turns a TXT that contains a list of file names into machine-readable JSON.
### TurnXMLtoTXT:
* _BibleVerseDetectionTrainingImplementationPythonClassObject.py_: this program calls on our fine-tuned MacBERTh model trained on the Geneva Bible to determine whether an inputted sentence was a Bible verse or not.
filteredXMLBibleRef.json: this JSON stores the content of StoringItalicsAndLineNumber.json that has been filtered by out fine-tuned MacBERTh to be a Geneva Bible verse.
IDandFilterVersesMain.py: This master program combines the many different functions of our Python architecture. It traverses through the XML files, selects the content from specific XML labels, stores them, performs filtering of the stored information according to the specific needs of each user, and other functionalities that the user needs.
StoringItalicsAndLineNumber.json: this JSON stores all words in EEBO that has been labeled as italics by the XML tags. This JSON could store the content of whichever XML label that a user wants to extract.
TrainMacBERThRecognitionBible.py: This program trains the default MacBERTh on the Geneva Bible.
XMLCitationTags.json: This JSON contains the specific XML tags that the user wants to extract content from
XMLJsonAccessTool.py: this program is designed to traverse the dictionary structure of StoringItalicsAndLineNumber.json for quick access to information.
XMLParseVizTests.ipynb: this Jupyter Notebook file was used to test methods of how to approach the data gathered in filteredXMLBibleRef.json. In other words, this Notebook contains our attempts to analyze Biblical reference data.
XMLtagCollectorInterface.py: this is a user interface that allows the user to update what types of XML tags they wish to analyze. User inputs are stored in XMLCitationTags.json.

Shiny
data: the file names are simplified for easy access. [1.csv = projecting Virginia on G1] [2.csv = projecting Virginia on G2] [3.csv = projecting Native on G2] [4.csv = projecting Native on G3]
modules: contains modules to be called by the main app.R for creating visualizations
app.R: a R Shiny App that creates interactive plots for distribution (clustering) and line graphs (time-series)

