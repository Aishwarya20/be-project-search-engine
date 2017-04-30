Search Engine For an E-book Portal

Following are the modules implemented
1. Spell Checker: This helps correct spelling using the edit distance method
2.Text Segmentation: Segments the text using the bigram model
3.Language Modelling: Corrects the sequence of words to the most cooherent using the bigram model

Note: In cluster_text.py : the path for storing the csv and json files,requried to form the chord diagram, has been hardcoded to the js directory ensure to change it to your path
in your directory. (Line number:251,255,258,268,274)

Pre-requisites to the project:
1. Ensure the files-extract_books,cluster_text,spell_check,language_model,text_segment are in the same folder as server.py
2.The folders Dataset-1,js,static and template must be loacted where server.py is
3. Remember to copy your dataset into js folder, else the documents won't download.

Running the project
1. We first run cluster_text.py to form the clusters for the given dataset. This may require several re-runs but once the 
result converges we save it as a model using joblib. cluster_text.py run as python cluster_text.py "path_to_dataset"
2. We then ingest the documents along with the clustering information by running ingest.py as python ingest.py "path_to_dataset"
3. Finally, we run server.py as python server.py "path_to_dataset"
