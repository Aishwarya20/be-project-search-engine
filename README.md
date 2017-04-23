Search Engine For an E-book Portal

Following are the modules implemented
1. Spell Checker: This helps correct spelling using the edit distance method
2.Text Segmentation: Segments the text using the bigram model
3.Language Modelling: Corrects the sequence of words to the most cooherent using the bigram model

Note: In cluster_text.py : the path for storing the csv and json files,requried to form the chord diagram, has been hardcoded to the js directory ensure to change it to your path
in your directory. (Line number:251,255,258,268,274)

Running the project:
1. Ensure the files-extract_books,cluster_text,spell_check,language_model,text_segment are in the same folder as server.py
2.Now the Dataset-1 can be stored wherever convenient
3.The folders js,static and template must be loacted where server.py is
4.Finally, we run the project in the command line as: python server.py "path_to_dataset"

