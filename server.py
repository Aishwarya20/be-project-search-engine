#back-end : The clustering module is first invoked
#front-end server side script to help bring together the various modules
#two tabs search by conetent and search by author/book title exist. For both the modules separate vocabulary for quicker front end validation
#elasticsearch is invoked to perform searching on the ingested documents
#doc.html->this page displays retrieves the search result(ninja template)
#query.html->this shall take the query and pass it to text segment, spell check as well as language modelling
import spell_check as sc
import text_segment as ts
import language_model as lm
from flask import Flask
from flask import request
from flask import render_template,send_from_directory
import string
import re
from nltk.corpus import stopwords
from collections import Counter
from elasticsearch import Elasticsearch
import cluster_text as ct
import sys
import extract_books as eb
import ingest
ct.main(sys.argv[1])
stopword=set(stopwords.words("english"))
es=Elasticsearch()
print es.indices.delete(index='top100', ignore=[400, 404])
ingest.main(sys.argv[1])
app = Flask(__name__,static_folder='js',template_folder='template')

@app.route('/js/<path:path>/')
def static_page(page_name):
    return send_from_directory('js' ,path)

@app.route('/',methods=["GET","POST"])
def process_author():

    answer1=' '.join(i for i in sc.display_author(request.form['query_author'].lower()))
    print answer1

    answer2=ts.display_segment_author(answer1)
    print answer2

    final=lm.display_lang_author(answer2)
    dloads=[]
    keys=[]
    error=""
    if final == "":
        error="No results to found"
    else:
        tokens=re.findall("[a-zA-Z']+",final.lower())
        new_tokens=list(set(tokens)-stopword)
        new_final=' '.join(new_tokens)
        res=es.search(index="top100",body={"query":{"bool":
                    {"should":
                    [{"match":{"author":new_final}},
                     {"match":{"title":new_final}}
                    ],
                    "minimum_should_match" : 1
                    }
                    }})

        for hit in res['hits']['hits']:
            keys.append(hit['_source']['title'])
            dloads.append("http://127.0.0.1:5000/js/Dataset-1/"+hit['_source']['bookname'])



    return render_template('doc.html',dloads=dloads,keys=keys,query=final,error=error)

    #return  '<br>'.join(s2.extract_book_path(final))

@app.route('/query.html',methods=["GET","POST"])
def process_content():

    answer_content1=' '.join(i for i in sc.display_content(request.form['query_content'].lower()))
    print answer_content1
    answer_content2=ts.display_segment_content(answer_content1)
    print answer_content2
    final_content=lm.display_lang_content(answer_content2)
    dloads={}
    keys={}
    error=""
    count={}
    if final_content == "":
        error= "No results to found"
    else:
        tokens=re.findall("[a-zA-Z']+",final_content.lower())
        new_tokens=list(set(tokens)-stopword)
        new_final_content=' '.join(new_tokens)

        res=es.search(index="top100",body={"query":
                     {"match":{"text":new_final_content}},
                     })
        flag=[]
        n_clusters=[]
        for hit in res['hits']['hits']:
            n_clusters.append(hit['_source']['cluster'])
        count=Counter(n_clusters)
        for key in count:
            dloads[key]=[]
            keys[key]=[]
        for i in range(0,len(res['hits']['hits'])):
            flag.append(False)
        for hit1 in res['hits']['hits']:
            i=0
            for hit2 in res['hits']['hits']:
                if hit1['_source']['cluster'] == hit2['_source']['cluster'] and flag[i] == False:
                    dloads[hit1['_source']['cluster']].append("http://127.0.0.1:5000/js/Dataset-1/"+hit2['_source']['bookname'])
                    keys[hit1['_source']['cluster']].append(hit2['_source']['title'])
                    flag[i]=True
                i=i+1

    return render_template('content.html',dloads=dloads,keys=keys,count=count,query=final_content,error=error)



if __name__ == '__main__':
    app.run()
