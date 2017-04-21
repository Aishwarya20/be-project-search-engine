import os
import re
import sys
from datetime import datetime
from elasticsearch import Elasticsearch
import codecs
es = Elasticsearch()

def extract_title_author(file_name):
    contents = open(file_name).read()
    title_re = re.compile(r"Title\:.+")
    author_re = re.compile(r"Author\:.+")

    title_match = title_re.search(contents)
    author_match = author_re.search(contents)

    title = ""
    author = ""

    if title_match:
        title = title_match.group()
        title = title.strip().replace("Title: ", "")

    if author_match:
        author = author_match.group()
        author = author.strip().replace("Author: ", "")

    return (title, author)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        dir_to_process = sys.argv[1]
    else:
        print "Please specify input directory"
        sys.exit(-1)

    file_to_save = open("authors.txt", "w")
    ebook_cluster={}
    with open("doc_to_cluster_map.csv") as csv_file:
        for row in csv_file:
            value,index= row.split(',')
            #print value,index
            ebook_cluster[index.replace("\n","")]=value
    #print ebook_cluster



    info = {
    "1jcfs10.txt": {"title": "The History of the Thirty Years' War", "author": "Friedrich Schiller"},
    "10028.txt":{"title":"Spalding's Official Baseball Guide - 1913","author":"John B. Foster"},
    "allry10.txt":{"title":"Contributions to All The Year Round","author":"Charles Dickens"},
    "balen10.txt":{"title":"The Tale of Balen","author":"Algernon Charles Swinburne"},
    "baleng2.txt":{"title":"Ancient Poems,Ballads and Songs of the Peasantry of England","author":"Robert Bell"},
    "batlf10.txt":{"title":"The Battle of Life","author":"Charles Dickens"},
    "bgopr10.txt":{"title":"The Beggar's Opera","author":"John Gay"},
    "bstjg10.txt":{"title":"Beast in the Jungle","author":"Henry James"},
    "crsnk10.txt":{"title":"The Cruise of the Snark","author":"Jack London"},
    "mklmt10.txt":{"title":"the Makaloa Mat/Island Tales","author":"London"},
    "mspcd10.txt":{"title":"Miscellaneous Papers","author":"Charles Dickens"},
    "rlsl110.txt":{"title":"The Letters of Robert Louis Stevenson","author":"Robert Louis Stevenson"},
    "rlsl210.txt":{"title":"Letters of Robert Louis Stevenson","author":"Robert Louis Stevenson"},
    "sesli10.txt":{"title":"Sesame and Lilies","author":"John Ruskin"},
    "svyrd10.txt":{"title":"Songs of a Savoyard","author":"W. S. Gilbert"},
    "utrkj10.txt":{"title":"Unbeaten Tracks in Japan","author":"Bird"},
    "vpasm10.txt":{"title":"Vailima Prayers & Sabbath Morn","author":"Robert Louis Stevenson"},
    "wldsp10.txt":{"title":"Shorter Prose Pieces","author":"Oscar Wilde"},
    "zncli10.txt":{"title":"The Zincali","author":"George Borrow"}}

    for current_doc_id, current_file in enumerate(os.listdir(dir_to_process)):

        # Skip Hidden Files
        if current_file[0] == ".":
            continue

        if current_file in info:
            data = info[current_file]
            title = data["title"]
            author = data["author"]
        else:
            title, author = extract_title_author(
                os.path.join(dir_to_process, current_file))

        doc = {
            'author': author.lower(),
            'title': title.lower(),
            'bookname':current_file,
            'text': codecs.open(os.path.join(dir_to_process, current_file), "r",encoding='utf-8', errors='ignore').read(),
            'timestamp': datetime.now(),
            'cluster':ebook_cluster[current_file]
              }

        rec = "%s\n" % author
        file_to_save.write(rec)

        try:
            res = es.index(index="top100", doc_type='ebook', id=current_doc_id, body=doc)
            print res['created'],current_file


        except:
            print "Cannot index:%s" % current_file
