# extract names of books and corresponding authors.
#author variable holds all the information regarding book titles and author
import re
import string
import os
import codecs
import json
import sys


def get_files(raw_path):
    files_extracted=[]
    root_extracted=[]
    print ("Files in: " +raw_path)
    root_extracted.append(raw_path)
    for current_doc_id, current_file in enumerate(os.listdir(raw_path)):
        files_extracted.append(current_file)
    return files_extracted,root_extracted



def main(args):
    dir_to_process=args
    files,roots=get_files(dir_to_process)
    info=[]
    books=[]
    string_content=''

    for root in roots:
        for filename in files:
            with codecs.open(root+'\\'+filename, "r",encoding='utf-8', errors='ignore') as file_name:
                text=file_name.readline()
                string_content +='\n'+file_name.read()
                info.append(text.lower())
                book_info=''

                non_useful=['project',"gutenberg's",'ebooks','etexts','etext','ebook','gutenberg','this','presented','file','s']
                result=[word  for word in re.findall('[a-z0-9]+',text.lower()) if word not in non_useful]

                book_info=' '.join(result)
                book_info=re.sub("of","",book_info,count=1).strip()
                books.append(book_info)

        #print books
        #print book_info
        author='\n'.join(books)
        return author,string_content




author,content=main(sys.argv[1])
if __name__=='__main__':
    author,content=main(sys.argv[1])
