from flask import Flask, jsonify, request
from flask import render_template
from flask import Response
from werkzeug.utils import secure_filename
import esummarizer as es
import os, os.path
import PyPDF2
#UPLOAD_FOLDER = '/Users/guathwalow/Documents/workarea/capstone/temp/'
UPLOAD_FOLDER = './temp/'

#nltk.download('punkt') # one time execution
#nltk.download('popular')
#-----------------------------------------------------
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def hello_world():
    return 'Hello, World!'

@app.route('/getasummary2', methods = ["GET", "POST"])
def get_text_s():
    output1 = None
    output2 = None
    errors = []
    #output3 = None

    if request.method == "POST":
        try:
            file = request.files["file"]
            url = request.form["url"]
            block = request.form["block"]
            numsent = request.form["numsent"]
        except:
            errors.append(
            "Missing inputs. Please make sure it's valid and try again."
            )
            return render_template('index.html', errors=errors)

        if numsent =="":
            numsent = 5
        input = request.form["input"]
        ssize = request.form["ssize"]
        model1 = request.form.getlist('model1')
        model2 = request.form.getlist('model2')
        #model3 = request.form.getlist('model3')
        print(input)
        print(ssize)
        print(model1)
        print(model2)
        #print(model3)

        # check input type
        if input == 'fromfile':
            try:
                #check file extension (.txt, .pdf)
                file_type = file.filename.split('.')[1]
                print(file_type)
                if file_type == 'txt':
                    text = file.read()
                    #text = str(text.decode("utf-8")).replace("\n","").lstrip("\n")
                    text = str(text.decode("utf-8")).replace("\n"," ")
                    text = text.replace(".",". ")
                elif file_type == 'pdf':
                    filename = secure_filename(file.filename)
                    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    print(filename)
                    print(file_path)
                    file.save(file_path)

                    #textpages = es.readPDF(file_name)
                    #text = ' '.join(textpages)
                    textpages = []
                    pdf_file_object = open(file_path,'rb')
                    #pdf_file_object = open(file_name, 'rb')
                    #print(pdf_file_object)
                    pdfReader = PyPDF2.PdfFileReader(pdf_file_object)
                    numPages = pdfReader.numPages
                    print("No. of pages: ",numPages)
                    for i in range(numPages):
                        page_object = pdfReader.getPage(i).extractText().replace("\n","")
                        textpages.append(page_object)
                        print("length of page: ",len(page_object))
                    text = ' '.join(textpages)

                    #remove uploaded pdf
                    os.remove(file_path)
            except:
                errors.append(
                "Unable to read File. Please make sure it's valid and try again."
                )
                return render_template('summarizer.html', errors=errors)


        elif input == 'fromurl':
            try:
                text = es.readUrl(url)
            except:
                errors.append(
                "Unable to get URL. Please make sure it's valid and try again."
                )
                return render_template('summarizer.html', errors=errors)
        else:
            text = block

        # check summary Size, added 1min,3mins,5mins
        text_word_count = es.get_wordcount(text)
        text_sent_count = es.get_sentencecount(text)
        try:
            if ssize == "s1min":
                top_n = int(200 / text_word_count * text_sent_count)

            elif ssize == "s3mins":
                top_n = int(600 / text_word_count * text_sent_count)

            elif ssize == "s5mins":
                top_n = int(1000 / text_word_count * text_sent_count)
            else:
                top_n = int(numsent)
        except:
            errors.append(
            "Missing values. Please make sure it's valid and try again."
            )
            return render_template('summarizer.html', errors=errors)

        if top_n < 0 or top_n == "":
            top_n=1

        if top_n > text_sent_count:
            top_n = text_sent_count

        print("top_n", top_n)


        # check model selected
        # if none is selected, set it to model1
        if len(model1) == 0 and len(model2) ==0 and len(model3) == 0:
            model1 = ['model1']

        if model2 == ['model2']:
        #word_embeddings = get_word_vectors_glove()
            embedding_path = './glove/glove.6B.100d'
            word_embeddings = es.load_embeddings_binary(embedding_path)
            generated_summary_textrank = es.generate_summary_textrank(text,top_n,word_embeddings)
            output2 = generated_summary_textrank

        if model1 == ['model1']:
            generated_summary_cbow = es.generate_summary_cbow(text,top_n)
            output1 = generated_summary_cbow

        #if model3 == ['model3']:
        #    text_sent_count = es.get_sentencecount(text)
        #    generated_summary_bert = es.generate_summary_bert(text,text_sent_count,top_n)
        #    output3 = generated_summary_bert

    return render_template("summarizer.html", errors=errors, output1 = output1,output2 = output2)
