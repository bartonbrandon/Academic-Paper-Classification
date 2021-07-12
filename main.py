import os

from bokeh.models.widgets import FileInput, TextAreaInput, Button
from bokeh.models import CustomJS
from pybase64 import b64decode
import io
import pandas as pd
from bokeh.io import curdoc
from bokeh.layouts import row, column, layout
from bokeh.models import HoverTool, ColumnDataSource, Div, MultiSelect, Select, Slider, TextInput, ColorBar, \
    LinearColorMapper, LinearAxis, Range1d, Toggle, Paragraph, Span, BoxZoomTool
from bokeh.plotting import figure, output_file, show, save
from bokeh.palettes import Inferno256
from bokeh.transform import linear_cmap, transform
from bokeh.models.widgets import Panel, Tabs
from bokeh.io import show
from bokeh.models import CustomJS, TextInput
# from model_class import mod
from sklearn.model_selection import train_test_split

from FK_score import add_fk_scores
from LDA import add_lda_topics
from SVM import SVM_model

setup_data = False
data_file = "data_file.csv"

svm_model = SVM_model()
svm_model.setup()

df = pd.read_csv("Citations-Project/data_file/data.csv")
publishers = ["Choose publisher..."] + sorted(list(dict.fromkeys(df["Publishers"])))

abstract_input = TextAreaInput(value="", title="Paste abstract here:", height=250, max_length=5000)
abstract_input.js_on_change("value", CustomJS(code="""
    console.log('text_input: value=' + this.value, this.toString())
"""))

title_input = TextInput(value="", title="Enter paper title here")
title_input.js_on_change("value", CustomJS(code="""
    console.log('text_input: value=' + this.value, this.toString())
"""))

author_number = Select(title="Enter the number of authors:", value="0",
                       options=["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"], width=60)
author_number.js_on_change("value", CustomJS(code="""
    console.log('select: value=' + this.value, this.toString())
"""))
error_message = Div(text="")

publisher_input = Select(title="Select the publisher you wish to use:", value=publishers[0], options=publishers)
publisher_input.js_on_change("value", CustomJS(code="""
    console.log('select: value=' + this.value, this.toString())
"""))

p = Paragraph(text="Based on the information you entered, your paper is expected to garner:")

y1 = Paragraph(text="-")
y1text = Paragraph(text=" citations after 1 year")
y5 = Paragraph(text="-")
y5text = Paragraph(text=" citations after 5 years")
y10 = Paragraph(text="-")
y10text = Paragraph(text=" citations after 10 years")
grid = row(column([y1, y5, y10]), column([y1text, y5text, y10text]))


def setup_data_file():
    input_file = "Citations-Project/input_file/" + os.listdir("Citations-Project/input_file")[0]
    df = pd.read_csv(input_file)
    df = add_fk_scores(df)
    df = add_lda_topics(df)
    df.to_csv("Citations-Project/data_file/data.csv")


def setup():
    if setup_data:
        setup_data_file()



def make_row_to_predict_from(title, authors, abstract, paper, publisher, year):
    df = pd.DataFrame()
    df["Title"] = [title]
    df["Abstract Text"] = [abstract]
    df["Age of Paper"] = [year]
    df["No. Authors"] = [authors]
    df["Length of Title"] = [len(title)]
    df["Length of Abstract"] = [len(abstract)]
    df["Publisher_Scores"] = [svm_model.publisher_score[publisher]]
    df = add_fk_scores(df)
    df = add_lda_topics(df, 20, 20)
    X = df[["Length of Title", "Length of Abstract", "Age of Paper", "No. Authors",
            "Abstract FK grade", "Title FK grade", "Topic 1", "Topic 2", "Topic 3",
            "Topic 4", "Topic 5", "Topic 6", "Topic 7", "Topic 8", "Topic 9", "Topic 10", "Topic 11",
            "Topic 12", "Topic 13", "Topic 14", "Topic 15", "Topic 16", "Topic 17", "Topic 18",
            "Topic 19", "Topic 20", "Publisher_Scores"]]
    X_pred = X.values
    return X_pred


def calculate(title, authors, abstract, paper, publisher):
    X_1yr = make_row_to_predict_from(title, authors, abstract, paper, publisher, 1)
    X_5yr = make_row_to_predict_from(title, authors, abstract, paper, publisher, 5)
    X_10yr = make_row_to_predict_from(title, authors, abstract, paper, publisher, 10)
    print(X_1yr)
    pred_1yr = round(svm_model.predict(X_1yr)[0])
    pred_5yr = round(svm_model.predict(X_5yr)[0])
    pred_10yr = round(svm_model.predict(X_10yr)[0])
    print(f"\n\n\n calculated: {pred_1yr}, {pred_5yr}, {pred_10yr} \n\n\n")
    return (pred_1yr, pred_5yr, pred_10yr)


def get_paper_string():
    if ".pdf" in file_input.filename:
        import urllib.request
        import PyPDF2
        import os
        os.chdir("../../Downloads/")

        pdf_name = file_input.filename
        pdfFileObj = open(pdf_name, 'rb')
        pdfReader = PyPDF2.PdfFileReader(pdfFileObj)
        pdfText = ""
        for page in range(pdfReader.numPages):
            pageObj = pdfReader.getPage(page)
            pdfText += pageObj.extractText()

        start = 0
        while start != -1:
            start = pdfText.find('\n', start)
            if start != -1:
                pdfText = pdfText[:start] + pdfText[start + 1:]
        start = 0
        pdfText = pdfText.replace(u"\u2122", "'")
        pdfText = pdfText.replace('ﬁ', '"')
        pdfText = pdfText.replace('ﬂ', '"')
        data = pdfText

        os.chdir("../Desktop/PycharmProjects")
    else:
        data = str(b64decode(file_input.value))
    return data


y1.text = " - "


def update_output():
    title = title_input.value
    authors = author_number.value
    abstract = abstract_input.value
    publisher = publisher_input.value
    paper = get_paper_string()
    if check_for_errors():
        y1.text = "-"
        y5.text = "-"
        y10.text = "-"
    else:
        y1.text = str(calculate(title, authors, abstract, paper, publisher)[0])
        y5.text = str(calculate(title, authors, abstract, paper, publisher)[1])
        y10.text = str(calculate(title, authors, abstract, paper, publisher)[2])


def check_for_errors():
    errors = False
    error_message.text = "<p style='color:red;'>"
    if file_input.value == "":
        errors = True
        error_message.text += "<br>Error: No file uploaded!"
    if author_number.value == "0":
        errors = True
        error_message.text += "<br>Error: Number of authors must be greater than 0"
    if title_input.value == "":
        errors = True
        error_message.text += "<br>Error: Title cannot be empty"
    if len(title_input.value) > 4000:
        errors = True
        error_message.text += "<br>Error: Title is too long!"
    if title_input.value == "":
        errors = True
        error_message.text += "<br>Error: Abstract cannot be empty"
    if publisher_input.value == "Choose publisher...":
        errors = True
        error_message.text += "<br>Error: Must select a publisher"
    if len(abstract_input.value) > 10000:
        errors = True
        error_message.text += "<br>Error: Abstract is too long!"
    return errors


button = Button(label='Predict citations', button_type='success', aspect_ratio=3)
# button.js_on_click(CustomJS(code="console.log('button: click!', this.toString())"))
button.on_click(update_output)

setup()

file_input = FileInput(accept=".txt,.pdf", name="Import paper here")

controls = [file_input, title_input, author_number, publisher_input, abstract_input, button, error_message]
displays = [p, grid]

inputs = column(controls, width=500, height=1000)
outputs = column(displays, width=500, height=100)

inputs2 = column(Div(text="<h1>Inputs</h1>"), inputs)
outputs2 = column(Div(text="<h1>Outputs</h1>"), outputs)
final_row = row(inputs2, outputs2)
pan_oi = Panel(child=final_row, title="Citations")

tabs = Tabs(tabs=[pan_oi])
curdoc().add_root(tabs)
