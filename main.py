import sys
import time

import torch
import transformers
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import pipeline
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import AutoModelForQuestionAnswering, pipeline
from PyQt5.QtWidgets import QApplication, QMainWindow, QTextEdit, QPushButton, QWidget, QVBoxLayout
import logging
#logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

context_global = ''
class SummarizerApp(QMainWindow):
    def __init__(self):
        super().__init__()

        # Load the pre-trained BART model for summarization
        self.summarization_model = transformers.AutoModelForSeq2SeqLM.from_pretrained(r"./bart_summarisation")
        self.summarization_tokenizer = transformers.AutoTokenizer.from_pretrained(r"./bart_summarisation")

        # Load the pre-trained pegasus model for content chunking
        self.chunking_model = transformers.AutoModelForSeq2SeqLM.from_pretrained(r"./pegasus")
        self.chunking_tokenizer = transformers.AutoTokenizer.from_pretrained(r"./pegasus")

        #Load the pre-trained flan-t5-base model for Q and A
        self.model_name = "roberta-qna"
        self.qanda_model = transformers.AutoModelForQuestionAnswering.from_pretrained(self.model_name)
        self.qanda_tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_name)

        # Create the main window
        self.setWindowTitle("AI tool using LLM App")

        # Create the main window and set its geometry (larger size)
        self.setGeometry(100, 100, 800, 600)  # Adjust the dimensions as needed

        # Create the main widget
        self.mainWidget = QWidget()
        self.setCentralWidget(self.mainWidget)

        # Create the input text box
        self.input_text = QTextEdit()
        self.input_text.setPlaceholderText("Enter some text to summarize or chunk or a question:")

        # Create the summarize button
        self.summarize_button = QPushButton("Summarize")
        # first clear the input box
        self.summarize_button.clicked.connect(self.summarize)

        # Create the content chunking button
        self.chunk_button = QPushButton("Content Chunking")
        self.chunk_button.clicked.connect(self.chunk_text)

        # Create the Q and A button
        self.answer_button = QPushButton("Question Answering")
        self.answer_button.clicked.connect(self.qna)


        # Create the output text box
        self.output_text = QTextEdit()
        self.output_text.setPlaceholderText("The output will appear here...")

        # Create the layout for the main widget
        self.layout = QVBoxLayout()
        self.layout.addWidget(self.input_text)
        self.layout.addWidget(self.summarize_button)
        self.layout.addWidget(self.chunk_button)
        self.layout.addWidget(self.answer_button)
        self.layout.addWidget(self.output_text)

        # Set the layout for the main widget
        self.mainWidget.setLayout(self.layout)

    def summarize(self):

        # Get the input text
        input_text = self.input_text.toPlainText()

        self.context = input_text

        context_global = input_text
        # Check if the input text is empty
        if not input_text:
            # Display a message to the user
            print("Please enter some text to summarize.")
            return

        # Tokenize and summarize the input text using the BART model
        input_ids = self.summarization_tokenizer.encode("Summarize: " + input_text, return_tensors="pt", max_length=1024,truncation=True)

        summary_ids = self.summarization_model.generate(input_ids, max_length=200, min_length=40, length_penalty=2.0, num_beams=4,early_stopping=False)


        # Decode the generated summary and display it in the output text box
        summary = self.summarization_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        #print(f"Summary Text: {summary}")
        self.output_text.setPlainText(summary)
        self.input_text.clear()

    def chunk_text(self):
        # Get the input text
        input_text = self.input_text.toPlainText()
        self.context = input_text
        # Check if the input text is empty
        if not input_text:
            # Display a message to the user
            print("Please enter some text to summarize.")
            return

        # Tokenize and summarize the input text using the pegasus model
        input_ids = self.chunking_tokenizer.encode("Generate 5 key points: " + input_text, return_tensors="pt",max_length=1024, truncation=True)
        summary_ids = self.chunking_model.generate(input_ids, max_length=200, min_length=40,length_penalty=2.0, num_beams=4, early_stopping=False)

        chunked_text =self.chunking_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        output = chunked_text.split("<n>")
        print(output)
        # Create a new HTML string, with each sentence on a new line.
        html_string = "<br>".join(output)

        self.output_text.setHtml(html_string)

    def qna(self):
        # Get the input text
        input_text = self.input_text.toPlainText()

        # Check if the input text is empty
        if not input_text:
            # Display a message to the user
            print("Please enter a question")
            return

        nlp = pipeline('question-answering', model=self.model_name, tokenizer=self.model_name)

        QA_input = {
            'question': input_text,
            'context': self.context
            }
        res = nlp(QA_input)
        print('In QNA fn')
        print(res['answer'])
        # Encode the question and context.
        # input_ids = self.qanda_tokenizer(input_text, self.context, return_tensors="pt")

        # Generate the predictions.
        # outputs = self.qanda_model(**input_ids)

        # Get the start and end logits.
        # start_logits = outputs.start_logits
        # end_logits = outputs.end_logits

        # Get the predicted start and end indices.
        # predicted_start_index = start_logits.argmax().item()
        # predicted_end_index = end_logits.argmax().item()
        # Check if the predicted start and end indices are valid.
        # if predicted_start_index >= predicted_end_index:
        #     return "No answer found"

        # Get the answer text from the context.
        # answer_text = self.context[predicted_start_index:predicted_end_index + 1]
        answer_text = res['answer']
        # print(answer_text)
        # #Displaying Output
        self.output_text.setPlainText(answer_text)

if __name__ == "__main__":
    app = QApplication(sys.argv)

    window = SummarizerApp()
    window.show()

    sys.exit(app.exec_())


# Arizona is the 48th state and last of the contiguous states to be admitted to the Union, achieving statehood on February 14, 1912. Historically part of the territory of Alta California in New Spain, it became part of independent Mexico in 1821. After being defeated in the Mexican–American War, Mexico ceded much of this territory to the United States in 1848. The southernmost portion of the state was acquired in 1853 through the Gadsden Purchase.

# Southern Arizona is known for its desert climate, with very hot summers and mild winters. Northern Arizona features forests of pine, Douglas fir, and spruce trees; the Colorado Plateau; mountain ranges (such as the San Francisco Mountains); as well as large, deep canyons, with much more moderate summer temperatures and significant winter snowfalls. There are ski resorts in the areas of Flagstaff, Sunrise, and Tucson. In addition to the internationally known Grand Canyon National Park, which is one of the world's seven natural wonders, there are several national forests, national parks, and national monuments.

# Arizona's population and economy have grown dramatically since the 1950s because of inward migration, and the state is now a major hub of the Sun Belt. 

# When was Arizona admitted to the Union?