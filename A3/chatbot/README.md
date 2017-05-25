A neural chatbot using sequence to sequence model with
attentional decoder. This is a fully functional chatbot.

This is based on Google Translate Tensorflow model 
https://github.com/tensorflow/models/blob/master/tutorials/rnn/translate/

Sequence to sequence model by Cho et al.(2014)

Created by Chip Huyen as the starter code for assignment 3,
class CS 20SI: "TensorFlow for Deep Learning Research"
cs20si.stanford.edu

The detailed assignment handout and information on training time can be found at http://web.stanford.edu/class/cs20si/assignments/a3.pdf 


## Usage

**Step 1**: create a data folder in your project directory, download
the [Cornell Movie-Dialogs Corpus](
https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html).
Unzip it.

```
wget http://www.mpi-sws.org/~cristian/data/cornell_movie_dialogs_corpus.zip
unzip cornell_movie_dialogs_corpus.zip
rm -fr __MACOSX
```

**Step 2**: Do pre-processing for the Cornell dataset.
```
python3 data.py
```

**Step 3**: Train the model
```
python chatbot.py --mode train
```

If mode is train, then you train the chatbot. By default, the model will
restore the previously trained weights (if there is any) and continue
training up on that.

If you want to start training from scratch, please delete all the checkpoints
in the checkpoints folder.

**Step 3**: Start chatting !!
```
python chatbot.py --mode chat
```

If the mode is chat, you'll go into the interaction mode with the bot.

By default, all the conversations you have with the chatbot will be written
into the file output_convo.txt in the processed folder. If you run this chatbot,
I kindly ask you to send me the output_convo.txt so that I can improve
the chatbot. My email is huyenn@stanford.edu

If you find the tutorial helpful, please head over to <a href="http://web.stanford.edu/class/cs20si/anonymous_chatlog.pdf">Anonymous Chatlog Donation</a>
to see how you can help us create the first realistic dialogue dataset.

Thank you very much!