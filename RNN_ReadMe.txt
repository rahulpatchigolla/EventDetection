1. Download and extract the corpus for "http://nactem.ac.uk/MLEE/" site.
2. Download the pretrained word embeddings with file name "PubMed-w2v.bin" form "http://evexdb.org/pmresources/vec-space-models/" website.
3. Update the path of word embeddings file in loadWordEmbeddings method in Utils.py(for RNN and Other_Argument models).
4. Download and extract gdep parser from "http://www.cs.cmu.edu/~sagae/parser/gdep/" website.
5. Run PrePreprocess.sh file to create the folder structure.
6. Copy all the files form the extracted corpus folder into Corpus folder.
7. Copy the file form ReplaceFile folder and replace it with the file in "./standoff/test/train/" folder (So as to correct an annotation mistake)
8. Copy all files from the extracted gdep parser into gdep-beta2 folder and also run make command.
9. Run Preprocess.sh file to perform preprocessing.
10. Run Trigger.py file to train,test and store best predicted triggers of trigger identification model.
11. Run Process4.sh file to create parsing information for the model
12. Run Argument.py file to train,test and store best predicted arguments of argument identification model w.r.t predicted triggers.
13. Run Process3.py file to create event annotation files based on the predicted triggers and arguments
14. Copy files from Corpus_filtered/test/ and Predicted_test_events/  to Evaluation/Acutal_test_events/ and Evaluation/Predicted_test_events/ and run Evaluation.sh file to get the final results.
15. Run Process5.py,Process6.py getting analysis results.
