## What is this?
This script will take EEG brainwaves and create a static dataset through a sliding window approach. Overlapping windows consider wave data and many mathematical attributes are generated in order to describe the wave. This means for Machine Learning you're not classifying point-data and thus temporal techniques such as an LSTM are no longer necessary 

The script is set to resample and generate a dataset from the csv data format exported by Alexandre Barachant's [MuseLSL](https://github.com/alexandrebarachant/muse-lsl)


## Usage
There is demo Muse EEG data under *dataset/original_data/*

Notice that there is a noise column at the end of the CSV, this would be the Right AUX input to the Muse. The script will ignore this column, so make sure you add a column of zeroes to the end.

Run the following code:

python src/EEG_generate_training_matrix.py dataset/original_data/ out.csv

This specifies the feature extraction script, where the data is stored, and where the final dataset will be output.

Class names are taken from the filename (see example data). If you would like to add custom classes, scroll down to "How do I add a new class?"

## What features are generated?
Please see [this work](https://link.springer.com/chapter/10.1007/978-3-030-29933-0_37) for the full description of features generated (I removed them from here because the readme got far too long!)

## How do I add a new class?
In *EEG_generate_training_matrix.py* there are the cases for mental state classes (see line 31 onwards), simply add/remove cases here and make sure they all have a unique integer state. This integer is inserted into the final column (after all features) of the class in the output

## Where do these features come from?
We have been slowly growing this list of features over time through multiple research papers.

[A Study on Mental State Classification using EEG-based Brain-Machine Interface](https://ieeexplore.ieee.org/abstract/document/8710576) - this work first proposes the technique and features. We used them to classify mental state. 

@inproceedings{bird2018study,  
  title={A study on mental state classification using eeg-based brain-machine interface},  
  author={Bird, Jordan J and Manso, Luis J and Ribeiro, Eduardo P and Ek{\'a}rt, Anik{\'o} and Faria, Diego R},  
  booktitle={2018 International Conference on Intelligent Systems (IS)},  
  pages={795--800},  
  year={2018},  
  organization={IEEE}  
}

[Mental Emotional Sentiment Classification with an EEG based Brain-Machine Interface](https://www.researchgate.net/profile/Jordan_Bird2/publication/329403546_Mental_Emotional_Sentiment_Classification_with_an_EEG-based_Brain-machine_Interface/links/5c2f74c092851c22a35975c5/Mental-Emotional-Sentiment-Classification-with-an-EEG-based-Brain-machine-Interface.pdf) - this work applies the technique to classification of emotional experiences

@inproceedings{bird2019mental,  
  title={Mental Emotional Sentiment Classification with an EEG-based Brain-machine Interface},  
  author={Bird, Jordan J and Ekart, Anik{\'o} and Buckingham, Christopher D and Faria, Diego R},  
  booktitle={Proceedings of the International Conference on Digital Image and Signal Processing (DISP’19)},  
  year={2019}  
}

[A Deep Evolutionary Approach to Bioinspired Classifier Optimisation for Brain-Machine Interaction](https://www.hindawi.com/journals/complexity/2019/4316548/abs/) - the technique is used alongside evolutionary hyperparameter optimisation of neural network topologies and compared to an LSTM 

@article{bird2019deep,  
  title={A Deep Evolutionary Approach to Bioinspired Classifier Optimisation for Brain-Machine Interaction},  
  author={Bird, Jordan J and Faria, Diego R and Manso, Luis J and Ek{\'a}rt, Anik{\'o} and Buckingham, Christopher D},  
  journal={Complexity},  
  volume={2019},  
  year={2019},  
  publisher={Hindawi}  
}

[Classification of EEG Signals Based on Image Representation of Statistical Features](https://link.springer.com/chapter/10.1007/978-3-030-29933-0_37) - this work introduced new features to the generator and then applied them in reforming the signals to images and using a CNN to classify them 

@inproceedings{ashford2019classification,  
  title={Classification of EEG Signals Based on Image Representation of Statistical Features},  
  author={Ashford, Jodie and Bird, Jordan J and Campelo, Felipe and Faria, Diego R},  
  booktitle={UK Workshop on Computational Intelligence},  
  pages={449--460},  
  year={2019},  
  organization={Springer}  
}  



If you cannot access any of these papers, all preprints are available on my [ResearchGate Profile](https://www.researchgate.net/profile/Jordan_Bird2)


## To Do
Make the final column to be ignored optional  
Add more features  
Diagrams for this readme to show what is going on a bit better  

## Acknowledgements
Thank you to Alexandre Barachant for his work on [MuseLSL](https://github.com/alexandrebarachant/muse-lsl)

Special thanks to these guys for their expert guidance and supervision for the research projects mentioned  
*Dr. Diego R. Faria*  
*Dr. Luis J. Manso*  
*Dr. Felipe Campelo*   

## Questions? 
Contact me on birdj1@aston.ac.uk

Thanks,  
[Jordan J. Bird](http://jordanjamesbird.com/)


