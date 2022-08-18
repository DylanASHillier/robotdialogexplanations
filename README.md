# robotdialogexplanations
System for generating explanations for the actions of robotic systems.
The rosplan-som branch is Catherine Ning's summer internship work where we try to integrate the question-answering system with the ROSPlan-TIAGo setup. See https://github.com/cognitive-robots/rosplan-som for details.

## Installation
Just run pip install -r requirements.txt -- you probably want to do the torch install first though to get the right package for your graphics card setup if you have one.

## Setup steps:
Steps:
1. run `setup.py`
2. run the command `gzip -d datasets/KGs/conceptnetassertions.csv.gz`

## Recreate Datasets:
### Associating questions, answers with knowledge graphs
In the first step we must pair up knowledge graphs with the questions and answers we will use for training
#### QText datasets (ROPES,COQA,SQUAD)
You will firstly need to set up the conceptnet knowledge graph which is used as the baseline graph
This can be done using the script setupconceptnet.py in dialogsystem.
The next step is to run qtext_data_prep.py
-- Note that I only did this for on the order of 1000 samples before terminating the script
#### GQA dataset
Just run scene_graph_data_prep.py
### Embedding graphs with questions
1. Firstly train an autoencoder using the train script train.py with argument --system_type=autoencoder and output path --output_path=dialogsystem/trained_models/autoencoder.ckpt
1. The embedding can then by done by calling an instance of the GraphTrainDataset class -- it is handled in the init function.
## Train Models:
### Graph Models:
Use the train.py script with setting --system_type=gnn
### Language Models:
I just utilised fairly standard scripts that use the hugging face library. Text generation is definitely an issue here -- I didn't have time to fiddle with the parameters much at all
-- in particular use the conv_qa_pipeline and triples_to_text_pipeline for doing this training
## Evaluation and Visualisation
There are a number of scripts within the dialogystem module that can be used for creating evaluations and visualisations as seen in the report -- see the package for more detail
## Example Systems:
### Robot doing pickup tasks
run `examplesystems/ebbhrd.py`

## Next Steps:
1. I recommend trying out different language models. The recent release of OPT for example would be a nice thing to try for the triple-to-text generation --> simply include an example of a successful triple text translation in the prompt and it should be able to translate triples in a zero shot setting.
2. Try to keep track of models better than I did! I've got a bunch of .ckpt files to load in graph models
3. Deal with 1-place predicates
4. Deal with empty graphs properly
5. Temporal Component (mpnn-lstm)
6. Change triple candidates to be top p
## Miscellaneous
### AWS Tips:
1. I had a lot of issues with installs on AWS instances due to the varying GPUs, CUDA levels etc. In particular you may have problems with installing the torch geometric packages which seem particularly problematic.
2. I recommend keeping two AWS instances in a stopped state: one with high memory GPUs for running inference and training the language/graph models. Another with large CPU memory for doing dataset/graph generation. This is essential for generating the datasets fresh. 
3. If you have trouble finding out how to utilise all the memory on your instances 
use this link for setting up the internal storage: [aws mount file systems](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ebs-using-volumes.html).

