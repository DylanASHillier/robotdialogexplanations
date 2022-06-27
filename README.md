# robotdialogexplanations
System for generating explanations for the actions of robotic systems

## Setup steps:
Steps:
1. run `setup.py`
2. run the command `gzip -d datasets/KGs/conceptnetassertions.csv.gz`

## Recreate Datasets:
Steps:


## Train Models:
### Graph Models:
### Language Models:
I just utilised fairly standard scripts that use the hugging face library in a fairly standard way. Text generation is definitely an issue here

## Example Systems:
### Robot doing pickup tasks
run `carerobot.py`

## Next Steps:
1. I recommend trying out different language models. The recent release of OPT for example would be a nice thing to try for the triple-to-text generation --> simply include an example of a successful triple text translation in the prompt and it should be able to translate triples in a zero shot setting.
2. Try to keep track of models better than I did! I've got a bunch of .ckpt files to load in graph models
## Miscellaneous
### AWS Tips:
1. I had a lot of issues with installs on AWS instances due to the varying GPUs, CUDA levels etc. In particular you may have problems with installing the torch geometric packages which seem particularly problematic.
2. I recommend keeping two AWS instances in a stopped state: one with high memory GPUs for running inference and training the language/graph models. Another with large CPU memory for doing dataset/graph generation. This is essential for generating the datasets fresh. 
3. If you have trouble finding out how to utilise all the memory on your instances 
use this link for setting up the internal storage: [aws mount file systems](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ebs-using-volumes.html).

