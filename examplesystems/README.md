# Use Cases
## EBBHRD
`ebbhrd.py` contains code for interfacing with EBBHRD recordings. To do this it interfaces with the EBBInterface package to obtain recordings of interaction instances, from which it extracts knowledge graphs.

The main class is `RobotDialogManager` which is an implementation of the `DialogueKBManager` class in 
`dialogsystem/kb_dial_management/kb_dial_manager.py`. In the `initialise_kbs` method all the requisite knowledge graph preperation is performed.

## Requirements and Setup
You need to install the EBB Interface package made by Matthew, and additionally obtain/host the database on a mongodb install
