### Parse OWL ontology
### Extract KG from episodic memories
import sys
from os.path import dirname
from networkx import DiGraph
sys.path.append(dirname("./dialogsystem"))
from dialogsystem.models.kgqueryextract import LightningKGQueryMPNN
from dialogsystem.kb_dial_management.kb_dial_manager import DialogueKBManager
from dialogsystem.models.triples2text import Triples2TextSystem
sys.path.append(dirname("../ebbhrd_hrd/src"))
from src.EBB_Offline_Interface import EBB_interface

class RobotDialogueManager(DialogueKBManager):
    def __init__(self, mpnn, convqa, triples2text,db_port = 27017, knowledge_base_args={'session':(2022,4,11)}) -> None:
        self.ebb_interface = EBB_interface(port=db_port)
        super().__init__(knowledge_base_args, mpnn, convqa, triples2text)


    def initialise_kbs(self, session) -> list[DiGraph]:
        """
        session_date: <(int,int,int)> (yyyy,mm,dd) 
        """
        sessions = self.ebb_interface.getSessionNums_date(*session)
        output = self.ebb_interface.getCollectionFromEBB(["observations_coll"],sessions)
        statement_array =[]
        for output in output:
            _, statement_array = self.ebb_interface.state_observations_list(output["observations_coll"], statement_array)
        
            print("\n\n\n\n lets a go")
            print(statement_array[0])

if __name__ == '__main__':
    convqa = lambda x: "i'm not sure"
    triples2text = Triples2TextSystem.load_from_checkpoint("dialogsystem/trained_models/t2t.ckpt")
    mpnn = LightningKGQueryMPNN.load_from_checkpoint("dialogsystem/trained_models/meddim.ckpt")
    mpnn.k = 3
    rdm = RobotDialogueManager(mpnn,convqa,triples2text)