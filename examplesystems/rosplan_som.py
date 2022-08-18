import sys
from os.path import dirname
from xml.etree.ElementPath import prepare_predicate
from networkx import MultiDiGraph
sys.path.append(dirname("./dialogsystem"))
from dialogsystem.models.convqa import ConvQASystem
from dialogsystem.models.kgqueryextract import LightningKGQueryMPNN
from dialogsystem.kb_dial_management.kb_dial_manager import DialogueKBManager
from dialogsystem.models.triples2text import Triples2TextSystem
#sys.path.append(dirname("../ebbhrd_hrd/src"))
#from src.EBB_Offline_Interface import EBB_interface
import math
import numpy
from networkx import compose_all
import pymongo



class RosplanDialogueManager(DialogueKBManager):
    def __init__(self, mpnn, convqa, triples2text,db_port = 27017, knowledge_base_args={'session':(2022,4,26)}) -> None:
        client = pymongo.MongoClient("mongodb://localhost:62345/")
        self.db = client["database_test"]

        super().__init__(knowledge_base_args, mpnn, convqa, triples2text)
    
    def initialise_kbs(self, **knowledge_base_args) -> list[MultiDiGraph]:
        query = {"SESSION_NUM":int(input("Which session : "))}

        #knowledgeitem graph (state of the world)
        collection = self.db["knowledgeitems"]
        queryresults = collection.find(query)
        rosplan_knowledgeitems_graph = MultiDiGraph()
        for res in queryresults:
            if res["knowledgeItem"]["knowledge_type"]==1:
                subject = res["knowledgeItem"]["values"][0]["value"]
                
                object = res["knowledgeItem"]["values"][-1]["value"]
                label = self.predicate_mapping(res["knowledgeItem"]["attribute_name"])
                if res["update_type"] == "add_knowledge":
                    rosplan_knowledgeitems_graph.add_edge(subject,object,
                    label = label)
                elif res["update_type"] == "remove_knowledge":
                    rosplan_knowledgeitems_graph.remove_edge(subject,object)
                
        # graph for pick and place results with failure message
        task_result_collection = self.db["pickplaceresults"]
        task_query_results = task_result_collection.find(query)
        task_result_graph = MultiDiGraph()
        for res in task_query_results:
            
            subject = "Action "+res["action_type"]
            label = "has"
            
            object = res["result"]
            task_result_graph.add_edge(subject,object, label = label)


        # success and failure count messages
        successcount_collection = self.db["successcounts"]
        successcount_results = successcount_collection.find(query)
        successcount_graph = MultiDiGraph()
        for res in successcount_results:
            
            subject = res["count_name"]
            label = "is"
            
            object = str(res["count"])
            successcount_graph.add_edge(subject,object, label = label)

        # get the messages in the database collection "plan" which describe the actions to be taken
        plan_collection = self.db["plan"]
        plan_results =plan_collection.find(query)
        plan_graph = MultiDiGraph()
        node_before = "start of plan"
        for res in plan_results:
            split_action = res["plan_action"].split(":")
            subject = split_action[1]
            label = "is part of"
            object = "Plan 1" # TODO change the number of plan so it is updated when new plan is generated
            plan_graph.add_edge(subject,object, label = label)
            if (node_before == "start of plan"):
                plan_graph.add_edge(node_before, subject, label = "is")
            else:
                plan_graph.add_edge(node_before, subject, label = "before")
                plan_graph.add_edge(subject, node_before, label = "after")
            node_before = subject
        plan_graph.add_edge(subject, "end of plan", label = "is")

        return [rosplan_knowledgeitems_graph, task_result_graph, successcount_graph, plan_graph]
    

    def predicate_mapping(self, predicate):
        if predicate == "robot_at_wp":
            return "is at"
        if predicate == "box_at_wp":
            return "is on table at"
        if predicate == "box_on_robot":
            return "is held by"
        if predicate == "robot_does_not_have_box":
            return "is not holding a box"
        if predicate == "wp_visited":
            return "Robot has visited"
        if predicate == "robot_done_pick_place":
            return "is done with picking or placing"

    
    def print_textual_logs(self):
        data = {
            'questions': self.logs['questions'][-1],
            'extracted_triples': self.logs['extracted_triples'][-1],
            'extracted_text': self.logs['extracted_text'][-1],
            'extracted_answers': self.logs['extracted_answers'][-1],
            # 'extracted_context': self.logs['extracted_context'],
        }
        print(data)

if __name__ == '__main__':
    convqa = ConvQASystem("./dialogsystem/trained_models/convqa")
    triples2text = Triples2TextSystem("./dialogsystem/trained_models/t2t/t2ttrained")
    mpnn = LightningKGQueryMPNN.load_from_checkpoint("dialogsystem/trained_models/gqanew.ckpt")
    mpnn.avg_pooling=False
    
    rdm = RosplanDialogueManager(mpnn,convqa,triples2text)
    quit = False
    rdm.triples2text = lambda x: x
    while not quit:
        user_input = input("input: ")
        if user_input == "quit":
            quit = True
        # elif user_input == "disable triples2text":     
        else:
            print(rdm.question_and_response(user_input))
            rdm.print_textual_logs()
    # rdm.question_and_response("where is the person in relation to the robot")
    # rdm.question_and_response("Who did you talk to?")
    # rdm.question_and_response("and to whom did you bring the object?")
    # rdm.question_and_response("what did you bring the person?")
    # rdm.question_and_response("where did you find the object?")
    # rdm.question_and_response("what objects did you see on the table?")
    # rdm.question_and_response("what did you do after picking up the plant?")
    rdm.save_logs("logs/")