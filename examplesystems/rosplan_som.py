import sys
from os.path import dirname
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
import re

# MetaDialogueManager: switches between RPPlanDM and RPADM acc. to the question, contains list of plan_graphs for different plans, and actions and associated action_graphs for each plan
# RPPlanDialogueManager plan dialogue: given problem and domain, we talk about the associated plan
# RPActionDialogueManager action dialogue (related to state of the world after an action)



class RPActionDialogueManager(DialogueKBManager):
    def __init__(self, mpnn, convqa, triples2text, session_num, knowledge_base_args={'session':(2022,4,26)}) -> None:
        client = pymongo.MongoClient("mongodb://localhost:62345/")
        self.db = client["database_test"]
        self.session_num = session_num

        super().__init__(knowledge_base_args, mpnn, convqa, triples2text)

    def initialise_kbs(self, **knowledge_base_args) -> list[MultiDiGraph]:
        ## proceses knowledgeitem graph (state of the world)
        collection = self.db["knowledgeitems"]
        queryresults = collection.find({"SESSION_NUM": self.session_num})
        graph_list = []
        current_graph = MultiDiGraph()
        

        ## assumes that the knowledgeitems are in order of action_id
        i = 0 # tracker of action_id to be included in kg
        for res in queryresults:
            if res["action_id"] > i:
                graph_list.append(current_graph)
                current_graph = current_graph.copy()
                i +=1
            if res["knowledgeItem"]["knowledge_type"]==1: # this is of type FACT
                subject = res["knowledgeItem"]["values"][0]["value"]
                
                object = res["knowledgeItem"]["values"][-1]["value"]
                label = self.predicate_mapping(res["knowledgeItem"]["attribute_name"])
                if res["update_type"] == "add_knowledge":
                    current_graph.add_edge(subject,object,
                    label = label)
                elif res["update_type"] == "remove_knowledge":
                    current_graph.remove_edge(subject,object)
        graph_list.append(current_graph) # add the last graph
        return graph_list

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
        

class RosplanDialogueManager(DialogueKBManager):
    def __init__(self, mpnn, convqa, triples2text, session_num, knowledge_base_args={}) -> None:
        client = pymongo.MongoClient("mongodb://localhost:62345/")
        self.db = client["database_test"]
        self.session_num = session_num
        self.ActionDialogueManager = RPActionDialogueManager(mpnn, convqa, triples2text, session_num, knowledge_base_args)
        self.action_kbs = self.ActionDialogueManager.kbs
        self.selected_action = None

        super().__init__(knowledge_base_args, mpnn, convqa, triples2text)
        self.kbs.append(self.action_kbs[-1])
        print(self.kbs)

    def _process_plan_db(self) -> MultiDiGraph:
        # get the messages in the database collection "plan" which describe the actions to be taken
        plan_collection = self.db["plan"]
        plan_results =plan_collection.find({"SESSION_NUM":self.session_num})
        plan_graph = MultiDiGraph()
        node_before = "start of plan"

        # print the plan for second prompt
        print("Here is the plan: \n")
        print("State -1 Initial State")
        i = 0
        for res in plan_results:
            subject = self.naturalise_plan_action(res["plan_action"])
            print(f"Action {i} {subject} ")
            i += 1
            # **************** plan_graph generation **********************************
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
        return plan_graph

    def _check_for_action_question(self, user_input) -> bool:
        # check if the user input is a question about an action and set the action number
        match = re.search(r'tell me about action (\d+)', user_input)
        if match:
            self.selected_action = int(match.group(1))
            self.ActionDialogueManager.kbs = [self.action_kbs[self.selected_action]]
            return True
        return False



    def _process_pick_and_place_db(self) -> MultiDiGraph:
        # graph for pick and place results with failure message
        task_result_collection = self.db["pickplaceresults"]
        task_query_results = task_result_collection.find({"SESSION_NUM":self.session_num})
        task_result_graph = MultiDiGraph()
        for res in task_query_results:
            
            subject = "Action "+str(res["action_id"])
            label = "has"
            
            object = res["result"]
            task_result_graph.add_edge(subject,object, label = label)
        return task_result_graph

    def _process_success_count_graph(self) -> MultiDiGraph:
         # success and failure count messages
        successcount_collection = self.db["successcounts"]
        successcount_results = successcount_collection.find({"SESSION_NUM":self.session_num})
        successcount_graph = MultiDiGraph()
        for res in successcount_results:
            
            subject = res["count_name"]
            label = "is"
            
            object = str(res["count"])
            successcount_graph.add_edge(subject,object, label = label)
        return successcount_graph
    
    def initialise_kbs(self, **knowledge_base_args) -> list[MultiDiGraph]:

        plan_graph = self._process_plan_db()

        #next line of code is no longer needed 
        #action_query = {"$lte":int(input("Enter number of action you would like to know about : "))}  #$lte: means less than or equal, so we generate a knowledge graph using all the items UP UNTIL that action (included)

        task_result_graph = self._process_pick_and_place_db()
        successcount_graph = self._process_success_count_graph()


        return [task_result_graph, successcount_graph, plan_graph]
    
    def question_and_response(self, question: str):
        ### Handles mode switching???
        if self.selected_action is not None:
            self.ActionDialogueManager.question_and_response(question,)
        elif self._check_for_action_question(question):
            return "Ok, I will tell you about action "+str(self.selected_action)
        else:
            return super().question_and_response(question)
    
    def naturalise_plan_action(self,pddl_string):
        # input string is of form : "0.000: (move tiago init wp1)  [2.571] "
        pddl_string = pddl_string.split(": ") 
        action_duration = pddl_string[1].split("  ")
        action = action_duration[0][1:-1].split()
        duration = action_duration[1][1:-1]
        if action[0] == "move":
            return "move from "+action[2]+" to "+action[3]+" in "+duration+ " s"
        elif action[0] == "grasp":
            return "pick "+action[2]+" from table at "+action[3]+" in "+duration+ " s"
        elif action[0] == "place":
            return "place from "+action[2]+" onto table at "+action[3]+" in "+duration+ " s"
        return "None"

    
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
    
    rdm = RosplanDialogueManager(mpnn,convqa,triples2text, session_num=2)
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