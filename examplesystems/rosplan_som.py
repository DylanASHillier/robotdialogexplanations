import sys
from os.path import dirname
from networkx import MultiDiGraph
sys.path.append(dirname("./dialogsystem"))
from dialogsystem.models.convqa import ConvQASystem
from dialogsystem.models.kgqueryextract import LightningKGQueryMPNN
from dialogsystem.kb_dial_management.kb_dial_manager import DialogueKBManager
from dialogsystem.models.triples2text import PretrainedTriples2TextSystem, Triples2TextSystem
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
    def __init__(self, mpnn, convqa, triples2text, session_num, top_p=0.5, db_string= "scenarios_db", port=27017) -> None:
        client = pymongo.MongoClient(f"mongodb://localhost:{port}/")
        self.db = client[db_string]
        self.session_num = session_num

        super().__init__({}, mpnn, convqa, triples2text, top_p=top_p)

    def initialise_kbs(self, **knowledge_base_args) -> list[MultiDiGraph]:
        ## proceses knowledgeitem graph (state of the world)
        collection = self.db["knowledgeitems"]
        queryresults = collection.find({"SESSION_NUM": self.session_num})
        graph_list = []
        current_graph = MultiDiGraph()
        

        ## assumes that the knowledgeitems are in order of action_id
        i = 0 # tracker of action_id to be included in kg
        ##TODO: YOU MAY WANT i = -1 TO INCLUDE GOALS AND CONDITIONS?
        for res in queryresults:
            if res["action_id"] > i:
                graph_list.append(current_graph)
                current_graph = current_graph.copy()
                i +=1

            # !!!! knowledge_type = 1 is of type FACT --> if you need to log INSTANCE, knowledge_type = 0 
            # you may remove the first condition if you want to include static knowledge/typing into graph
            if (res["knowledgeItem"]["knowledge_type"]==1) and (res["update_type"] != "add_goal"): 
                subject, object, label = process_knowledgeitem(res["knowledgeItem"])
                if res["update_type"] == "add_knowledge":
                    current_graph.add_edge(subject,object,
                    label = label)
                elif res["update_type"] == "remove_knowledge":
                    current_graph.remove_edge(subject,object)
        graph_list.append(current_graph) # add the last graph


        return graph_list

def process_knowledgeitem(knowledgeitem):
    subject = knowledgeitem["values"][0]["value"]
    object = knowledgeitem["values"][-1]["value"]
    label = attribute_predicate_mapping(knowledgeitem["attribute_name"])
    return subject, object, label

def attribute_predicate_mapping(predicate):
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
    if predicate == "wp_checked_out":
        return "Robot tried finding a box at"
        

class RosplanDialogueManager(DialogueKBManager):
    def __init__(self, mpnn, convqa, triples2text, session_num, top_p=0.5, db_string="scenarios_db", port=27017) -> None:
        client = pymongo.MongoClient(f"mongodb://localhost:{port}/")
        self.db = client[db_string]
        self.session_num = session_num
        self.ActionDialogueManager = RPActionDialogueManager(mpnn, convqa, triples2text, session_num, top_p=top_p, db_string=db_string, port=port)
        self.action_kbs = self.ActionDialogueManager.kbs
        self.selected_action = None

        super().__init__({}, mpnn, convqa, triples2text, top_p=top_p)
        self.kbs.append(self.action_kbs[-1])
        self.logs["base_graphs"].append(self.action_kbs[-1])

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
        subject = None
        for res in plan_results:
            subject = self.naturalise_plan_action(res["plan_action"])
            action_id = f"Action {i}"
            print(f"{action_id}: {subject} ")
            i += 1
            # **************** plan_graph generation **********************************
            label = "is part of"
            plan_id = "Plan 1"
            plan_graph.add_edge(subject,action_id,label="is")
            plan_graph.add_edge(subject,plan_id, label = label)
            if (node_before == "start of plan"):
                plan_graph.add_edge(node_before, subject, label = "is")
            else:
                plan_graph.add_edge(node_before, subject, label = "before")
                plan_graph.add_edge(subject, node_before, label = "after")
            node_before = subject

        if subject is None:
            raise ValueError("No plan found in database")    
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
        # graph for pick and place result of the respective action with failure message
        # TODO: link up with action_id 
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
         # success and failure count, as well as box count (how many boxes found in perception task)
         # NOTE: no action_id needed since it is a final count
        successcount_collection = self.db["successcounts"]
        successcount_results = successcount_collection.find({"SESSION_NUM":self.session_num})
        successcount_graph = MultiDiGraph()
        for res in successcount_results:
            
            subject = res["count_name"]
            label = "is"
            
            object = str(res["count"])
            successcount_graph.add_edge(subject,object, label = label)
        return successcount_graph
    
    def _process_static_knowledge_graph(self) -> MultiDiGraph:
         # static knowledge items for labeling objects and action names with their types
         # TODO: add or remove table as waypoint?
        ki_collection = self.db["knowledgeitems"]
        static_ki_results = ki_collection.find({"SESSION_NUM":self.session_num,"update_type":"add_instance"})
        static_ki_graph = MultiDiGraph()
        for res in static_ki_results:
            
            subject = res["knowledgeItem"]["instance_name"]
            label = "is a"
            object = res["knowledgeItem"]["instance_type"]
            static_ki_graph.add_edge(subject,object, label = label)
        return static_ki_graph

    def _process_conditions_knowledge_graph(self) -> MultiDiGraph:
         # preconditions for each action, can be "positive" or "negative" knowledge
         # TODO: link up via action_id with the respective action the condition belongs to
        ki_collection = self.db["knowledgeitems"]
        pos_conditions_results = ki_collection.find({"SESSION_NUM":self.session_num,"update_type": "positive_condition"})
        neg_conditions_results = ki_collection.find({"SESSION_NUM":self.session_num,"update_type": "negative_condition"})
        conditions_graph = MultiDiGraph()
        for res in pos_conditions_results:
            action_id = "Action "+str(res["action_id"])
            condition_subject, condition_object, condition_label = process_knowledgeitem(res["knowledgeItem"])
            condition_summary = condition_subject + " " + condition_label + " " + condition_object
            conditions_graph.add_edge(condition_summary, condition_object, label = "involves")
            conditions_graph.add_edge(condition_summary, condition_subject, label = "involves")
            label = "required that"
            conditions_graph.add_edge(action_id,condition_summary, label = label)
        for res in neg_conditions_results:
            action_id = "Action "+str(res["action_id"])
            condition_subject, condition_object, condition_label = process_knowledgeitem(res["knowledgeItem"])
            condition_summary = condition_subject + "," + condition_label + "," + condition_object
            conditions_graph.add_edge(condition_summary, condition_object, label = "involves that NOT")
            conditions_graph.add_edge(condition_summary, condition_subject, label = "involves that NOT")
            label = "required that NOT"
            conditions_graph.add_edge(action_id,condition_summary, label = label)
        return conditions_graph

    def _process_goals_graph(self) -> MultiDiGraph:
         # graph for the goals of the plan, can be used to JUSTIFY a specific action
        ki_collection = self.db["knowledgeitems"]
        goals_results = ki_collection.find({"SESSION_NUM":self.session_num,"update_type":"add_goal"})
        goals_graph = MultiDiGraph()

        # TODO: link up with action_ids : e.g. action 1 (pick green wp1) has goal (place green wp3)
        # NOTE: action_id is always -1 for these items
        # TODO: decide whether we use this or just add an edge to each action in the "PLAN" with its respective end goal action
        for res in goals_results:
            goal_subject, goal_object, goal_label = process_knowledgeitem(res["knowledgeItem"])
            goal_summary = goal_subject + " " + goal_label + " " + goal_object
            goals_graph.add_edge(goal_summary, goal_object, label = "involves")
            goals_graph.add_edge(goal_summary, goal_subject, label = "involves")
            goals_graph.add_edge(goal_summary, "gool", label = "is a")
            action_id = "Action "+str(res["action_id"])
            label = "has goal"
            goals_graph.add_edge(action_id,goal_summary, label = label)
            # goals_graph.add_edge(goal_subject, goal_object, label = goal_label)
            
        return goals_graph 

    def initialise_kbs(self, **knowledge_base_args) -> list[MultiDiGraph]:

        plan_graph = self._process_plan_db()

        #next line of code is no longer needed 
        #action_query = {"$lte":int(input("Enter number of action you would like to know about : "))}  #$lte: means less than or equal, so we generate a knowledge graph using all the items UP UNTIL that action (included)

        task_result_graph = self._process_pick_and_place_db()
        successcount_graph = self._process_success_count_graph()
        static_ki_graph = self._process_static_knowledge_graph()
        conditions_graph = self._process_conditions_knowledge_graph()
        goals_graph = self._process_goals_graph()

        return [plan_graph, task_result_graph, successcount_graph, static_ki_graph, conditions_graph, goals_graph]
        # return [cosmpose_all([task_result_graph, successcount_graph, plan_graph, static_ki_graph, conditions_graph, goals_graph])]
    
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
        if action[0] == "move":
            return "move from "+action[2]+" to "+action[3]
        elif action[0] == "grasp":
            return "pick "+action[2]+" from table at "+action[3]
        elif action[0] == "place":
            return "place from "+action[2]+" onto table at "+action[3]
        elif action[0] == "perceive":
            return "check for box at table "+action[2]
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
    # convqa = ConvQASystem("./dialogsystem/trained_models/convqa")
    convqa = lambda x: x
    # triples2text = PretrainedTriples2TextSystem()
    triples2text = lambda x: x
    mpnn = LightningKGQueryMPNN.load_from_checkpoint("dialogsystem/trained_models/gqanew.ckpt")
    mpnn.avg_pooling=False
    rdm = RosplanDialogueManager(mpnn,convqa,triples2text, session_num=2, db_string="scenarios_db_2")
    quit = False
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