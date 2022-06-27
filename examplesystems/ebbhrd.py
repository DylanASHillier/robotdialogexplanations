import sys
from os.path import dirname
from networkx import DiGraph
sys.path.append(dirname("./dialogsystem"))
from dialogsystem.models.convqa import ConvQASystem
from dialogsystem.models.kgqueryextract import LightningKGQueryMPNN
from dialogsystem.kb_dial_management.kb_dial_manager import DialogueKBManager
from dialogsystem.models.triples2text import Triples2TextSystem
sys.path.append(dirname("../ebbhrd_hrd/src"))
from src.EBB_Offline_Interface import EBB_interface
import math
import numpy
from networkx import compose_all

### Dialogue example
'''
[{'session_num': 52, 'dialogue_speak_and_listen': [{'_id': ObjectId('6261bb34d0dbf8b6bf86c6c0'), 'human_response': '[NONE]', 'robot_question': 'Hi, my name is Bam Bam.', 'base_info': {'ros_timestamp': 1.6505720848924406e+18, 'global_timestamp': datetime.datetime(2022, 4, 21, 21, 14, 44, 892000)}, 'session_num': 52, 'entry_uid': 79}, {'_id': ObjectId('6261bb3ed0dbf8b6bf86c73a'), 'human_response': '[NONE]', 'robot_question': 'Hello', 'base_info': {'ros_timestamp': 1.6505720947918707e+18, 'global_timestamp': datetime.datetime(2022, 4, 21, 21, 14, 54, 791000)}, 'session_num': 52, 'entry_uid': 201}, {'_id': ObjectId('6261bb3fd0dbf8b6bf86c741'), 'human_response': '[NONE]', 'robot_question': 'What is your name?', 'base_info': {'ros_timestamp': 1.6505720957920776e+18, 'global_timestamp': datetime.datetime(2022, 4, 21, 21, 14, 55, 792000)}, 'session_num': 52, 'entry_uid': 208}, {'_id': ObjectId('6261bb42d0dbf8b6bf86c764'), 'human_response': 'Matthew', 'robot_question': 'What is your name?', 'base_info': {'ros_timestamp': 1.650572098346359e+18, 'global_timestamp': datetime.datetime(2022, 4, 21, 21, 14, 58, 346000)}, 'session_num': 52, 'entry_uid': 243}, {'_id': ObjectId('6261bb43d0dbf8b6bf86c776'), 'human_response': '[NONE]', 'robot_question': 'What do you want me to pick up?', 'base_info': {'ros_timestamp': 1.6505720999908145e+18, 'global_timestamp': datetime.datetime(2022, 4, 21, 21, 14, 59, 990000)}, 'session_num': 52, 'entry_uid': 261}, {'_id': ObjectId('6261bb4ed0dbf8b6bf86c7e3'), 'human_response': 'Please pick up the potted plant.', 'robot_question': 'What do you want me to pick up?', 'base_info': {'ros_timestamp': 1.650572110565169e+18, 'global_timestamp': datetime.datetime(2022, 4, 21, 21, 15, 10, 565000)}, 'session_num': 52, 'entry_uid': 370}, {'_id': ObjectId('6261bb87d0dbf8b6bf86ca1f'), 'human_response': '[NONE]', 'robot_question': "Hi, Matthew, I've brought you the potted_plant", 'base_info': {'ros_timestamp': 1.6505721673928028e+18, 'global_timestamp': datetime.datetime(2022, 4, 21, 21, 16, 7, 392000)}, 'session_num': 52, 'entry_uid': 942}, {'_id': ObjectId('6261bb9ed0dbf8b6bf86cadb'), 'human_response': '[NONE]', 'robot_question': 'It looks like my job here is done. Have a nice day!', 'base_info': {'ros_timestamp': 1.6505721905940237e+18, 'global_timestamp': datetime.datetime(2022, 4, 21, 21, 16, 30, 594000)}, 'session_num': 52, 'entry_uid': 1130}, {'_id': ObjectId('6261bba9d0dbf8b6bf86cb69'), 'human_response': '[NONE]', 'robot_question': "I'm back where I started, woohoo!", 'base_info': {'ros_timestamp': 1.6505722013919053e+18, 'global_timestamp': datetime.datetime(2022, 4, 21, 21, 16, 41, 391000)}, 'session_num': 52, 'entry_uid': 1272}]}]
'''

### State Change example
'''
[{'session_num': 52, 'state_changes_coll': [{'_id': ObjectId('6261bb32d0dbf8b6bf86c6a4'), 'base_info': {'entry_type': 1, 'location': {'orientation': {'w': nan, 'x': nan, 'y': nan, 'z': nan}, 'position': {'x': nan, 'y': nan, 'z': nan}}, 'ros_timestamp': 1.6505720824988321e+18, 'global_timestamp': datetime.datetime(2022, 4, 21, 21, 14, 42, 498000)}, 'changing_from': '[INITIAL_STATE]', 'changing_to': 'StoreInitialLocation', 'previous_state_result': 1, 'session_num': 52, 'entry_uid': 51}, {'_id': ObjectId('6261bb32d0dbf8b6bf86c6ab'), 'base_info': {'entry_type': 1, 'location': {'orientation': {'w': -0.7275106985634158, 'x': -0.672863414515885, 'y': 0.09277567744269441, 'z': -0.09682810829941332}, 'position': {'x': 1.2374735311165468, 'y': 0.9148515662095564, 'z': -0.6104560585218771}}, 'ros_timestamp': 1.650572082813442e+18, 'global_timestamp': datetime.datetime(2022, 4, 21, 21, 14, 42, 813000)}, 'changing_from': 'StoreInitialLocation', 'changing_to': 'Intro', 'previous_state_result': 1, 'session_num': 52, 'entry_uid': 57}, {'_id': ObjectId('6261bb34d0dbf8b6bf86c6c1'), 'base_info': {'entry_type': 1, 'location': {'orientation': {'w': -0.7275215636537887, 'x': -0.6728710809839555, 'y': 0.09270930524509705, 'z': -0.09675674402954725}, 'position': {'x': 1.2381815412013066, 'y': 0.914763592239022, 'z': -0.6115145774833226}}, 'ros_timestamp': 1.6505720848964782e+18, 'global_timestamp': datetime.datetime(2022, 4, 21, 21, 14, 44, 896000)}, 'changing_from': 'Intro', 'changing_to': 'SetNavToOperator', 'previous_state_result': 2, 'session_num': 52, 'entry_uid': 80}, {'_id': ObjectId('6261bb34d0dbf8b6bf86c6c2'), 'base_info': {'entry_type': 1, 'location': {'orientation': {'w': -0.7275215636537887, 'x': -0.6728710809839555, 'y': 0.09270930524509705, 'z': -0.09675674402954725}, 'position': {'x': 1.2381815412013066, 'y': 0.914763592239022, 'z': -0.6115145774833226}}, 'ros_timestamp': 1.6505720849003215e+18, 'global_timestamp': datetime.datetime(2022, 4, 21, 21, 14, 44, 900000)}, 'changing_from': 'SetNavToOperator', 'changing_to': 'NavToOperator', 'previous_state_result': 2, 'session_num': 52, 'entry_uid': 81}, {'_id': ObjectId('6261bb3ed0dbf8b6bf86c738'), 'base_info': {'entry_type': 1, 'location': {'orientation': {'w': -0.47605928204368675, 'x': -0.7339712524270723, 'y': 0.4078734863450879, 'z': -0.26132925539898394}, 'position': {'x': 0.7284739273613348, 'y': 1.0817068699381982, 'z': 0.001134553830078211}}, 'ros_timestamp': 1.6505720942054523e+18, 'global_timestamp': datetime.datetime(2022, 4, 21, 21, 14, 54, 205000)}, 'changing_from': 'NavToOperator', 'changing_to': 'EBBHRD_DialogueSys', 'previous_state_result': 2, 'session_num': 52, 'entry_uid': 199}, {'_id': ObjectId('6261bb4ed0dbf8b6bf86c7e7'), 'base_info': {'entry_type': 1, 'location': {'orientation': {'w': -0.47563832445864707, 'x': -0.7333142483739016, 'y': 0.40905353364135766, 'z': -0.26209464712868236}, 'position': {'x': 0.7295575474419371, 'y': 1.0773298346033204, 'z': 0.010854823540857705}}, 'ros_timestamp': 1.6505721106542008e+18, 'global_timestamp': datetime.datetime(2022, 4, 21, 21, 15, 10, 654000)}, 'changing_from': 'EBBHRD_DialogueSys', 'changing_to': 'SetNavToPickUp', 'previous_state_result': 5, 'session_num': 52, 'entry_uid': 374}, {'_id': ObjectId('6261bb4ed0dbf8b6bf86c7e8'), 'base_info': {'entry_type': 1, 'location': {'orientation': {'w': -0.47563832445864707, 'x': -0.7333142483739016, 'y': 0.40905353364135766, 'z': -0.26209464712868236}, 'position': {'x': 0.7295575474419371, 'y': 1.0773298346033204, 'z': 0.010854823540857705}}, 'ros_timestamp': 1.6505721106580815e+18, 'global_timestamp': datetime.datetime(2022, 4, 21, 21, 15, 10, 658000)}, 'changing_from': 'SetNavToPickUp', 'changing_to': 'NavToPickUp', 'previous_state_result': 2, 'session_num': 52, 'entry_uid': 375}, {'_id': ObjectId('6261bb57d0dbf8b6bf86c839'), 'base_info': {'entry_type': 1, 'location': {'orientation': {'w': -0.5320026395294926, 'x': -0.8216927913173961, 'y': 0.17290853357784353, 'z': -0.10907239452142659}, 'position': {'x': 1.4015021432122914, 'y': 1.1921789481997633, 'z': -0.25226148112854946}}, 'ros_timestamp': 1.6505721196619717e+18, 'global_timestamp': datetime.datetime(2022, 4, 21, 21, 15, 19, 661000)}, 'changing_from': 'NavToPickUp', 'changing_to': 'PickUpItem', 'previous_state_result': 2, 'session_num': 52, 'entry_uid': 456}, {'_id': ObjectId('6261bb7bd0dbf8b6bf86c99c'), 'base_info': {'entry_type': 1, 'location': {'orientation': {'w': -0.7017326592889497, 'x': -0.6493648766124855, 'y': 0.20085018130531984, 'z': -0.21343789864719503}, 'position': {'x': 1.5106569688965195, 'y': 0.9357927006246807, 'z': -0.2768120612987401}}, 'ros_timestamp': 1.6505721550376184e+18, 'global_timestamp': datetime.datetime(2022, 4, 21, 21, 15, 55, 37000)}, 'changing_from': 'PickUpItem', 'changing_to': 'SetNavBackToOperator', 'previous_state_result': 2, 'session_num': 52, 'entry_uid': 811}, {'_id': ObjectId('6261bb7bd0dbf8b6bf86c99d'), 'base_info': {'entry_type': 1, 'location': {'orientation': {'w': -0.7017326592889497, 'x': -0.6493648766124855, 'y': 0.20085018130531984, 'z': -0.21343789864719503}, 'position': {'x': 1.5106569688965195, 'y': 0.9357927006246807, 'z': -0.2768120612987401}}, 'ros_timestamp': 1.6505721550408604e+18, 'global_timestamp': datetime.datetime(2022, 4, 21, 21, 15, 55, 40000)}, 'changing_from': 'SetNavBackToOperator', 'changing_to': 'NavBackToOperator', 'previous_state_result': 2, 'session_num': 52, 'entry_uid': 812}, {'_id': ObjectId('6261bb84d0dbf8b6bf86c9ef'), 'base_info': {'entry_type': 1, 'location': {'orientation': {'w': -0.4751262575103582, 'x': -0.7325183166274698, 'y': 0.41047937840075766, 'z': -0.26301831711343954}, 'position': {'x': 0.7305655243610082, 'y': 1.0744670759732227, 'z': 0.005021535031709101}}, 'ros_timestamp': 1.6505721642478556e+18, 'global_timestamp': datetime.datetime(2022, 4, 21, 21, 16, 4, 247000)}, 'changing_from': 'NavBackToOperator', 'changing_to': 'SpeakToOperator', 'previous_state_result': 2, 'session_num': 52, 'entry_uid': 894}, {'_id': ObjectId('6261bb87d0dbf8b6bf86ca20'), 'base_info': {'entry_type': 1, 'location': {'orientation': {'w': -0.4752150036833476, 'x': -0.7326568188008672, 'y': 0.41023211740624066, 'z': -0.2628579387943626}, 'position': {'x': 0.729303072412566, 'y': 1.0763300896289, 'z': 0.0008935403856124569}}, 'ros_timestamp': 1.6505721673958218e+18, 'global_timestamp': datetime.datetime(2022, 4, 21, 21, 16, 7, 395000)}, 'changing_from': 'SpeakToOperator', 'changing_to': 'GiveItemBack', 'previous_state_result': 2, 'session_num': 52, 'entry_uid': 943}, {'_id': ObjectId('6261bb9ad0dbf8b6bf86cabc'), 'base_info': {'entry_type': 1, 'location': {'orientation': {'w': -0.6428303919381637, 'x': -0.5933731273122679, 'y': 0.3305432478318153, 'z': -0.3541448577871497}, 'position': {'x': 0.7303817153651284, 'y': 0.9291952782657272, 'z': -0.4009910641085735}}, 'ros_timestamp': 1.6505721867581535e+18, 'global_timestamp': datetime.datetime(2022, 4, 21, 21, 16, 26, 758000)}, 'changing_from': 'GiveItemBack', 'changing_to': 'ThankOperator', 'previous_state_result': 2, 'session_num': 52, 'entry_uid': 1099}, {'_id': ObjectId('6261bb9ed0dbf8b6bf86cadc'), 'base_info': {'entry_type': 1, 'location': {'orientation': {'w': -0.6428027729815262, 'x': -0.5933473490119393, 'y': 0.3305895193802273, 'z': -0.35419498605106475}, 'position': {'x': 0.7293009541175949, 'y': 0.9285261529379989, 'z': -0.40016980565154636}}, 'ros_timestamp': 1.6505721905986575e+18, 'global_timestamp': datetime.datetime(2022, 4, 21, 21, 16, 30, 598000)}, 'changing_from': 'ThankOperator', 'changing_to': 'SetNavToStart', 'previous_state_result': 2, 'session_num': 52, 'entry_uid': 1131}, {'_id': ObjectId('6261bb9ed0dbf8b6bf86cadd'), 'base_info': {'entry_type': 1, 'location': {'orientation': {'w': -0.6428027729815262, 'x': -0.5933473490119393, 'y': 0.3305895193802273, 'z': -0.35419498605106475}, 'position': {'x': 0.7293009541175949, 'y': 0.9285261529379989, 'z': -0.40016980565154636}}, 'ros_timestamp': 1.6505721906032108e+18, 'global_timestamp': datetime.datetime(2022, 4, 21, 21, 16, 30, 603000)}, 'changing_from': 'SetNavToStart', 'changing_to': 'NavToStart', 'previous_state_result': 2, 'session_num': 52, 'entry_uid': 1132}, {'_id': ObjectId('6261bba7d0dbf8b6bf86cb51'), 'base_info': {'entry_type': 1, 'location': {'orientation': {'w': -0.538458905606037, 'x': -0.8319856464352557, 'y': 0.11347819178590093, 'z': -0.07060163658344187}, 'position': {'x': 1.2289117243022094, 'y': 1.1605530941784419, 'z': -0.19213763203051892}}, 'ros_timestamp': 1.6505721993104566e+18, 'global_timestamp': datetime.datetime(2022, 4, 21, 21, 16, 39, 310000)}, 'changing_from': 'NavToStart', 'changing_to': 'Finish', 'previous_state_result': 2, 'session_num': 52, 'entry_uid': 1248}]}]
'''

state_change_result_dict = {
        0 : "NULL",
        1 : "UNKNOWN",
        2 : "SUCCESS",
        3 : "TASK_SUCCESS",
        4 : "CLARIFY",
        5 : "SUCCESS_PICK_UP",
        16 : "FAILURE",
        32 : "REPEATED_FAILURE",
        64 : "TASK_FAILURE"
    }

def translate_quaternion_to_euler(x,y,z,w):
    ysqr = y * y
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + ysqr)
    X = math.radians(math.atan2(t0, t1))
    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    Y = math.radians(math.asin(t2))
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (ysqr + z * z)
    Z = math.radians(math.atan2(t3, t4))
    return [X, Y, Z]

def calculate_relative_position(object_a_position, object_b_position):
    '''
    compares the differences between two objects across each axis (direction from b to a)
    arguments:
        object_a_position: [x,y,z]
        object_b_position: [x,y,z]
    returns:
        [x,y,z]
    '''
    return [object_a_position[0] - object_b_position[0], object_a_position[1] - object_b_position[1], object_a_position[2] - object_b_position[2]]

def rotate_vector(robot_orientation, vector):
    '''
    (assumes that the z axis is the upwards direction -- will not work if the robot falls over)
    reorient a vector with the x axis being how far in front the object is, the y axis being how far to the left the object is, and the z axis being how far up the object is
    arguments:
        robot_orientation: [x,y,z]
        vector: [x,y,z]
    returns:
        [x,y,z]
    '''
    # set z axis of robot_orientation to be fixed
    robot_orientation[2] = 0
    # normalise vectors:
    robot_orientation = numpy.array(robot_orientation)
    robot_orientation = robot_orientation / numpy.linalg.norm(robot_orientation)
    vector = numpy.array(vector)
    vector = vector / numpy.linalg.norm(vector)
    robot_x_axis = robot_orientation
    robot_z_axis = numpy.array([0,0,1])
    robot_y_axis = numpy.cross(robot_z_axis, robot_x_axis)
    # rotate vector
    rotation_matrix = numpy.array([robot_x_axis, robot_y_axis, robot_z_axis])
    rotated_vector = numpy.dot(rotation_matrix, vector)
    return rotated_vector
    
def relations_from_robot_perspective(robot_orientation, object_a, object_b):
    '''
    returns a tuple of whether a is in front or behind b, left of or right of, and above or below
    arguments:
        robot_orientation: [x,y,z]
        object_a: {x,y,z}
        object_b: {x,y,z}
    returns:
        (str,str,str)
    '''
    object_a = [object_a["x"], object_a["y"], object_a["z"]]
    object_b = [object_b["x"], object_b["y"], object_b["z"]]
    # calculate relative position
    relative_position = calculate_relative_position(object_a, object_b)
    # rotate vector
    rotated_relative_position = rotate_vector(robot_orientation, relative_position)
    # calculate relations
    frontedness = "behind" if rotated_relative_position[0] < 0 else "in front of"
    leftness = "right of" if rotated_relative_position[1] < 0 else "left of"
    upness = "below" if rotated_relative_position[2] < 0 else "above"
    return (frontedness, leftness, upness)

class RobotDialogueManager(DialogueKBManager):
    def __init__(self, mpnn, convqa, triples2text,db_port = 27017, knowledge_base_args={'session':(2022,4,26)}) -> None:
        self.ebb_interface = EBB_interface(port=db_port)
        self.observed_objects = set()
        self.observed_dialogues = set()
        self.observed_state_changes = set()
        self.map_to_old_id = {}
        super().__init__(knowledge_base_args, mpnn, convqa, triples2text)


    def initialise_kbs(self, session_date) -> list[DiGraph]:
        """
        session_date: <(int,int,int)> (yyyy,mm,dd) 
        """
        sessions = self.ebb_interface.getSessionNums_date(*session_date)
        print(f"choose a session from: {sessions}")
        session_num = int(input("session number: "))
        outputs = self.ebb_interface.getCollectionFromEBB(["observations_coll"],[session_num])
        # print(output)
        statement_array =[]
        object_graph = DiGraph()
        dialogue_graph = DiGraph()
        state_graph = DiGraph()
        time_graph = DiGraph()
        times = []

        for output in outputs:
            ### OBSERVATIONS
            
            _, statement_array = self.ebb_interface.state_observations_list(output["observations_coll"], statement_array)
            # print("\n\n\n\n lets a go")
            # print(statement_array)
            assert(len(statement_array))>0
            observations = statement_array[0]
            object_observations = observations[3]
            obs_location_dict = {}
            object_graph.add_edge("robot", "you", label="is_known_as")
            object_graph.add_edge("you","robot", label="is_known_as")
            for observation in object_observations:
                base_info = observation["base_info"]
                time = base_info["global_timestamp"]
                times.append(time)
                obj_location = observation["location_of_object"]["position"]
                robot_location = base_info["location"]["position"]
                robot_orientation = base_info["location"]["orientation"]
                robot_orientation = translate_quaternion_to_euler(robot_orientation["x"],robot_orientation["y"],robot_orientation["z"],robot_orientation["w"])
                obj_type = observation["obj_type"]
                old_obj_id = f"object: {observation['id_of_object']}"
                self.observed_objects.add(old_obj_id)
                obj_id = f"{obj_type} no. {len(self.observed_objects)}"
                self.map_to_old_id[obj_id] = old_obj_id
                obj_colour = observation["obj_colour"]
                prev_identified = observation["obj_previously_identified"]
                obs_location_dict[obj_id]=obj_location
                obj_relation_to_robot = relations_from_robot_perspective(robot_orientation, obj_location, robot_location)
                object_graph.add_edge(obj_id, "robot", label=obj_relation_to_robot[0])
                object_graph.add_edge(obj_id, "robot", label=obj_relation_to_robot[1])
                object_graph.add_edge(obj_id, "robot", label=obj_relation_to_robot[2])
                object_graph.add_edge(obj_id, obj_type, label="is")
                object_graph.add_edge(obj_type, obj_id, label="is")
                object_graph.add_edge(obj_id, obj_colour, label="has colour")
                if prev_identified:
                    object_graph.add_edge(obj_id, str(time), label="reobserved at")
                else:
                    object_graph.add_edge(obj_id, str(time), label="first observed at")

            
            # compute positional relations between observed objects
            for obj_id in obs_location_dict:
                for other_obj_id in obs_location_dict:
                    if obj_id != other_obj_id:
                        obj_location = obs_location_dict[obj_id]
                        other_obj_location = obs_location_dict[other_obj_id]
                        obj_relation_to_other = relations_from_robot_perspective(robot_orientation, obj_location, other_obj_location)
                        object_graph.add_edge(obj_id, other_obj_id, label=obj_relation_to_other[0])
                        object_graph.add_edge(obj_id, other_obj_id, label=obj_relation_to_other[1])
                        object_graph.add_edge(obj_id, other_obj_id, label=obj_relation_to_other[2])
            for obj_id in obs_location_dict:
                object_graph.add_edge(obj_id, "object", label="is an")
                object_graph.add_edge("object", obj_id, label="has member")
            print(object_graph.edges(data=True))

            ### DIALOGUE
            
            statement_array = self.ebb_interface.getCollectionFromEBB(["dialogue_speak_and_listen"],[session_num])
            # parse dialogue
            assert len(statement_array)>0
            dialogue = statement_array[0]
            robot_said_prev = None
            human_said_prev = None
            turn_prev_id = None

            for turn in dialogue["dialogue_speak_and_listen"]:
                robot_said = turn["robot_question"]
                human_said = turn["human_response"]
                self.observed_dialogues.add(turn["entry_uid"])
                time = turn["base_info"]["global_timestamp"]
                times.append(time)

                turn_id = f"turn {len(self.observed_dialogues)}"
                # add to graph
                if robot_said != "[NONE]":
                    dialogue_graph.add_edge("robot", robot_said, label="said")
                    dialogue_graph.add_edge(robot_said, f"turn {len(self.observed_dialogues)}", label="said during")
                    dialogue_graph.add_edge(turn_id, robot_said, label="said during")

                if human_said != "[NONE]":
                    dialogue_graph.add_edge("human", human_said, label="said")
                    dialogue_graph.add_edge(human_said, f"turn {len(self.observed_dialogues)}", label="said during")
                    dialogue_graph.add_edge(turn_id, human_said, label="said during")
                    dialogue_graph.add_edge(turn_id, str(time), label="said at")
                    dialogue_graph.add_edge(human_said, robot_said, label="said in response to")
                    if robot_said_prev is not None:
                        dialogue_graph.add_edge(robot_said_prev, robot_said, label="said after")
                        dialogue_graph.add_edge(human_said_prev, human_said, label="said after")
                        dialogue_graph.add_edge(robot_said, human_said_prev, label="response to")

                if turn_prev_id is not None:
                    dialogue_graph.add_edge(turn_prev_id, turn_id, label="followed by")
                    dialogue_graph.add_edge(turn_id, turn_prev_id, label="preceded by")
                else:
                    dialogue_graph.add_edge("start", turn_id, label="is at")
                    dialogue_graph.add_edge(turn_id, "first", label="is")
                    dialogue_graph.add_edge("start", "first", label="is")
                    dialogue_graph.add_edge("first", "start", label="is")


                
                robot_said_prev = robot_said
                human_said_prev = human_said
                turn_prev_id = turn_id
                
            ### STATE CHANGES
            
            statement_array = self.ebb_interface.getCollectionFromEBB(["state_changes_coll"],[session_num])
            # parse state changes
            assert len(statement_array)>0
            state_changes = statement_array[0]
            for state_change in state_changes["state_changes_coll"]:
                self.observed_state_changes.add(state_change["entry_uid"])

                prev_state = state_change["changing_from"]
                cur_state = state_change["changing_to"]
                prev_state_result = state_change_result_dict[state_change["previous_state_result"]]

                state_graph.add_edge(cur_state, "state", label="is")
                state_graph.add_edge("state", cur_state, label="is")
                state_graph.add_edge(prev_state, "state", label="is")
                state_graph.add_edge("state", prev_state, label="is")
                
                state_change_id = f"{len(self.observed_state_changes)} state change"

                state_graph.add_edge(state_change_id, prev_state_result, label="has result")

                state_graph.add_edge(state_change_id,prev_state,label="changed from")
                state_graph.add_edge(state_change_id,cur_state,label="changed to")
                
                state_graph.add_edge(prev_state,state_change_id,label="changed in")
                state_graph.add_edge(cur_state,state_change_id,label="changed in")

                
                time = state_change["base_info"]["global_timestamp"]
                times.append(time)
                
                state_graph.add_edge(state_change_id, str(time), label="change during")
                state_graph.add_edge(str(time), state_change_id, label="had state change")
                state_graph.add_edge(prev_state_result, str(time), label="occured at")
                state_graph.add_edge(str(time), prev_state_result, label="had state result")



            # compute temporal relations between temporal events and add to graph
            threshold = 0.5

            # sort times and extract relations
            times.sort()

            for i in range(len(times)-1):
                time = times[i]
                windows = [2^i for i in range(1,10)]
                windows = filter(lambda x: x< len(times)-i,windows)
                other_times = [times[window] for window in windows]
                for other_time in other_times:
                    delta_time = (time - other_time).total_seconds()
                    # if delta_time > threshold:
                    time_graph.add_edge(str(time), str(other_time), label="after")
                    time_graph.add_edge(str(other_time), str(time), label="before")
                    # else:
                    #     time_graph.add_edge(str(time), str(other_time), label="similar time to")
                    #     time_graph.add_edge(str(other_time), str(time), label="similar time to")

        

        # remove location attributes from graph
        return [
            # compose_all(
            # [
            # object_graph,
            dialogue_graph,
            # state_graph,
            # time_graph
        # ])
        ]

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
    rdm = RobotDialogueManager(mpnn,convqa,triples2text)
    quit = False
    # rdm.triples2text = lambda x: x
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