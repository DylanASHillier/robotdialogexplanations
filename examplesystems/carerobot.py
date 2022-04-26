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
import math
import numpy

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
        object_graph = DiGraph()
        times = []
        for output in output:
            _, statement_array = self.ebb_interface.state_observations_list(output["observations_coll"], statement_array)
            print("\n\n\n\n lets a go")
            observations = statement_array[0]
            object_observations = observations[3]
            time = observations[1]
            times.append(time)
            location_dict = {}
            object_graph.add_edge("robot", "you", label="is_known_as")
            for observation in object_observations:
                base_info = observation["base_info"]
                obj_location = observation["location_of_object"]["position"]
                robot_location = base_info["location"]["position"]
                robot_orientation = base_info["location"]["orientation"]
                robot_orientation = translate_quaternion_to_euler(robot_orientation["x"],robot_orientation["y"],robot_orientation["z"],robot_orientation["w"])
                obj_type = observation["obj_type"]
                obj_id = f"object: {observation['id_of_object']}"
                obj_colour = observation["obj_colour"]
                prev_identified = observation["obj_previously_identified"]
                location_dict[obj_id]=obj_location
                obj_relation_to_robot = relations_from_robot_perspective(robot_orientation, obj_location, robot_location)
                object_graph.add_edge(obj_id, "robot", label=obj_relation_to_robot[0])
                object_graph.add_edge(obj_id, "robot", label=obj_relation_to_robot[1])
                object_graph.add_edge(obj_id, "robot", label=obj_relation_to_robot[2])
                object_graph.add_edge(obj_id, obj_type, label="is")
                object_graph.add_edge(obj_id, obj_colour, label="has_colour")
                if prev_identified:
                    object_graph.add_edge(obj_id, str(time), label="reobserved at")
                else:
                    object_graph.add_edge(obj_id, str(time), label="first observed at")
            # compute positional relations between observed objects
            for obj_id in location_dict:
                for other_obj_id in location_dict:
                    if obj_id != other_obj_id:
                        obj_location = location_dict[obj_id]
                        other_obj_location = location_dict[other_obj_id]
                        obj_relation_to_other = relations_from_robot_perspective(robot_orientation, obj_location, other_obj_location)
                        object_graph.add_edge(obj_id, other_obj_id, label=obj_relation_to_other[0])
                        object_graph.add_edge(obj_id, other_obj_id, label=obj_relation_to_other[1])
                        object_graph.add_edge(obj_id, other_obj_id, label=obj_relation_to_other[2])

            # compute temporal relations between observations and add to graph
            threshold = 0.5
            for t1, t2 in zip(times, times):
                if t1 != t2:
                    delta_time = t2 - t1
                    if delta_time.total_seconds > threshold:
                        object_graph.add_edge(str(t2), str(t1), label="is_after")
                        object_graph.add_edge(str(t1), str(t2), label="is_before")
                    elif delta_time.total_seconds < - threshold:
                        object_graph.add_edge(str(t1), str(t2), label="is_after")
                        object_graph.add_edge(str(t2), str(t1), label="is_before")
                    else:
                        object_graph.add_edge(str(t1), str(t2), label="is_same_time_as")
                        object_graph.add_edge(str(t2), str(t1), label="is_same_time_as")
                    

        # remove location attributes from graph
        print(object_graph)
        return [object_graph]


if __name__ == '__main__':
    convqa = lambda x: "i'm not sure"
    triples2text = Triples2TextSystem.load_from_checkpoint("dialogsystem/trained_models/t2t.ckpt").load_from_hf_checkpoint("./dialogsystem/trained_models/t2t/t2ttrained")
    # print(triples2text)
    # print(triples2text(["graph, is used in, China"]))
    mpnn = LightningKGQueryMPNN.load_from_checkpoint("trained_models/newmeddim.ckpt")
    mpnn.k = 10
    rdm = RobotDialogueManager(mpnn,convqa,triples2text)
    rdm.question_and_response("where is the bottle in relation to the robot")

    # orientation = [1,0,0]
    # vector = [1,1,0]
    # print(rotate_vector(orientation, vector))